"""PE-Core embeddings from bounded NVDEC batches, with optional timestamp sampling."""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from src.utils.gpu_video import CudaVideoReader, pe_preprocess_cuda
from src.utils.frame_sampling import save_embedding_cache, filter_sample_plan


def encode_cuda_video_frames(model, video_path: Path, output_path: Path,
                             batch_size: int, image_size: int = 336,
                             sample_plan: dict | None = None,
                             decode_batch_size: int | None = None,
                             quality_scenes: np.ndarray | None = None,
                             min_hist_bins: int = 5) -> np.ndarray | dict:
    if batch_size < 1:
        raise ValueError('Embedding batch_size must be positive')
    if sample_plan is not None and quality_scenes is not None:
        return encode_sampled_with_quality(model, video_path, output_path, batch_size,
                                           image_size, sample_plan, quality_scenes,
                                           decode_batch_size or batch_size, min_hist_bins)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features = None
    written = 0
    pending = []
    pending_count = 0
    source_seen = 0
    with CudaVideoReader(video_path, decode_batch_size or batch_size) as reader, torch.inference_mode():
        indexes = sample_plan['frame_indexes'] if sample_plan is not None else None
        count = len(indexes) if indexes is not None else reader.total
        if sample_plan is not None and reader.total != sample_plan['source_total_frames']:
            raise RuntimeError('NVDEC frame count differs from the packet PTS timeline')
        bar = tqdm(total=count, desc=f'Embed {video_path.name}', unit='sample' if indexes is not None else 'frame')

        def encode(images):
            nonlocal features,written
            with torch.autocast('cuda', dtype=torch.float16):
                embedding = model.encode_image(images)
            batch = F.normalize(embedding.float(), dim=-1).cpu().numpy()
            if not np.isfinite(batch).all():
                raise RuntimeError('PE-Core produced nonfinite embeddings')
            if features is None:
                features = np.empty((count,batch.shape[1]),dtype=np.float32)
            end = written+len(batch)
            if end > count:
                raise RuntimeError('Encoded more frames than the sampling plan')
            features[written:end] = batch
            written = end
            bar.update(len(batch))

        try:
            for start,rgb in reader:
                if start != source_seen:
                    raise RuntimeError('Frame order changed during embedding')
                source_seen += len(rgb)
                if indexes is not None:
                    left,right = np.searchsorted(indexes,[start,start+len(rgb)])
                    if left == right:
                        continue
                    positions = torch.as_tensor(indexes[left:right]-start,device=rgb.device)
                    rgb = rgb.index_select(0,positions)
                pending.append(pe_preprocess_cuda(rgb,image_size))
                pending_count += len(rgb)
                if pending_count >= batch_size:
                    images = torch.cat(pending)
                    pending = []
                    while len(images) >= batch_size:
                        encode(images[:batch_size])
                        images = images[batch_size:]
                    pending_count = len(images)
                    if pending_count:
                        pending.append(images.clone())
                    del images
            if pending_count:
                encode(torch.cat(pending))
        finally:
            pending.clear()
            bar.close()
    if features is None or written != len(features):
        raise RuntimeError('Incomplete embeddings')
    result = features if sample_plan is None else dict(sample_plan,features=features)
    save_embedding_cache(result,output_path)
    return result


def encode_sampled_with_quality(model, video_path, output_path, batch_size, image_size,
                                plan, scenes, decode_batch_size, min_hist_bins):
    """Quality-gate and PE-embed samples during one bounded NVDEC pass."""
    from src.keyframe_extraction.models.quality import score_batch
    indexes = plan['frame_indexes']
    scenes = np.asarray(scenes, dtype=np.int64).reshape(-1, 2)
    selected = np.zeros(len(indexes), dtype=bool)
    sharpness_scores = np.empty(len(indexes), dtype=np.float32)
    features = None
    pending_images, pending_rows = [], []
    shot = 0
    shot_seen = False
    shot_has_valid = False
    best_nonflat_score = -np.inf
    best_nonflat_row, best_nonflat_frame = None, None
    fallbacks = 0
    dropped_shots = 0
    kept_scenes = []

    def flush():
        nonlocal features
        if not pending_rows:
            return
        images = torch.cat(pending_images)
        with torch.autocast('cuda', dtype=torch.float16, enabled=images.is_cuda):
            embedding = model.encode_image(images)
        batch = F.normalize(embedding.float(), dim=-1).cpu().numpy()
        if not np.isfinite(batch).all() or len(batch) != len(pending_rows):
            raise RuntimeError('Invalid PE-Core embedding batch')
        if features is None:
            features = np.empty((len(indexes), batch.shape[1]), dtype=np.float32)
        features[pending_rows] = batch
        pending_rows.clear()
        pending_images.clear()

    def enqueue(rows, frames):
        if not rows:
            return
        images = pe_preprocess_cuda(torch.stack(frames), image_size)
        for row, image in zip(rows, images):
            selected[row] = True
            pending_rows.append(row)
            pending_images.append(image[None])
            if len(pending_rows) >= batch_size:
                flush()

    def finish_shot():
        nonlocal shot_seen, shot_has_valid
        nonlocal best_nonflat_score, best_nonflat_row, best_nonflat_frame
        nonlocal fallbacks, dropped_shots
        if not shot_seen:
            raise RuntimeError('Shot contains no timestamp-sampled frame')
        if not shot_has_valid:
            if best_nonflat_frame is None:
                dropped_shots += 1
            else:
                enqueue([best_nonflat_row], [best_nonflat_frame])
                fallbacks += 1
        if shot_has_valid or best_nonflat_frame is not None:
            kept_scenes.append(scenes[shot].copy())
        shot_seen = False
        shot_has_valid = False
        best_nonflat_score, best_nonflat_row, best_nonflat_frame = -np.inf, None, None

    with CudaVideoReader(video_path, decode_batch_size) as reader, torch.inference_mode(), \
            tqdm(total=len(indexes), desc=f'Quality + embed {video_path.name}', unit='sample') as bar:
        if reader.total != plan['source_total_frames']:
            raise RuntimeError('NVDEC frame count differs from packet PTS timeline')
        source_seen = pointer = 0
        for start, rgb in reader:
            if start != source_seen:
                raise RuntimeError('Frame order changed during embedding')
            source_seen += len(rgb)
            left, right = np.searchsorted(indexes, [start, start + len(rgb)])
            if left != pointer:
                raise RuntimeError('Sampled frame skipped by decoder')
            if right == left:
                continue
            sample = rgb[torch.as_tensor(indexes[left:right] - start, device=rgb.device)]
            accepted, scores, nonflat, sharpness = score_batch(sample, min_hist_bins)
            sharpness_scores[left:right] = sharpness
            for local, row in enumerate(range(left, right)):
                frame_index = indexes[row]
                while shot < len(scenes) and frame_index > scenes[shot, 1]:
                    finish_shot()
                    shot += 1
                if shot >= len(scenes) or frame_index < scenes[shot, 0]:
                    raise RuntimeError('Sample outside detected shots')
                shot_seen = True
                if nonflat[local] and scores[local] > best_nonflat_score:
                    best_nonflat_score, best_nonflat_row = float(scores[local]), row
                    best_nonflat_frame = sample[local].clone()
                if accepted[local]:
                    shot_has_valid = True
            rows = [row for row in range(left, right) if accepted[row - left]]
            frames = [sample[row - left] for row in rows]
            enqueue(rows, frames)
            pointer = right
            bar.update(right - left)
        if pointer != len(indexes):
            raise RuntimeError('Video ended before all samples were inspected')
        while shot < len(scenes):
            finish_shot()
            shot += 1
        flush()
    if features is None or not selected.any():
        raise RuntimeError('All sampled shots contain only uniform-color frames; no keyframes to export')
    result = dict(filter_sample_plan(plan, selected), features=features[selected],
                  sharpness=sharpness_scores[selected],
                  source_frame_indexes=indexes.copy(),
                  source_scenes=scenes.copy(),
                  kept_scenes=np.asarray(kept_scenes, dtype=np.int64).reshape(-1, 2),
                  quality_policy=f'fused-v5-sharpness-{min_hist_bins}bins')
    save_embedding_cache(result, output_path)
    print(f'Quality + embed: {int(selected.sum())}/{len(indexes)} samples; '
          f'shot fallbacks: {fallbacks}; uniform-only shots dropped: {dropped_shots}', flush=True)
    return result
