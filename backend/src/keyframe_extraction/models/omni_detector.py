"""OmniShotCut v1.5 detector; reuse upstream model, query decoding and stitching."""
from __future__ import annotations

import gc
import math
from pathlib import Path
from contextlib import contextmanager
import numpy as np
import torch
from tqdm import tqdm


def clean_shot_ranges(ranges, intra_labels, general_id):
    """Match OmniShotCut.inference(mode='clean_shot') without host RGB decoding.

    Upstream clean_shot keeps general intra labels and returns ranges only.
    Source ranges are end-exclusive; conversion to inclusive happens below.
    """
    return [span for span, label in zip(ranges, intra_labels) if label == general_id]


@contextmanager
def _cuda_inputs(engine, windows):
    """Only used in a sequential worker; restore upstream globals even on error."""
    split, transform = engine.split_videos, engine.video_transform
    from src.utils.gpu_video import normalized_omni_clip
    engine.split_videos = lambda *_: windows
    engine.video_transform = normalized_omni_clip
    try:
        yield
    finally:
        engine.split_videos, engine.video_transform = split, transform


def load_omni_detector(cfg):
    import omnishotcut
    if cfg.weights_path:
        weights = str(cfg.weights_path)
    else:
        from huggingface_hub import hf_hub_download
        weights = hf_hub_download(
            repo_id='uva-cv-lab/OmniShotCut_v1.5', filename='OmniShotCut_ckpt.pth',
            revision='1bc678977ad48c65d57d600296d95f97c1be293f', cache_dir=str(cfg.cache_dir),
        )
    return omnishotcut.load(weights)


def detect_omni_scenes(video: Path, scene_path: Path, cfg, backend: str = 'nvdec',
                       model_cache=None) -> np.ndarray:
    from omnishotcut import engine
    from omnishotcut.label_correspondence import unique_intra_label_mapping
    detector = (load_omni_detector(cfg) if model_cache is None
                else model_cache.omni(cfg, load_omni_detector))
    try:
        if backend == 'nvdec':
            from src.utils.gpu_video import CudaVideoReader, iter_gpu_windows, resized_batches
            args = detector._model_args
            with CudaVideoReader(video, cfg.decode_batch_size) as reader:
                total = reader.total
                window, overlap = int(args.max_process_window_length), int(cfg.overlap)
                if not 0 <= overlap < window:
                    raise ValueError('omni.overlap must be smaller than the checkpoint window')
                windows = iter_gpu_windows(
                    resized_batches(reader, args.process_height, args.process_width), total, window, overlap,
                )
                count = 1 + max(0, math.ceil((total - window) / (window - overlap)))
                # Preserve the complete upstream inference/query/stitch loop. Only replace
                # the iterable and validation transform so frames stay on CUDA.
                with tqdm(windows, total=count, desc='OmniShotCut NVDEC', unit='window') as bar, \
                     _cuda_inputs(engine,bar):
                    ranges, labels, _ = engine._windows_to_ranges(
                        range(total), detector._model, args, overlap,
                    )
                # Same postprocessing as public mode='clean_shot', with bounded
                # GPU windows instead of its full-video host decode/resize.
                ranges = clean_shot_ranges(ranges, labels, unique_intra_label_mapping['general'])
        else:
            from src.utils.omni_progress import inference_with_progress
            ranges = inference_with_progress(detector, video, overlap=cfg.overlap, mode='clean_shot')
        scenes = np.asarray([[int(a), int(b)-1] for a, b in ranges
                             if b > a], dtype=np.int32).reshape(-1, 2)
        if not len(scenes):
            raise RuntimeError('OmniShotCut returned no clean shots')
        if np.any(scenes < 0) or np.any(scenes[:, 0] > scenes[:, 1]):
            raise RuntimeError('Invalid OmniShotCut ranges')
        scene_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = scene_path.with_suffix(scene_path.suffix + '.partial')
        try:
            np.savetxt(temporary, scenes, fmt='%d')
            temporary.replace(scene_path)
        finally:
            temporary.unlink(missing_ok=True)
        return scenes
    finally:
        del detector
        if model_cache is None:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
