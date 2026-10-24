"""GPU execution of the vbs-dev clustering/histogram rules; bounded RGB storage."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.keyframe_extraction.models.selector import split_large_scenes, EmbeddingDeduplicator
from src.utils.gpu_video import CudaVideoReader


def cluster_indexes_tensor(features: torch.Tensor, min_frames: int = 3) -> list[int]:
    """Same farthest-first/centroid merge/silhouette/nearest-frame rules as vbs-dev."""
    n = len(features)
    if n == 0:
        return []
    if n < min_frames:
        return [n // 2]
    x = features.float()
    def distance(a, b):
        return torch.cdist(a, b, compute_mode='donot_use_mm_for_euclid_dist')
    distances = distance(x, x)
    distances.fill_diagonal_(0)
    k = max(1, int(np.ceil(np.sqrt(n))))
    centers = [x[0]]
    nearest = distances[:, 0].clone()
    while len(centers) < k:
        index = int(nearest.argmax())
        if float(nearest[index]) < 1e-9:
            break
        centers.append(x[index])
        nearest = torch.minimum(nearest, distances[:, index])
    centroids = torch.stack(centers)
    labels = distance(x, centroids).argmin(dim=1)
    best_centroids, best_score = centroids.clone(), -1.0
    while len(centroids) > 2 and len(labels.unique()) > 1:
        unique, inverse = labels.unique(return_inverse=True)
        if 1 < len(unique) < n:
            memberships = F.one_hot(inverse, num_classes=len(unique)).to(x.dtype)
            counts = memberships.sum(0)
            sums = distances @ memberships
            own = sums.gather(1, inverse[:, None]).squeeze(1)
            own_count = counts[inverse]
            a = own / (own_count - 1).clamp_min(1)
            means = sums / counts
            means.scatter_(1, inverse[:, None], float('inf'))
            b = means.min(dim=1).values
            silhouette = (b-a) / torch.maximum(a,b).clamp_min(1e-12)
            silhouette[own_count == 1] = 0
            score = float(silhouette.mean())
            if score > best_score:
                best_score, best_centroids = score, centroids.clone()
        k = len(centroids)
        pairs = distance(centroids, centroids)
        pairs[torch.tril(torch.ones((k,k), device=x.device, dtype=torch.bool))] = float('inf')
        flat = int(pairs.argmin())
        i,j = divmod(flat,k)
        merged = (centroids[i] + centroids[j]) / 2
        keep = torch.arange(k, device=x.device)
        keep = keep[(keep != i) & (keep != j)]
        centroids = torch.cat([centroids[keep], merged[None]])
        labels = distance(x, centroids).argmin(dim=1)
    # Midpoints can be equally close to their two source frames. Match NumPy's
    # original float32 reduction/tie break only for numerically ambiguous minima.
    distances_to_best = distance(x,best_centroids)
    indexes = distances_to_best.argmin(dim=0).tolist()
    if n > 1:
        closest = distances_to_best.topk(2,dim=0,largest=False).values
        ambiguous = ((closest[1]-closest[0]).abs() <= 1e-6).nonzero().flatten().tolist()
        for column in ambiguous:
            near = (distances_to_best[:,column] <= closest[0,column]+1e-6).nonzero().flatten()
            rows = x[near].cpu().numpy()
            centroid = best_centroids[column].cpu().numpy()
            index = int(np.argmin(np.linalg.norm(rows-centroid,axis=1)))
            indexes[column] = int(near[index])
    return sorted(set(indexes))


def select_candidates_cuda(scenes: np.ndarray, features: np.ndarray, min_frames: int,
                           max_gap: int, device='cuda', frame_indexes: np.ndarray | None = None,
                           sample_ranges: np.ndarray | None = None) -> list[int]:
    if frame_indexes is None:
        scenes = split_large_scenes(scenes, max_gap)
    else:
        if len(frame_indexes) != len(features) or sample_ranges is None:
            raise ValueError('Sampled features require aligned frame indexes and sample ranges')
        scenes = sample_ranges
    selected = []
    with torch.inference_mode():
        for start,end in tqdm(scenes.tolist(), desc='Clustering', unit='scene'):
            local = torch.as_tensor(features[start:end+1], device=device)
            rows = [start+i for i in cluster_indexes_tensor(local, min_frames)]
            selected.extend(rows if frame_indexes is None else frame_indexes[rows].tolist())
    return sorted(set(selected))


def histogram_tensor(rgb: torch.Tensor, min_bins: int) -> torch.Tensor | None:
    """Match cv2.calcHist BGR [8,8,8], ranges [0,255), min-bins and MINMAX norm."""
    bgr = rgb.flip(0).permute(1,2,0).reshape(-1,3)
    valid = (bgr < 255).all(dim=1)  # 255 is excluded by the original OpenCV ranges.
    bins = torch.floor(bgr[valid].float() * (8.0/255.0)).long()
    indexes = bins[:,0]*64 + bins[:,1]*8 + bins[:,2]
    hist = torch.bincount(indexes, minlength=512).float()
    if int((hist > 0).sum()) <= min_bins:
        return None
    low,high = hist.min(), hist.max()
    return (hist-low) / (high-low).clamp_min(1e-12)


def _write_encoded(path, encoded):
    path.write_bytes(encoded.cpu().numpy().tobytes())


def export_selected_cuda(video: Path, indexes: list[int], image_dir: Path,
                         quality: int, batch_size: int = 8, save_images: bool = True) -> list[int]:
    if indexes != sorted(set(indexes)) or (indexes and indexes[0] < 0):
        raise ValueError('Selected indexes must be sorted and unique')
    if not indexes or not save_images:
        return indexes
    from torchvision.io import encode_jpeg
    image_dir.mkdir(parents=True, exist_ok=True)
    pointer = 0
    pending = []
    with CudaVideoReader(video, batch_size) as reader, \
            ThreadPoolExecutor(max_workers=2) as writers, \
            tqdm(total=len(indexes), desc='Keyframe JPEG', unit='frame') as bar, \
            torch.inference_mode():
        for start, rgb in reader:
            frames, paths = [], []
            while pointer < len(indexes) and indexes[pointer] < start + len(rgb):
                if indexes[pointer] < start:
                    raise RuntimeError('Selected frame skipped by decoder')
                frames.append(rgb[indexes[pointer] - start].contiguous())
                paths.append(image_dir / f'{pointer:06d}.jpg')
                pointer += 1
                bar.update(1)
            if frames:
                for path, encoded in zip(paths, encode_jpeg(frames, quality=quality)):
                    pending.append(writers.submit(_write_encoded, path, encoded))
                    if len(pending) >= 2:
                        pending.pop(0).result()
            if pointer == len(indexes):
                break
        for future in pending:
            future.result()
    if pointer != len(indexes):
        raise RuntimeError('Video ended before keyframes were exported')
    return indexes


def dedup_export_cuda(video: Path, candidates: list[int], candidate_features: np.ndarray, image_dir: Path,
                                threshold: float, min_bins: int, quality: int,
                                batch_size: int = 8, save_images: bool = True) -> list[int]:
    """Old flat-color histogram filter, PE-Core cosine dedup, then accepted JPEGs.

    Decode sequentially once. GPU RGB frames are retained only for the current batch;
    JPEG encoding uses torchvision's CUDA/nvJPEG path; CPU only writes encoded bytes.
    At most two file writes are in flight.
    """
    if not 0 <= threshold <= 1 or min_bins < 0:
        raise ValueError('Invalid embedding dedup/flat-color configuration')
    image_dir.mkdir(parents=True, exist_ok=True)
    if candidates != sorted(set(candidates)) or (candidates and candidates[0] < 0):
        raise ValueError('Candidates must be sorted, unique, nonnegative')
    if not candidates:
        return []
    from torchvision.io import encode_jpeg
    keep = []
    flat_removed, duplicate_removed = 0,0
    pointer = 0
    pending = []
    with CudaVideoReader(video,batch_size) as reader, \
         ThreadPoolExecutor(max_workers=2) as writers, \
         tqdm(total=len(candidates), desc='Flat filter + cosine + JPEG', unit='candidate') as bar, \
         torch.inference_mode():
        dedup = EmbeddingDeduplicator(candidate_features,candidates,threshold,device='cuda')
        for start,rgb in reader:
            accepted_frames, image_paths = [], []
            while pointer < len(candidates) and candidates[pointer] < start+len(rgb):
                index = candidates[pointer]
                if index < start:
                    raise RuntimeError('Candidate frame was skipped by decoder')
                frame = rgb[index-start]
                hist = histogram_tensor(frame,min_bins)
                if hist is not None:
                    if dedup.accept(index):
                        if save_images:
                            accepted_frames.append(frame.contiguous())
                            image_paths.append(image_dir/f'{len(keep):06d}.jpg')
                        keep.append(index)
                    else:
                        duplicate_removed += 1
                else:
                    flat_removed += 1
                pointer += 1
                bar.update(1)
            if accepted_frames:
                encoded_frames = encode_jpeg(accepted_frames,quality=quality)
                for path,encoded in zip(image_paths,encoded_frames):
                    pending.append(writers.submit(_write_encoded,path,encoded))
                    if len(pending) >= 2:
                        pending.pop(0).result()
                del encoded_frames
            del accepted_frames
            if pointer == len(candidates):
                break
        for future in pending:
            future.result()
        if pointer != len(candidates):
            raise RuntimeError('Video ended before all candidate frames were read')
    print(f'Candidates: {len(candidates)}; flat-color removed: {flat_removed}; '
          f'cosine > {threshold:g} removed: {duplicate_removed}; kept: {len(keep)}',flush=True)
    return keep
