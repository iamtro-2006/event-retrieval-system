from __future__ import annotations

from pathlib import Path
import logging

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.keyframe_extraction.models.clustering import select_cluster_keyframes
from src.utils.video_io import get_video_fps, read_frame


def split_large_scenes(scenes: np.ndarray, max_gap: int) -> np.ndarray:
    chunks: list[tuple[int, int]] = []
    for start, end in scenes.tolist():
        cur = int(start)
        end = int(end)
        while end - cur > max_gap:
            chunks.append((cur, cur + max_gap))
            cur += max_gap + 1
        chunks.append((cur, end))
    return np.asarray(chunks, dtype=np.int32)


def _histogram(frame: np.ndarray, min_bins: int) -> np.ndarray | None:
    hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 255, 0, 255, 0, 255]).flatten()
    if int(np.sum(hist > 0)) <= min_bins:
        return None
    return cv2.normalize(hist, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX).flatten()


class EmbeddingDeduplicator:
    """Chronological cosine dedup against all accepted candidates, on CPU or CUDA.

    Call accept only AFTER the flat-color filter, so a rejected flat frame cannot
    suppress a later valid frame with a similar embedding.
    """
    def __init__(self, features: np.ndarray, frame_indexes: list[int], threshold: float = .98, device='cpu'):
        import torch
        import torch.nn.functional as F
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError('Embedding dedup threshold must be in [0,1]')
        values = np.asarray(features,dtype=np.float32)
        if values.ndim != 2 or len(values) != len(frame_indexes) or not np.isfinite(values).all():
            raise ValueError('Candidate embeddings must be finite and aligned with frame indexes')
        if frame_indexes != sorted(set(frame_indexes)):
            raise ValueError('Dedup candidates must be sorted and unique')
        self.vectors = F.normalize(torch.as_tensor(values,device=device),dim=1)
        self.kept = torch.empty_like(self.vectors)
        self.rows = {index:row for row,index in enumerate(frame_indexes)}
        self.count = 0
        self.threshold = threshold

    def accept(self, frame_index: int) -> bool:
        vector = self.vectors[self.rows[frame_index]]
        if self.count and bool((self.kept[:self.count] @ vector > self.threshold).any()):
            return False
        self.kept[self.count] = vector
        self.count += 1
        return True


def embedding_dedup(video_path: Path, frame_indexes: list[int], features: np.ndarray,
                    threshold: float, min_bins: int) -> list[int]:
    keep: list[int] = []
    dedup = EmbeddingDeduplicator(features,frame_indexes,threshold)

    for idx in frame_indexes:
        frame = read_frame(video_path, idx)
        if frame is None:
            continue
        hist = _histogram(frame, min_bins)
        if hist is None:
            continue

        if dedup.accept(idx):
            keep.append(idx)

    return sorted(keep)


def extract_keyframe_indexes(
    video_path: Path,
    scenes: np.ndarray,
    features: np.ndarray,
    min_scene_frames: int,
    max_gap: int,
    dedup_threshold: float,
    min_hist_bins: int,
    logger: logging.Logger | None = None,
    max_scene_seconds: float | None = None,
) -> list[int]:
    effective_max_gap = max_gap
    if max_scene_seconds is not None:
        if max_scene_seconds <= 0:
            raise ValueError("max_scene_seconds must be positive when provided")
        fps = get_video_fps(video_path)
        if fps > 0:
            effective_max_gap = max(1, round(max_scene_seconds * fps) - 1)
    scenes = split_large_scenes(scenes, effective_max_gap)
    selected: list[int] = []

    if logger:
        logger.info("Selecting keyframes: %s scenes=%d", video_path.name, len(scenes))

    for start, end in tqdm(scenes.tolist(), desc=f"Keyframes {video_path.name}", unit="scene"):
        start, end = int(start), int(end)
        sub = features[start:end + 1]
        local = select_cluster_keyframes(sub, min_scene_frames)
        selected.extend([start + idx for idx in local])

    candidates = sorted(set(selected))
    deduped = embedding_dedup(video_path,candidates,features[candidates],dedup_threshold,min_hist_bins)
    if logger:
        logger.info("Selected keyframes: %s before_dedup=%d after_dedup=%d", video_path.name, len(set(selected)), len(deduped))
    return deduped


def save_keyframe_map(indexes: list[int], video_path: Path, csv_path: Path,
                      fps: float | None = None, timestamps: np.ndarray | None = None) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fps = get_video_fps(video_path) if fps is None else fps
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('Original video FPS must be positive')
    times = [idx/fps for idx in indexes] if timestamps is None else np.asarray(timestamps,dtype=np.float64)
    if len(times) != len(indexes) or not np.isfinite(times).all():
        raise ValueError('Keyframe timestamps must align with source frame indexes')
    pd.DataFrame(
        {
            "keyframe_id": list(range(len(indexes))),
            "video_id": video_path.stem,
            "frame_idx": indexes,
            "timestamp_sec": times,
            "fps": [fps] * len(indexes),
        }
    ).to_csv(csv_path, index=False)


def save_keyframe_images(indexes: list[int], video_path: Path, output_dir: Path, image_quality: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("keyframe_pipeline")

    for keyframe_id, frame_idx in enumerate(indexes):
        frame = read_frame(video_path, frame_idx)

        if frame is None:
            logger.warning("Skipped keyframe %d (frame_idx=%d): could not read frame", keyframe_id, frame_idx)
            continue

        out_path = output_dir / f"{keyframe_id:06d}.jpg"
        ok = cv2.imwrite(
            str(out_path),
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(image_quality)],
        )

        if not ok:
            logger.warning("Failed to write keyframe %d (frame_idx=%d) to %s", keyframe_id, frame_idx, out_path)
