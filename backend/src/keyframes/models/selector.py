from __future__ import annotations

from pathlib import Path
import logging

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.keyframes.models.clustering import select_cluster_keyframes
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


def histogram_dedup(video_path: Path, frame_indexes: list[int], threshold: float, min_bins: int) -> list[int]:
    keep: list[int] = []
    keep_hists: list[np.ndarray] = []

    for idx in frame_indexes:
        frame = read_frame(video_path, idx)
        if frame is None:
            continue
        hist = _histogram(frame, min_bins)
        if hist is None:
            continue

        duplicate = False
        for kept_hist in keep_hists:
            denom = np.linalg.norm(hist) * np.linalg.norm(kept_hist)
            sim = float(np.dot(hist, kept_hist) / denom) if denom > 0 else 0.0
            if sim > threshold:
                duplicate = True
                break
        if not duplicate:
            keep.append(idx)
            keep_hists.append(hist)

    return sorted(keep)


def extract_keyframe_indexes(
    video_path: Path,
    scenes: np.ndarray,
    features: np.ndarray,
    min_scene_frames: int,
    max_gap: int,
    hist_threshold: float,
    min_hist_bins: int,
    logger: logging.Logger | None = None,
) -> list[int]:
    scenes = split_large_scenes(scenes, max_gap)
    selected: list[int] = []

    if logger:
        logger.info("Selecting keyframes: %s scenes=%d", video_path.name, len(scenes))

    for start, end in tqdm(scenes.tolist(), desc=f"Keyframes {video_path.name}", unit="scene"):
        start, end = int(start), int(end)
        sub = features[start:end + 1]
        local = select_cluster_keyframes(sub, min_scene_frames)
        selected.extend([start + idx for idx in local])

    deduped = histogram_dedup(video_path, sorted(set(selected)), hist_threshold, min_hist_bins)
    if logger:
        logger.info("Selected keyframes: %s before_dedup=%d after_dedup=%d", video_path.name, len(set(selected)), len(deduped))
    return deduped


def save_keyframe_map(indexes: list[int], video_path: Path, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fps = get_video_fps(video_path)
    pd.DataFrame(
        {
            "keyframe_id": list(range(len(indexes))),
            "video_id": video_path.stem,
            "frame_idx": indexes,
            "timestamp_sec": [idx / fps if fps > 0 else 0.0 for idx in indexes],
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