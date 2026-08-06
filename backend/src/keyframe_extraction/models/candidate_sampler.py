from __future__ import annotations

from collections.abc import Iterable
from typing import Callable

import cv2
import numpy as np

from src.keyframe_extraction.schemas import Candidate
from src.utils.config import CandidateConfig


def cheap_phash(gray: np.ndarray) -> int:
    low = cv2.dct(cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32))[:8, :8]
    bits = (low > np.median(low[1:])).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def phash_hamming(left: int, right: int) -> int:
    return (int(left) ^ int(right)).bit_count()


def sample_candidates(
    frames: Iterable[tuple[int, np.ndarray]],
    scenes: np.ndarray,
    fps: float,
    cfg: CandidateConfig,
    observer: Callable[[Candidate, np.ndarray], None] | None = None,
    image_cache: dict[int, np.ndarray] | None = None,
) -> list[Candidate]:
    """Sequential adaptive scan matching the P3 notebook candidate gate.

    The observer evaluates a prospective candidate before it can update the
    sampler state. Rejected frames therefore cannot postpone the next valid
    candidate. Only accepted, resized RGB images are retained when requested.
    """
    ranges = np.asarray(scenes, dtype=np.int32).reshape(-1, 2)
    protected: set[int] = set()
    for start, end in ranges:
        protected.update((int(start), (int(start) + int(end)) // 2, int(end)))

    min_gap = max(1, int(round(cfg.min_gap_sec * fps)))
    max_gap = max(min_gap, int(round(cfg.max_gap_sec * fps)))
    candidates: list[Candidate] = []
    shot_id = 0
    last_idx = -10**9
    last_thumb: np.ndarray | None = None
    last_hash: int | None = None

    for frame_idx, bgr in frames:
        while shot_id + 1 < len(ranges) and frame_idx > int(ranges[shot_id, 1]):
            shot_id += 1
            last_idx, last_thumb, last_hash = -10**9, None, None
        if not len(ranges) or frame_idx < int(ranges[shot_id, 0]) or frame_idx > int(ranges[shot_id, 1]):
            continue

        height, width = bgr.shape[:2]
        scale = min(1.0, 320.0 / max(width, 1))
        small = cv2.resize(
            bgr,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        thumb = cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA)
        current_hash = cheap_phash(gray)
        gap = frame_idx - last_idx
        hamming = 64 if last_hash is None else phash_hamming(current_hash, last_hash)
        change = 1.0 if last_thumb is None else float(
            np.mean(np.abs(thumb.astype(np.float32) - last_thumb.astype(np.float32))) / 255.0
        )
        keep = (
            frame_idx in protected
            or gap >= max_gap
            or (gap >= min_gap and (hamming >= cfg.phash_min_distance or change >= cfg.pixel_change_threshold))
        )
        if keep:
            source = "protected" if frame_idx in protected else ("max_gap" if gap >= max_gap else "content_change")
            candidate = Candidate(shot_id, frame_idx, frame_idx / fps, source)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            if observer is not None:
                observer(candidate, rgb)
            if not candidate.valid:
                continue
            candidates.append(candidate)
            if image_cache is not None:
                image_cache[frame_idx] = rgb
            last_idx, last_thumb, last_hash = frame_idx, thumb, current_hash
    return candidates
