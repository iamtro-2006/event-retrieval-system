from __future__ import annotations

import cv2
import numpy as np

from src.keyframe_extraction.models.candidate_sampler import cheap_phash, phash_hamming
from src.keyframe_extraction.schemas import Candidate, DedupRecord
from src.utils.config import DedupConfig


def perceptual_hash(rgb: np.ndarray) -> int:
    return cheap_phash(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))


def dense_cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = left / max(float(np.linalg.norm(left)), 1e-12)
    right_norm = right / max(float(np.linalg.norm(right)), 1e-12)
    return float(np.dot(left_norm, right_norm))


def deduplicate(
    candidates: list[Candidate], images: dict[int, np.ndarray], cfg: DedupConfig
) -> tuple[list[Candidate], list[DedupRecord]]:
    kept: list[Candidate] = []
    hashes: list[int] = []
    dropped: list[DedupRecord] = []
    for candidate in sorted(candidates, key=lambda item: item.frame_idx):
        candidate_hash = perceptual_hash(images[candidate.frame_idx])
        duplicate: DedupRecord | None = None
        for index, old in enumerate(kept):
            if abs(candidate.timestamp_sec - old.timestamp_sec) > cfg.temporal_window_sec:
                continue
            hamming = phash_hamming(candidate_hash, hashes[index])
            cosine = dense_cosine(candidate.feature, old.feature)
            if hamming <= cfg.phash_hamming_threshold:
                duplicate = DedupRecord(candidate, old, f"phash_hamming_{hamming}", hamming, cosine)
                break
            if cosine >= cfg.dense_cosine_threshold:
                duplicate = DedupRecord(candidate, old, f"dense_cosine_{cosine:.4f}", hamming, cosine)
                break
        if duplicate is None:
            kept.append(candidate)
            hashes.append(candidate_hash)
        else:
            dropped.append(duplicate)
    return kept, dropped



