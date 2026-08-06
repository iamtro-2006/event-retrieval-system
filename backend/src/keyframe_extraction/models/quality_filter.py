from __future__ import annotations

import cv2
import numpy as np

from src.keyframe_extraction.schemas import Candidate
from src.utils.config import QualityConfig


def evaluate_quality(candidate: Candidate, rgb: np.ndarray, cfg: QualityConfig) -> Candidate:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).reshape(-1)
    probability = histogram / max(float(histogram.sum()), 1.0)
    nonzero = probability[probability > 0]
    entropy = float(-np.sum(nonzero * np.log2(nonzero)))
    edge_density = float(np.mean(cv2.Canny(gray, 80, 160) > 0))
    clipped_fraction = float(np.mean((gray <= 5) | (gray >= 250)))

    candidate.blur = blur
    candidate.brightness = brightness
    candidate.entropy = entropy
    candidate.edge_density = edge_density
    candidate.clipped_fraction = clipped_fraction
    candidate.valid = True
    candidate.rejection_reason = ""
    if blur < cfg.blur_min:
        candidate.valid, candidate.rejection_reason = False, "severe_blur"
    elif brightness < cfg.brightness_min:
        candidate.valid, candidate.rejection_reason = False, "near_black"
    elif brightness > cfg.brightness_max:
        candidate.valid, candidate.rejection_reason = False, "near_white"
    elif entropy < cfg.entropy_min and edge_density < cfg.edge_density_min:
        candidate.valid, candidate.rejection_reason = False, "very_low_information"
    elif clipped_fraction > cfg.clipped_fraction_max:
        candidate.valid, candidate.rejection_reason = False, "severe_exposure_clipping"

    sharpness = min(1.0, np.log1p(blur) / np.log1p(500.0))
    exposure = max(0.0, 1.0 - abs(brightness - 127.5) / 127.5)
    exposure *= max(0.0, 1.0 - clipped_fraction / max(cfg.clipped_fraction_max, 1e-12))
    candidate.quality = float(
        0.35 * sharpness
        + 0.25 * exposure
        + 0.20 * min(1.0, entropy / 7.0)
        + 0.20 * min(1.0, edge_density / 0.12)
    )
    return candidate


def filter_candidates(
    candidates: list[Candidate], images: dict[int, np.ndarray], cfg: QualityConfig
) -> list[Candidate]:
    return [evaluate_quality(candidate, images[candidate.frame_idx], cfg) for candidate in candidates]


def valid_with_shot_fallback(candidates: list[Candidate]) -> tuple[list[Candidate], int]:
    """Keep valid candidates, or the best-quality candidate when a shot has none."""
    by_shot: dict[int, list[Candidate]] = {}
    for candidate in candidates:
        by_shot.setdefault(candidate.shot_id, []).append(candidate)
    selected: list[Candidate] = []
    fallback_shots = 0
    for items in by_shot.values():
        valid = [candidate for candidate in items if candidate.valid]
        if valid:
            selected.extend(valid)
            continue
        fallback = max(items, key=lambda candidate: candidate.quality)
        fallback.source = f"{fallback.source}+quality_fallback"
        selected.append(fallback)
        fallback_shots += 1
    return selected, fallback_shots


