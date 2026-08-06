from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from src.keyframe_extraction.models.clustering import cluster_candidates, normalized_centroid
from src.keyframe_extraction.schemas import Candidate
from src.utils.config import ClusteringConfig, DedupConfig, SelectorConfig


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right))


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    span = float(values.max() - values.min()) if len(values) else 0.0
    return np.zeros_like(values) if span < 1e-12 else (values - values.min()) / span


def select_common_anchor(items: list[Candidate], cfg: SelectorConfig) -> Candidate:
    if not items:
        raise ValueError("cannot select an anchor from an empty shot")
    if len(items) == 1:
        candidate = items[0]
        candidate.typicality = 0.0
        candidate.volatility = 0.0
        return candidate
    features = np.stack([candidate.feature for candidate in items])
    affinity = features @ features.T
    typicality = (affinity.sum(axis=1) - 1.0) / (len(items) - 1)
    volatility = []
    for index in range(len(items)):
        neighbors = [
            other for other in range(max(0, index - cfg.local_neighbor_radius),
                                     min(len(items), index + cfg.local_neighbor_radius + 1))
            if other != index
        ]
        volatility.append(1.0 - float(np.mean(affinity[index, neighbors])) if neighbors else 0.0)
    scores = cfg.common_lambda * _minmax(typicality) - (1.0 - cfg.common_lambda) * _minmax(volatility)
    selected_index = int(np.argmax(scores))
    for index, candidate in enumerate(items):
        candidate.typicality = float(typicality[index])
        candidate.volatility = float(volatility[index])
    return items[selected_index]


def unique_gate(
    items: list[Candidate], common: Candidate, spread: float, cfg: SelectorConfig
) -> tuple[Candidate | None, bool, str]:
    if len(items) <= 1:
        return None, False, "too_few_candidates"
    features = np.stack([candidate.feature for candidate in items])
    affinity = features @ features.T
    typicality = (affinity.sum(axis=1) - 1.0) / (len(items) - 1)
    raw_volatility = []
    for index in range(len(items)):
        neighbors = [
            other for other in range(max(0, index - cfg.local_neighbor_radius),
                                     min(len(items), index + cfg.local_neighbor_radius + 1))
            if other != index
        ]
        raw_volatility.append(1.0 - float(np.mean(affinity[index, neighbors])) if neighbors else 0.0)
    normalized_typicality = _minmax(typicality)
    normalized_volatility = _minmax(np.asarray(raw_volatility))
    scores = cfg.unique_alpha * (1.0 - normalized_typicality) + (1.0 - cfg.unique_alpha) * normalized_volatility
    common_index = next(index for index, candidate in enumerate(items) if candidate is common)
    scores[common_index] = -1.0
    unique_index = int(np.argmax(scores))
    unique = items[unique_index]
    unique.typicality = float(typicality[unique_index])
    unique.volatility = float(raw_volatility[unique_index])
    unique.unique_score = float(scores[unique_index])
    start_time, end_time = items[0].timestamp_sec, items[-1].timestamp_sec
    checks = [
        (unique.quality >= cfg.min_unique_quality, "low_quality"),
        (min(unique.timestamp_sec - start_time, end_time - unique.timestamp_sec) >= cfg.unique_boundary_margin_sec,
         "near_boundary"),
        (_cosine(unique.feature, common.feature) < cfg.max_common_cosine, "too_similar_to_common"),
        (spread >= cfg.min_shot_spread, "low_shot_spread"),
        ((max(raw_volatility) - min(raw_volatility)) >= cfg.min_volatility_range, "low_volatility_range"),
        (normalized_volatility[unique_index] >= cfg.min_unique_volatility, "low_normalized_volatility"),
        ((1.0 - normalized_typicality[unique_index]) >= cfg.min_unique_atypicality, "low_atypicality"),
    ]
    failed = [reason for passed, reason in checks if not passed]
    return unique, not failed, failed[0] if failed else ""


def select_p3(
    by_shot: dict[int, list[Candidate]],
    clustering_cfg: ClusteringConfig,
    selector_cfg: SelectorConfig,
    dedup_cfg: DedupConfig,
) -> tuple[list[Candidate], dict[int, dict[str, Any]], dict[str, int]]:
    selected: list[Candidate] = []
    optional: list[tuple[Candidate, str]] = []
    shot_diagnostics: dict[int, dict[str, Any]] = {}
    metrics = {"unique_proposed": 0, "unique_passed": 0, "medoid_extras_proposed": 0}

    for shot_id, items in sorted(by_shot.items()):
        items.sort(key=lambda candidate: candidate.frame_idx)
        labels, medoids, aucc, spread, invalid_k = cluster_candidates(items, clustering_cfg)
        center = normalized_centroid(items)
        common = select_common_anchor(items, selector_cfg)
        common.is_anchor = True
        common.selection_source = "shot_common_anchor"
        common.representativeness = _cosine(common.feature, center)
        selected.append(common)
        for index, candidate in enumerate(items):
            candidate.cluster_id = int(labels[index])
            candidate.cluster_count = len(medoids)
            candidate.cluster_aucc = aucc
            candidate.spread = spread
        for medoid_index in medoids:
            medoid = items[int(medoid_index)]
            if medoid is common or _cosine(medoid.feature, common.feature) >= dedup_cfg.dense_cosine_threshold:
                continue
            medoid.representativeness = _cosine(medoid.feature, center)
            optional.append((medoid, "medoid"))
            metrics["medoid_extras_proposed"] += 1

        unique, passed, reason = unique_gate(items, common, spread, selector_cfg)
        if unique is not None:
            metrics["unique_proposed"] += 1
        if passed and unique is not None:
            metrics["unique_passed"] += 1
            optional.append((unique, "unique"))
        shot_diagnostics[shot_id] = {
            "spread": spread,
            "k": len(medoids),
            "aucc": aucc,
            "invalid_k": invalid_k,
            "common_frame_idx": common.frame_idx,
            "unique_frame_idx": None if unique is None else unique.frame_idx,
            "unique_passed": passed,
            "unique_reason": reason,
        }

    unique_optional = {id(candidate): (candidate, kind) for candidate, kind in optional}
    optional = list(unique_optional.values())
    extras: dict[int, int] = defaultdict(int)
    while optional:
        best: tuple[tuple[float, int], Candidate, str, float] | None = None
        for candidate, kind in optional:
            if extras[candidate.shot_id] >= selector_cfg.max_extras_per_shot:
                continue
            redundancy = max(_cosine(candidate.feature, old.feature) for old in selected)
            if redundancy >= selector_cfg.global_novelty_cosine:
                continue
            if any(abs(candidate.timestamp_sec - old.timestamp_sec) < selector_cfg.min_temporal_gap_sec for old in selected):
                continue
            relevance = (
                0.7 * candidate.representativeness + 0.3 * candidate.quality
                if kind == "medoid"
                else 0.5 * candidate.unique_score + 0.5 * candidate.quality
            )
            score = selector_cfg.mmr_lambda * relevance - (1.0 - selector_cfg.mmr_lambda) * redundancy
            key = (score, -candidate.frame_idx)
            if best is None or key > best[0]:
                best = (key, candidate, kind, score)
        if best is None:
            break
        _, candidate, kind, score = best
        candidate.mmr_score = float(score)
        candidate.selection_source = "cluster_medoid_global_mmr" if kind == "medoid" else "unique_global_mmr"
        selected.append(candidate)
        extras[candidate.shot_id] += 1
        optional = [(item, item_kind) for item, item_kind in optional if item is not candidate]

    metrics["cluster_medoid_selected"] = sum(
        candidate.selection_source == "cluster_medoid_global_mmr" for candidate in selected
    )
    metrics["unique_selected"] = sum(candidate.selection_source == "unique_global_mmr" for candidate in selected)
    return sorted(selected, key=lambda candidate: candidate.frame_idx), shot_diagnostics, metrics

