"""VLM/V3C P3 common/medoid/unique selection on sampled PE-Core features."""
from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
from tqdm import tqdm

@dataclass
class Candidate:
    scene_id: int
    frame_id: int
    timestamp_sec: float
    source: str
    quality: float = 0.0
    valid: bool = True
    rejection_reason: str = ""
    blur: float = 0.0
    brightness: float = 0.0
    entropy: float = 0.0
    edge_density: float = 0.0
    clipped_fraction: float = 0.0
    feature: np.ndarray | None = None
    is_anchor: bool = False
    selection_source: str = ""
    mmr_score: float = 0.0
    spread: float = 0.0
    complexity: str = ""
    cluster_id: int = -1
    cluster_aucc: float = float("nan")
    cluster_metric_name: str = ""
    cluster_metric_value: float = float("nan")
    cluster_count: int = 1
    representativeness: float = 0.0
    typicality: float = float("nan")
    volatility: float = float("nan")
    unique_score: float = float("nan")



MAX_CLUSTERS_PER_SHOT = 5
MIN_CLUSTER_SIZE = 2
CLUSTER_MIN_SPREAD = 0.025
AUCC_COMPLEXITY_PENALTY = 0.035
KMEDOIDS_MAX_ITER = 50
COMMON_LAMBDA = 0.70
LOCAL_NEIGHBOR_RADIUS = 2
UNIQUE_ALPHA = 0.50
MIN_UNIQUE_QUALITY = 0.35
UNIQUE_MAX_COMMON_COSINE = 0.90
UNIQUE_MIN_SHOT_SPREAD = 0.025
UNIQUE_MIN_VOLATILITY_RANGE = 0.02
MIN_UNIQUE_VOLATILITY = 0.60
MIN_UNIQUE_ATYPICALITY = 0.50
UNIQUE_BOUNDARY_MARGIN_SEC = 0.30
MMR_LAMBDA = 0.60
MIN_TEMPORAL_GAP_SEC = 1.0
GLOBAL_NOVELTY_COSINE = 0.90
MAX_EXTRAS_PER_SHOT = 4
DENSE_DEDUP_COSINE = 0.97



def cosine(a, b):
    return float(np.dot(a, b))


def normalized_centroid(items):
    center = np.stack([c.feature for c in items]).mean(0)
    return center / max(float(np.linalg.norm(center)), 1e-12)


def shot_spread(items):
    if len(items) < 2:
        return 0.0
    center = normalized_centroid(items)
    return float(np.mean([1.0 - cosine(c.feature, center) for c in items]))


def minmax(values):
    values = np.asarray(values, dtype=np.float32)
    span = float(values.max() - values.min()) if len(values) else 0.0
    return np.zeros_like(values) if span < 1e-12 else (values - values.min()) / span


def reset_selection(items):
    for c in items:
        c.is_anchor = False; c.selection_source = ""; c.mmr_score = 0.0
        c.cluster_id = -1; c.cluster_count = 1
        c.cluster_metric_name = ""; c.cluster_metric_value = float("nan")
        c.representativeness = 0.0; c.typicality = float("nan")
        c.volatility = float("nan"); c.unique_score = float("nan")


def valid_k_limit(n):
    return min(MAX_CLUSTERS_PER_SHOT, n // MIN_CLUSTER_SIZE, n - 1)


def kmedoids(distance, k):
    medoids = [int(np.argmin(distance.sum(1)))]
    while len(medoids) < k:
        nearest = distance[:, medoids].min(1); nearest[medoids] = -1
        medoids.append(int(np.argmax(nearest)))
    medoids = np.asarray(medoids, dtype=np.int32)
    for _ in range(KMEDOIDS_MAX_ITER):
        labels = np.argmin(distance[:, medoids], axis=1); updated = medoids.copy()
        for cluster_id in range(k):
            members = np.flatnonzero(labels == cluster_id)
            if len(members):
                updated[cluster_id] = members[np.argmin(distance[np.ix_(members, members)].sum(1))]
        if np.array_equal(updated, medoids): break
        medoids = updated
    return np.argmin(distance[:, medoids], axis=1), medoids


def aucc_score(distance, labels):
    upper = np.triu_indices(len(labels), 1); values = distance[upper]
    same = labels[upper[0]] == labels[upper[1]]
    positive, negative = values[same], values[~same]
    if not len(positive) or not len(negative): return float("nan")
    negative = np.sort(negative)
    left = np.searchsorted(negative, positive, side="left")
    right = np.searchsorted(negative, positive, side="right")
    return float(np.mean((len(negative) - right + 0.5 * (right - left)) / len(negative)))


def cluster_kmedoids(items):
    features = np.stack([c.feature for c in items])
    distance = np.clip(1.0 - features @ features.T, 0.0, 2.0)
    spread = shot_spread(items); best = None; invalid = 0
    if len(items) >= 2 * MIN_CLUSTER_SIZE and spread >= CLUSTER_MIN_SPREAD:
        for k in range(2, valid_k_limit(len(items)) + 1):
            labels, medoids = kmedoids(distance, k)
            if np.bincount(labels, minlength=k).min() < MIN_CLUSTER_SIZE:
                invalid += 1; continue
            aucc = aucc_score(distance, labels)
            objective = aucc - AUCC_COMPLEXITY_PENALTY * (k - 1)
            if best is None or objective > best[0]: best = (objective, labels, medoids, aucc)
    if best is None:
        labels, medoids = kmedoids(distance, 1); return labels, medoids, float("nan"), spread, invalid
    return best[1], best[2], best[3], spread, invalid


def select_p3(by_shot, budget):
    selected, optional, shot_meta = [], [], {}; diagnostics = {"unique_proposed": 0, "unique_passed": 0}
    for shot_id, items in tqdm(sorted(by_shot.items()), desc="P3 shot selection", unit="shot", leave=False):
        started = time.perf_counter(); labels, medoids, metric, spread, invalid = cluster_kmedoids(items)
        k = len(medoids); features = np.stack([c.feature for c in items]); affinity = features @ features.T
        typicality = (affinity.sum(1) - 1.0) / max(len(items)-1, 1)
        raw_volatility = []
        for i in range(len(items)):
            neighbors = [j for j in range(max(0, i-LOCAL_NEIGHBOR_RADIUS),
                                          min(len(items), i+LOCAL_NEIGHBOR_RADIUS+1)) if j != i]
            raw_volatility.append(1.0 - float(np.mean(affinity[i, neighbors])) if neighbors else 0.0)
        g, v = minmax(typicality), minmax(raw_volatility)
        common_scores = COMMON_LAMBDA*g - (1-COMMON_LAMBDA)*v
        common_idx = int(np.argmax(common_scores)); common = items[common_idx]
        common.is_anchor = True; common.selection_source = "shot_common_anchor"
        common.typicality = float(typicality[common_idx]); common.volatility = float(raw_volatility[common_idx])
        common.representativeness = cosine(common.feature, normalized_centroid(items)); selected.append(common)
        for i, c in enumerate(items):
            c.cluster_id = int(labels[i]); c.cluster_count = k
            c.cluster_metric_name = "aucc"; c.cluster_metric_value = metric
        for medoid_index in medoids:
            c = items[int(medoid_index)]
            if c is common or cosine(c.feature, common.feature) >= DENSE_DEDUP_COSINE: continue
            c.selection_source = "cluster_medoid_extra"; c.representativeness = cosine(c.feature, normalized_centroid(items))
            optional.append((c, "medoid"))
        unique_idx = None; unique_pass = False; reason = "too_few_candidates"
        if len(items) > 1:
            scores = UNIQUE_ALPHA*(1-g) + (1-UNIQUE_ALPHA)*v; scores[common_idx] = -1
            unique_idx = int(np.argmax(scores)); u = items[unique_idx]; diagnostics["unique_proposed"] += 1
            u.typicality = float(typicality[unique_idx]); u.volatility = float(raw_volatility[unique_idx]); u.unique_score = float(scores[unique_idx])
            start_t, end_t = items[0].timestamp_sec, items[-1].timestamp_sec
            checks = [(u.quality >= MIN_UNIQUE_QUALITY, "low_quality"),
                      (min(u.timestamp_sec-start_t, end_t-u.timestamp_sec) >= UNIQUE_BOUNDARY_MARGIN_SEC, "near_boundary"),
                      (cosine(u.feature, common.feature) < UNIQUE_MAX_COMMON_COSINE, "too_similar_to_common"),
                      (spread >= UNIQUE_MIN_SHOT_SPREAD, "low_shot_spread"),
                      ((max(raw_volatility)-min(raw_volatility)) >= UNIQUE_MIN_VOLATILITY_RANGE, "low_volatility_range"),
                      (v[unique_idx] >= MIN_UNIQUE_VOLATILITY, "low_normalized_volatility"),
                      ((1-g[unique_idx]) >= MIN_UNIQUE_ATYPICALITY, "low_atypicality")]
            failed = [label for ok, label in checks if not ok]
            unique_pass = not failed; reason = "" if unique_pass else failed[0]
            if unique_pass:
                diagnostics["unique_passed"] += 1; u.selection_source = "unique_candidate"; optional.append((u, "unique"))
        shot_meta[shot_id] = dict(spread=spread, k=k, metric_name="aucc", metric=metric,
                                  invalid_k=invalid, elapsed=time.perf_counter()-started,
                                  common=common.frame_id, unique=None if unique_idx is None else items[unique_idx].frame_id,
                                  unique_pass=unique_pass, unique_reason=reason)
    optional = list({id(c):(c, kind) for c, kind in optional}.values())
    extras = {shot_id:0 for shot_id in by_shot}
    target = float("inf") if budget is None else budget
    while optional and len(selected) < target:
        best = None
        for c, kind in optional:
            if extras[c.scene_id] >= MAX_EXTRAS_PER_SHOT: continue
            redundancy = max(cosine(c.feature, old.feature) for old in selected)
            if redundancy >= GLOBAL_NOVELTY_COSINE: continue
            if any(abs(c.timestamp_sec-old.timestamp_sec) < MIN_TEMPORAL_GAP_SEC for old in selected): continue
            relevance = ((0.7*c.representativeness + 0.3*c.quality) if kind == "medoid"
                         else (0.5*c.unique_score + 0.5*c.quality))
            score = MMR_LAMBDA*relevance - (1-MMR_LAMBDA)*redundancy
            key = (score, -c.frame_id)
            if best is None or key > best[0]: best = (key, c, kind, score)
        if best is None: break
        _, c, kind, score = best; c.is_anchor = False; c.mmr_score = float(score)
        c.selection_source = "cluster_medoid_global_mmr" if kind == "medoid" else "unique_global_mmr"
        selected.append(c); extras[c.scene_id] += 1; optional = [(x,k) for x,k in optional if x is not c]
    diagnostics["cluster_medoid_selected"] = sum(c.selection_source == "cluster_medoid_global_mmr" for c in selected)
    diagnostics["unique_selected"] = sum(c.selection_source == "unique_global_mmr" for c in selected)
    return selected, shot_meta, diagnostics



def select_sampled_p3(plan, features, quality):
    """Use bounded timestamp sampling regions (<=30s), keeping all quality samples."""
    by_shot = {}
    for shot_id, (first, last) in enumerate(plan['sample_ranges']):
        items = []
        for row in range(int(first), int(last) + 1):
            item = Candidate(shot_id, int(plan['frame_indexes'][row]),
                             float(plan['timestamps'][row]), 'timestamp_sample',
                             quality=float(quality[row]), feature=features[row])
            items.append(item)
        if items:
            by_shot[shot_id] = items
    selected, _, diagnostics = select_p3(by_shot, budget=None)
    selected.sort(key=lambda c: c.frame_id)
    return selected, diagnostics


def cosine_dedup(selected, scenes, threshold=0.96):
    """Strict cosine gate only between neighboring clean shots.

    Prefer quality, then earliest. Candidate.scene_id denotes a sampling region,
    so derive the actual shot from its inclusive source-frame boundaries.
    """
    shots = np.asarray(scenes, dtype=np.int64).reshape(-1, 2)
    if (not len(shots) or np.any(shots[:, 0] > shots[:, 1])
            or np.any(shots[1:, 0] <= shots[:-1, 1])
            or not np.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError('Invalid shot boundaries or cosine threshold')
    kept_by_shot = {}
    for item in sorted(selected, key=lambda c: (-c.quality, c.frame_id)):
        shot_id = int(np.searchsorted(shots[:, 0], item.frame_id, side='right')) - 1
        if shot_id < 0 or item.frame_id > shots[shot_id, 1]:
            raise ValueError('P3 candidate outside all clean shots')
        neighbors = (kept_by_shot.get(shot_id - 1, [])
                     + kept_by_shot.get(shot_id + 1, []))
        duplicate = False
        for start in range(0, len(neighbors), 256):
            vectors = np.stack([c.feature for c in neighbors[start:start + 256]])
            if np.any(vectors @ item.feature > threshold):
                duplicate = True
                break
        if not duplicate:
            kept_by_shot.setdefault(shot_id, []).append(item)
    return sorted(c.frame_id for items in kept_by_shot.values() for c in items)
