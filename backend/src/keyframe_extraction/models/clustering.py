from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score

from src.keyframe_extraction.schemas import Candidate
from src.utils.config import ClusteringConfig


def init_centroids(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(features)
    k = max(1, int(np.ceil(np.sqrt(n))))
    centers = [features[0]] if n else []

    while len(centers) < k:
        dists = cdist(features, np.asarray(centers), metric="euclidean")
        min_dists = np.min(dists, axis=1)
        if float(np.max(min_dists)) < 1e-9:
            break
        centers.append(features[int(np.argmax(min_dists))])

    centers = np.asarray(centers)
    labels = np.argmin(cdist(features, centers, metric="euclidean"), axis=1)
    return centers, labels


def select_cluster_keyframes(features: np.ndarray, min_frames: int = 3) -> list[int]:
    n = len(features)
    if n == 0:
        return []
    if n < min_frames:
        return [n // 2]

    centroids, labels = init_centroids(features)
    k = len(centroids)
    best_score = -1.0
    best_centroids = centroids.copy()

    while k > 2 and len(set(labels.tolist())) > 1:
        try:
            score = silhouette_score(features, labels)
            if score > best_score:
                best_score = score
                best_centroids = centroids.copy()
        except ValueError:
            pass

        pair = None
        min_dist = np.inf
        for i in range(k):
            for j in range(i + 1, k):
                dist = np.linalg.norm(centroids[i] - centroids[j])
                if dist < min_dist:
                    min_dist = dist
                    pair = (i, j)

        if pair is None:
            break

        i, j = pair
        merged = (centroids[i] + centroids[j]) / 2
        keep = [c for idx, c in enumerate(centroids) if idx not in pair]
        centroids = np.vstack([keep, merged]) if keep else np.asarray([merged])
        labels = np.argmin(cdist(features, centroids, metric="euclidean"), axis=1)
        k = len(centroids)

    indexes = [int(np.argmin(np.linalg.norm(features - centroid, axis=1))) for centroid in best_centroids]
    return sorted(set(indexes))


def normalized_centroid(items: list[Candidate]) -> np.ndarray:
    center = np.stack([candidate.feature for candidate in items]).mean(axis=0)
    return center / max(float(np.linalg.norm(center)), 1e-12)


def shot_spread(items: list[Candidate]) -> float:
    if len(items) < 2:
        return 0.0
    center = normalized_centroid(items)
    return float(np.mean([1.0 - float(np.dot(candidate.feature, center)) for candidate in items]))


def kmedoids(distance: np.ndarray, k: int, max_iter: int = 50) -> tuple[np.ndarray, np.ndarray]:
    if len(distance) == 0 or k < 1 or k > len(distance):
        raise ValueError("k must be between 1 and the number of samples")
    medoids = [int(np.argmin(distance.sum(axis=1)))]
    while len(medoids) < k:
        nearest = distance[:, medoids].min(axis=1)
        nearest[medoids] = -1
        medoids.append(int(np.argmax(nearest)))
    medoids_array = np.asarray(medoids, dtype=np.int32)
    for _ in range(max_iter):
        labels = np.argmin(distance[:, medoids_array], axis=1)
        updated = medoids_array.copy()
        for cluster_id in range(k):
            members = np.flatnonzero(labels == cluster_id)
            if len(members):
                costs = distance[np.ix_(members, members)].sum(axis=1)
                updated[cluster_id] = int(members[np.argmin(costs)])
        if np.array_equal(updated, medoids_array):
            break
        medoids_array = updated
    return np.argmin(distance[:, medoids_array], axis=1), medoids_array


def aucc_score(distance: np.ndarray, labels: np.ndarray) -> float:
    if len(labels) < 2:
        return float("nan")
    upper = np.triu_indices(len(labels), 1)
    values = distance[upper]
    same = labels[upper[0]] == labels[upper[1]]
    positive, negative = values[same], values[~same]
    if not len(positive) or not len(negative):
        return float("nan")
    return float(
        np.mean(positive[:, None] < negative[None, :])
        + 0.5 * np.mean(positive[:, None] == negative[None, :])
    )


def cluster_candidates(
    items: list[Candidate], cfg: ClusteringConfig
) -> tuple[np.ndarray, np.ndarray, float, float, int]:
    if not items:
        raise ValueError("cannot cluster an empty shot")
    features = np.stack([candidate.feature for candidate in items]).astype(np.float32)
    distance = np.clip(1.0 - features @ features.T, 0.0, 2.0)
    spread = shot_spread(items)
    best: tuple[float, np.ndarray, np.ndarray, float] | None = None
    invalid_k = 0
    max_k = min(cfg.max_clusters_per_shot, len(items) // cfg.min_cluster_size, len(items) - 1)
    if len(items) >= 2 * cfg.min_cluster_size and spread >= cfg.min_spread:
        for k in range(2, max_k + 1):
            labels, medoids = kmedoids(distance, k, cfg.kmedoids_max_iter)
            if np.bincount(labels, minlength=k).min() < cfg.min_cluster_size:
                invalid_k += 1
                continue
            aucc = aucc_score(distance, labels)
            objective = aucc - cfg.aucc_complexity_penalty * (k - 1)
            if best is None or objective > best[0]:
                best = (objective, labels, medoids, aucc)
    if best is None:
        labels, medoids = kmedoids(distance, 1, cfg.kmedoids_max_iter)
        return labels, medoids, float("nan"), spread, invalid_k
    return best[1], best[2], best[3], spread, invalid_k
