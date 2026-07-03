from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score


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
