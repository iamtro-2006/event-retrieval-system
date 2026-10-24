"""Shot-preserving representative selection from already clustered candidates."""
import numpy as np


def select_shot_mmr(candidates, features, scenes, budget_fraction=.5, anchors_per_shot=3,
                    relevance_weight=.4):
    indexes = np.asarray(candidates, dtype=np.int64)
    vectors = np.asarray(features, dtype=np.float32)
    shots = np.asarray(scenes, dtype=np.int64).reshape(-1, 2)
    if (len(indexes) != len(vectors) or vectors.ndim != 2 or
            not 0 < budget_fraction <= 1 or anchors_per_shot < 1 or
            not 0 <= relevance_weight <= 1 or
            np.any(np.diff(indexes) <= 0) or not np.isfinite(vectors).all()):
        raise ValueError('Invalid MMR candidate set or configuration')
    if not len(indexes):
        return []
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norm, 1e-12)
    anchors = []
    shot_for_row = np.full(len(indexes), -1, dtype=np.int32)
    for shot_id, (first, last) in enumerate(shots):
        rows = np.flatnonzero((indexes >= first) & (indexes <= last))
        if not len(rows):
            continue  # Quality gate found no usable signal in this shot.
        shot_for_row[rows] = shot_id
        local = vectors[rows] @ vectors[rows].T
        relevance = local.mean(axis=1)
        selected = []
        for _ in range(min(anchors_per_shot, len(rows))):
            redundancy = local[:, selected].max(axis=1) if selected else np.zeros(len(rows))
            score = relevance_weight * relevance - (1 - relevance_weight) * redundancy
            score[selected] = -np.inf
            selected.append(int(np.argmax(score)))
        anchors.extend(rows[selected].tolist())
    if np.any(shot_for_row < 0):
        raise ValueError('Candidate outside all shots')
    target = min(len(indexes), max(len(anchors), int(np.ceil(budget_fraction * len(indexes)))))
    selected = list(anchors)
    # Relevance is local to a shot; diversity is global, including all anchors.
    relevance = np.empty(len(indexes), dtype=np.float32)
    for shot_id in range(len(shots)):
        rows = np.flatnonzero(shot_for_row == shot_id)
        if len(rows):
            relevance[rows] = (vectors[rows] @ vectors[rows].T).mean(axis=1)
    while len(selected) < target:
        redundancy = (vectors @ vectors[selected].T).max(axis=1)
        score = relevance_weight * relevance - (1 - relevance_weight) * redundancy
        score[selected] = -np.inf
        selected.append(int(np.argmax(score)))
    return indexes[sorted(selected)].tolist()


def dedup_selected_keyframes(indexes, features, sharpness, threshold=.97,
                             novelty_weight=.6, device='cpu'):
    """Resolve cosine conflicts by local temporal novelty and image sharpness."""
    frame_indexes = np.asarray(indexes, dtype=np.int64)
    vectors = np.asarray(features, dtype=np.float32)
    clarity = np.asarray(sharpness, dtype=np.float32)
    if (vectors.ndim != 2 or len(vectors) != len(frame_indexes)
            or clarity.shape != (len(frame_indexes),)
            or np.any(np.diff(frame_indexes) <= 0)
            or not np.isfinite(vectors).all() or not np.isfinite(clarity).all()
            or np.any(clarity < 0) or not np.isfinite(threshold)
            or not 0 <= threshold <= 1 or not 0 <= novelty_weight <= 1):
        raise ValueError('Invalid quality-aware dedup inputs')
    if len(frame_indexes) < 2:
        return frame_indexes.tolist()
    vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    if device == 'cuda':
        import torch
        tensor = torch.as_tensor(vectors, device='cuda')
        similarities = (tensor @ tensor.T).cpu().numpy()
    elif device == 'cpu':
        similarities = vectors @ vectors.T
    else:
        raise ValueError('Dedup device must be cpu or cuda')
    np.clip(similarities, -1., 1., out=similarities)
    conflicts = np.argwhere(np.triu(similarities > threshold, k=1))
    order = np.argsort(-similarities[conflicts[:, 0], conflicts[:, 1]], kind='stable')
    active = np.ones(len(frame_indexes), dtype=bool)

    def local_novelty(row, rival):
        others = np.flatnonzero(active)
        others = others[others != row]
        others = others[others != rival]
        before = others[others < row]
        after = others[others > row]
        neighbors = []
        if len(before):
            neighbors.append(before[-1])
        if len(after):
            neighbors.append(after[0])
        return 1. - max(similarities[row, neighbors]) if neighbors else 1.

    for position in order:
        first, second = conflicts[position]
        if not (active[first] and active[second]):
            continue
        novel_first = local_novelty(first, second)
        novel_second = local_novelty(second, first)
        novelty_total = novel_first + novel_second
        novelty_share = (.5 if novelty_total <= 1e-12
                         else novel_first / novelty_total)
        sharp_first = np.log1p(clarity[first])
        sharp_second = np.log1p(clarity[second])
        sharp_total = sharp_first + sharp_second
        sharp_share = .5 if sharp_total <= 1e-12 else sharp_first / sharp_total
        first_score = novelty_weight * novelty_share + (1 - novelty_weight) * sharp_share
        active[second if first_score >= .5 else first] = False
    return frame_indexes[active].tolist()
