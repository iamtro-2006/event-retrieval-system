from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import njit


@dataclass(slots=True)
class TemporalMatch:
    video_id: str
    score: float
    avg_score: float
    selected_indices: list[int]
    selected_keyframes: list[dict]
    duration_sec: float
    start_time: float
    end_time: float


@dataclass(slots=True)
class Candidates:
    """Per-video candidate keyframes handed to Temporal Search by a
    Retrieval Backend.

    All arrays are positional and share the same length ``n``. The
    ``embedding_matrix`` has ``n`` rows of unit-norm float32 vectors.
    ``records`` carries the backend-specific metadata dicts (one per row)
    which Temporal Search returns verbatim inside each ``TemporalMatch``.
    """

    video_id: np.ndarray
    timestamp_sec: np.ndarray
    row_index: np.ndarray
    records: list[dict]
    embedding_matrix: np.ndarray


def _python(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    try:
        return None if pd.isna(value) else value
    except Exception:
        return value


def _clean_row(row: dict) -> dict:
    return {str(k): _python(v) for k, v in row.items()}


# =====================================================================
# DP core (Numba JIT). O(m*n) strict-order alignment.
# =====================================================================
@njit(fastmath=True, nogil=True, cache=True)
def _run_dp_on_window_numba(S: np.ndarray) -> tuple[float, np.ndarray]:
    """O(m*n) strict-order DP using compiled C-loop for maximum speed."""
    m, n = S.shape
    if m == 0 or n < m:
        return -np.inf, np.empty(0, dtype=np.int32)

    dp_prev = S[0].copy()
    parents = np.full((m, n), -1, dtype=np.int32)

    for qi in range(1, m):
        dp_cur = np.full(n, -np.inf, dtype=np.float32)
        current_max_val = -np.inf
        current_max_idx = -1

        for j in range(n):
            if j > 0:
                prev_j = j - 1
                if dp_prev[prev_j] > current_max_val:
                    current_max_val = dp_prev[prev_j]
                    current_max_idx = prev_j

            if current_max_idx != -1:
                dp_cur[j] = current_max_val + S[qi, j]
                parents[qi, j] = current_max_idx

        dp_prev = dp_cur

    best_score = -np.inf
    last = -1
    for j in range(n):
        if dp_prev[j] > best_score:
            best_score = dp_prev[j]
            last = j

    if last == -1 or not np.isfinite(best_score):
        return -np.inf, np.empty(0, dtype=np.int32)

    path = np.empty(m, dtype=np.int32)
    path[m - 1] = last
    for qi in range(m - 1, 0, -1):
        last = parents[qi, last]
        if last < 0:
            return -np.inf, np.empty(0, dtype=np.int32)
        path[qi - 1] = last

    return float(best_score), path


def _run_dp_on_window(S: np.ndarray) -> tuple[float, list[int]]:
    score, path_array = _run_dp_on_window_numba(S)
    return score, path_array.tolist() if path_array.size > 0 else []


def _time_iou(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 1e-12 else 0.0


def _overlap(a: list[int], b: list[int]) -> float:
    return len(set(a).intersection(b)) / max(1, min(len(a), len(b)))


def _temporal_topk_dp(
    S: np.ndarray,
    timestamps: np.ndarray,
    duration_limit: float = -1,
    max_sequences: int = 3,
    overlap_threshold: float = 0.6,
) -> list[tuple[float, list[int]]]:
    m, n = S.shape
    if n < m:
        return []

    if duration_limit < 0:
        work = S.copy()
        candidates = []

        for _ in range(max_sequences * 4):
            score, path = _run_dp_on_window(work)
            if not path:
                break

            candidates.append(
                (score, path, float(timestamps[path[0]]), float(timestamps[path[-1]]))
            )

            for qi, frame_idx in enumerate(path):
                work[qi, frame_idx] = -np.inf

    else:
        candidates = []
        ends = np.searchsorted(timestamps, timestamps + duration_limit, side="right")

        last_end = -1
        for start, end in enumerate(ends):
            end = int(end)
            if end - start < m or end == last_end:
                continue

            last_end = end
            score, local = _run_dp_on_window(S[:, start:end])
            if local:
                path = [start + i for i in local]
                candidates.append(
                    (
                        score,
                        path,
                        float(timestamps[path[0]]),
                        float(timestamps[path[-1]]),
                    )
                )

    candidates.sort(key=lambda x: x[0], reverse=True)

    selected = []
    for candidate in candidates:
        score, path, start, end = candidate
        if any(
            _overlap(path, p) >= overlap_threshold
            or _time_iou(start, end, s, e) >= overlap_threshold
            for _, p, s, e in selected
        ):
            continue

        selected.append(candidate)
        if len(selected) >= max_sequences:
            break

    return [(score, path) for score, path, _, _ in selected]


def search(
    candidates: Candidates,
    query_embeddings: np.ndarray,
    sub_queries: list[str],
    duration_limit: float = -1,
    top_k_videos: int = 10,
    max_sequences_per_video: int = 3,
    overlap_threshold: float = 0.6,
) -> list[TemporalMatch]:
    """Align ``query_embeddings`` against per-video candidate keyframes.

    Solves a strict-order dynamic program over the query x frame similarity
    matrix for each video, then non-maximum-suppresses overlapping results
    and returns the top matches ranked by average score.
    """
    if len(sub_queries) != len(query_embeddings):
        raise ValueError("sub_queries/query_embeddings length mismatch")

    query_embeddings = np.ascontiguousarray(query_embeddings, dtype=np.float32)
    results: list[TemporalMatch] = []

    v_ids = np.asarray(candidates.video_id)
    timestamps_all = np.asarray(candidates.timestamp_sec, dtype=np.float32)
    row_index_all = np.asarray(candidates.row_index, dtype=np.int64)
    embedding_matrix = np.asarray(candidates.embedding_matrix, dtype=np.float32)
    records = candidates.records

    unique_vids = np.unique(v_ids)

    for video_id in unique_vids:
        idx_for_vid = np.where(v_ids == video_id)[0]

        sub_times = timestamps_all[idx_for_vid]
        sort_order = np.argsort(sub_times, kind="mergesort")
        idx_for_vid = idx_for_vid[sort_order]

        sub_row_index = row_index_all[idx_for_vid]
        sub_times_final = timestamps_all[idx_for_vid]

        if len(idx_for_vid) < len(query_embeddings):
            continue

        frame_embeddings = embedding_matrix[sub_row_index]

        S = (query_embeddings @ frame_embeddings.T).astype(np.float32, copy=False)

        matches = _temporal_topk_dp(
            S,
            sub_times_final,
            duration_limit,
            max_sequences_per_video,
            overlap_threshold,
        )
        if not matches:
            continue

        for score, path in matches:
            selected = []
            for qi, local_frame_idx in enumerate(path):
                global_row = int(sub_row_index[local_frame_idx])
                row = _clean_row(records[global_row])
                row.update(
                    sub_query_idx=qi,
                    sub_query=str(sub_queries[qi]),
                    score=float(S[qi, local_frame_idx]),
                )
                selected.append(row)

            start = float(sub_times_final[path[0]])
            end = float(sub_times_final[path[-1]])

            results.append(
                TemporalMatch(
                    video_id=str(video_id),
                    score=float(score),
                    avg_score=float(score / len(query_embeddings)),
                    selected_indices=[int(x) for x in path],
                    selected_keyframes=selected,
                    duration_sec=end - start,
                    start_time=start,
                    end_time=end,
                )
            )

    results.sort(key=lambda item: item.avg_score, reverse=True)
    return results[:top_k_videos]


def matches_to_dataframe(matches: list[TemporalMatch]) -> pd.DataFrame:
    """Shape ``TemporalMatch`` results into the legacy response DataFrame.

    Shared by both Retrieval Backends so the df-shaping logic lives in one
    place. Each row represents one video-level match and carries the matched
    keyframe sequence under ``matched_sequence``.
    """
    rows = []
    for rank, item in enumerate(matches, 1):
        row = dict(item.selected_keyframes[len(item.selected_keyframes) // 2])
        row.update(
            rank=rank,
            display_rank=rank,
            video_score=item.score,
            avg_score=item.avg_score,
            retrieval_score=item.avg_score,
            alignment_score=item.avg_score,
            temporal_start_time=item.start_time,
            temporal_end_time=item.end_time,
            temporal_duration_sec=item.duration_sec,
            matched_sequence=item.selected_keyframes,
            selected_indices=item.selected_indices,
        )
        rows.append(row)

    return pd.DataFrame.from_records(rows)
