from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


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


def _load_embedding(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    emb = np.asarray(np.load(path), dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 1e-12 else emb


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


def _run_dp_on_window(S: np.ndarray) -> tuple[float, list[int]]:
    """O(m*n) strict-order DP using vectorized prefix maxima."""
    m, n = S.shape
    if m == 0 or n < m:
        return -np.inf, []
    dp_prev = S[0].astype(np.float32, copy=True)
    parents = np.full((m, n), -1, dtype=np.int32)
    for qi in range(1, m):
        prefix_idx = np.maximum.accumulate(
            np.where(dp_prev == np.maximum.accumulate(dp_prev), np.arange(n), -1)
        )
        prev_idx = np.empty(n, dtype=np.int32)
        prev_idx[0] = -1
        prev_idx[1:] = prefix_idx[:-1]
        valid = prev_idx >= 0
        dp_cur = np.full(n, -np.inf, dtype=np.float32)
        dp_cur[valid] = dp_prev[prev_idx[valid]] + S[qi, valid]
        parents[qi] = prev_idx
        dp_prev = dp_cur
    last = int(np.argmax(dp_prev))
    score = float(dp_prev[last])
    if not np.isfinite(score):
        return -np.inf, []
    path = [last]
    for qi in range(m - 1, 0, -1):
        last = int(parents[qi, last])
        if last < 0:
            return -np.inf, []
        path.append(last)
    return score, path[::-1]


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
    # Unlimited duration only needs one global DP. Additional sequences are generated
    # by masking selected frames, avoiding n repeated full-window DP calls.
    if duration_limit < 0:
        work = S.copy()
        candidates = []
        for _ in range(max_sequences * 4):
            score, path = _run_dp_on_window(work)
            if not path:
                break
            candidates.append((score, path, float(timestamps[path[0]]), float(timestamps[path[-1]])))
            for qi, frame_idx in enumerate(path):
                work[qi, frame_idx] = -np.inf
    else:
        candidates = []
        ends = np.searchsorted(timestamps, timestamps + duration_limit, side="right")
        # Run only windows capable of changing the feasible right boundary.
        last_end = -1
        for start, end in enumerate(ends):
            end = int(end)
            if end - start < m or end == last_end:
                continue
            last_end = end
            score, local = _run_dp_on_window(S[:, start:end])
            if local:
                path = [start + i for i in local]
                candidates.append((score, path, float(timestamps[path[0]]), float(timestamps[path[-1]])))
    candidates.sort(key=lambda x: x[0], reverse=True)
    selected = []
    for candidate in candidates:
        score, path, start, end = candidate
        if any(_overlap(path, p) >= overlap_threshold or _time_iou(start, end, s, e) >= overlap_threshold
               for _, p, s, e in selected):
            continue
        selected.append(candidate)
        if len(selected) >= max_sequences:
            break
    return [(score, path) for score, path, _, _ in selected]


def temporal_search_from_candidates(
    query_embeddings: np.ndarray,
    sub_queries: list[str],
    candidate_df: pd.DataFrame,
    duration_limit: float = -1,
    top_k_videos: int = 10,
    max_sequences_per_video: int = 3,
    overlap_threshold: float = 0.6,
    embedding_matrix: np.ndarray | None = None,
) -> pd.DataFrame:
    required = {"video_id", "timestamp_sec"}
    missing = required - set(candidate_df.columns)
    if missing:
        raise ValueError(f"Missing columns for temporal search: {missing}")
    if len(sub_queries) != len(query_embeddings):
        raise ValueError("sub_queries/query_embeddings length mismatch")

    query_embeddings = np.ascontiguousarray(query_embeddings, dtype=np.float32)
    results: list[TemporalMatch] = []
    # groupby(sort=False) avoids sorting group keys; stable mergesort preserves chronology.
    for video_id, raw in candidate_df.groupby("video_id", sort=False):
        sort_cols = ["timestamp_sec"]
        video_df = raw.sort_values(sort_cols, kind="mergesort")
        dedupe_col = "_faiss_id" if "_faiss_id" in video_df else "keyframe_id"
        video_df = video_df.drop_duplicates(dedupe_col, keep="first").reset_index(drop=True)
        if len(video_df) < len(query_embeddings):
            continue

        if embedding_matrix is not None and "_faiss_id" in video_df:
            ids = video_df["_faiss_id"].to_numpy(dtype=np.int64, copy=False)
            frame_embeddings = embedding_matrix[ids]
        elif "embedding_path" in video_df:
            frame_embeddings = np.stack([_load_embedding(p) for p in video_df["embedding_path"]])
        else:
            raise ValueError("No cached vectors and no embedding_path available")

        S = (query_embeddings @ frame_embeddings.T).astype(np.float32, copy=False)
        timestamps = pd.to_numeric(video_df["timestamp_sec"], errors="coerce").fillna(0).to_numpy(np.float32)
        matches = _temporal_topk_dp(S, timestamps, duration_limit, max_sequences_per_video, overlap_threshold)
        if not matches:
            continue
        records = video_df.to_dict(orient="records")
        for score, path in matches:
            selected = []
            for qi, frame_local_idx in enumerate(path):
                row = _clean_row(dict(records[frame_local_idx]))
                row.update(sub_query_idx=qi, sub_query=str(sub_queries[qi]), score=float(S[qi, frame_local_idx]))
                selected.append(row)
            start, end = float(timestamps[path[0]]), float(timestamps[path[-1]])
            results.append(TemporalMatch(str(video_id), float(score), float(score / len(query_embeddings)),
                                         [int(x) for x in path], selected, end - start, start, end))

    results.sort(key=lambda item: item.avg_score, reverse=True)
    rows = []
    for rank, item in enumerate(results[:top_k_videos], 1):
        row = dict(item.selected_keyframes[len(item.selected_keyframes) // 2])
        row.update(rank=rank, display_rank=rank, video_score=item.score, avg_score=item.avg_score,
                   retrieval_score=item.avg_score, alignment_score=item.avg_score,
                   temporal_start_time=item.start_time, temporal_end_time=item.end_time,
                   temporal_duration_sec=item.duration_sec, matched_sequence=item.selected_keyframes,
                   selected_indices=item.selected_indices)
        rows.append(row)
    return pd.DataFrame.from_records(rows)
