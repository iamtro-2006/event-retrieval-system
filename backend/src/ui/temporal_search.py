from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.ui.embedding_memmap import EmbeddingMemmapStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Numba JIT – optional, falls back to pure-Python implementations
# ---------------------------------------------------------------------------
try:
    from src.ui.temporal_dp_numba import (
        has_near_tied_suffix_choices as _numba_has_near_tied,
        run_dp_on_window as _numba_run_dp,
        select_non_overlapping as _numba_select_nms,
        suffix_temporal_candidates as _numba_suffix_candidates,
        window_temporal_candidates as _numba_window_candidates,
    )

    _HAS_NUMBA = True
    logger.info("Numba JIT kernels loaded for temporal search")
except ImportError:
    _HAS_NUMBA = False
    logger.info("Numba not available – using pure-Python temporal DP")


@dataclass
class TemporalMatch:
    video_id: str
    score: float
    avg_score: float
    selected_indices: list[int]
    selected_keyframes: list[dict]
    duration_sec: float
    start_time: float
    end_time: float


def _resolve_embedding_path(path: str | Path) -> Path:
    path = Path(path)

    if not path.is_absolute():
        backend_dir = Path(__file__).resolve().parents[2]
        path = backend_dir / path

    return path


@lru_cache(maxsize=8192)
def _load_embedding_cached(path: str) -> np.ndarray:
    emb = np.load(path)

    if emb.ndim == 2:
        emb = emb[0]

    emb = emb.astype(np.float32)
    
    # Faster norm computation using scalar math
    norm_sq = float(np.dot(emb, emb))
    
    if norm_sq > 1e-24:
        emb = emb / math.sqrt(norm_sq)

    return emb


def _load_embedding(path: str | Path) -> np.ndarray:
    return _load_embedding_cached(str(_resolve_embedding_path(path)))


def _to_python_scalar(value):
    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    return value


def _clean_row_dict(row: dict) -> dict:
    return {str(k): _to_python_scalar(v) for k, v in row.items()}


def _run_dp_on_window(S: np.ndarray) -> tuple[float, list[int]]:
    """
    Strict temporal order:
        query_0 -> frame_i
        query_1 -> frame_j
        ...
    with:
        i < j < ...
    """

    m, n = S.shape

    dp = np.full((m, n), -np.inf, dtype=np.float32)
    parent = np.full((m, n), -1, dtype=np.int32)

    dp[0] = S[0]

    for qi in range(1, m):
        prev = dp[qi - 1]
        prefix_scores = np.empty(n, dtype=np.float32)
        prefix_scores[0] = -np.inf

        if n > 1:
            np.maximum.accumulate(prev[:-1], out=prefix_scores[1:])

        valid = np.isfinite(prefix_scores)
        dp[qi, valid] = prefix_scores[valid] + S[qi, valid]

        best_prev_score = -np.inf
        best_prev_idx = -1

        for fj in range(n):
            if fj > 0 and prev[fj - 1] > best_prev_score:
                best_prev_score = prev[fj - 1]
                best_prev_idx = fj - 1

            if best_prev_idx != -1 and valid[fj]:
                parent[qi, fj] = best_prev_idx

    last_idx = int(np.argmax(dp[m - 1]))
    best_score = float(dp[m - 1, last_idx])

    if not math.isfinite(best_score):
        return -np.inf, []

    path = [last_idx]

    for qi in range(m - 1, 0, -1):
        last_idx = int(parent[qi, last_idx])

        if last_idx < 0:
            return -np.inf, []

        path.append(last_idx)

    path.reverse()

    return best_score, path


def _sequence_overlap_ratio(path_a: list[int], path_b: list[int]) -> float:
    """
    Overlap theo index keyframe được chọn.
    Dùng để loại các sequence gần như giống nhau.
    """

    if not path_a or not path_b:
        return 0.0

    set_a = set(path_a)
    set_b = set(path_b)

    intersection = len(set_a & set_b)
    denominator = min(len(set_a), len(set_b))

    if denominator <= 0:
        return 0.0

    return intersection / denominator


def _time_iou(
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
) -> float:
    inter = max(0.0, min(end_a, end_b) - max(start_a, start_b))
    union = max(end_a, end_b) - min(start_a, start_b)

    if union <= 1e-12:
        return 0.0

    return inter / union


def _select_non_overlapping_candidates(
    candidates: list[tuple[float, list[int], float, float]],
    max_sequences: int,
    overlap_threshold: float,
) -> list[tuple[float, list[int]]]:
    if not candidates:
        return []

    candidates.sort(key=lambda x: x[0], reverse=True)

    selected: list[tuple[float, list[int], float, float]] = []

    for score, path, start_time, end_time in candidates:
        duplicated = False

        for _, selected_path, selected_start, selected_end in selected:
            path_overlap = _sequence_overlap_ratio(path, selected_path)
            time_overlap = _time_iou(
                start_time,
                end_time,
                selected_start,
                selected_end,
            )

            if (
                path_overlap >= overlap_threshold
                or time_overlap >= overlap_threshold
            ):
                duplicated = True
                break

        if duplicated:
            continue

        selected.append((score, path, start_time, end_time))

        if len(selected) >= max_sequences:
            break

    return [(score, path) for score, path, _, _ in selected]


def _has_near_tied_suffix_choices(
    S: np.ndarray,
    *,
    atol: float = 1e-6,
) -> bool:
    """
    The suffix-DP path is much faster for unlimited-duration searches, but it
    accumulates float32 scores from right to left. The legacy window-DP
    accumulates left to right. In rare near-tie cases, that one-ULP difference
    can alter which overlapping sequence survives NMS. Detect only near-ties
    between distinct paths so repeated suffix candidates for the same path do
    not unnecessarily trigger the exact legacy-compatible fallback.
    
    Optimized with early termination when first near-tie is detected.
    """

    m, n = S.shape
    dp = np.full((m, n), -np.inf, dtype=np.float32)
    dp[m - 1] = S[m - 1]

    for qi in range(m - 2, -1, -1):
        next_scores = dp[qi + 1]
        suffix_scores = np.full(n, -np.inf, dtype=np.float32)
        best_score = -np.inf

        for fj in range(n - 1, -1, -1):
            suffix_scores[fj] = best_score
            candidate_score = float(next_scores[fj])

            if (
                math.isfinite(candidate_score)
                and math.isfinite(best_score)
                and abs(candidate_score - best_score) <= atol
            ):
                return True

            if candidate_score >= best_score:
                best_score = candidate_score

        valid = np.isfinite(suffix_scores)
        dp[qi, valid] = S[qi, valid] + suffix_scores[valid]

    best_score = -np.inf

    for start in range(n - 1, -1, -1):
        candidate_score = float(dp[0, start])

        if (
            math.isfinite(candidate_score)
            and math.isfinite(best_score)
            and abs(candidate_score - best_score) <= atol
        ):
            return True

        if candidate_score >= best_score:
            best_score = candidate_score

    return False


def _window_temporal_candidates(
    S: np.ndarray,
    timestamps: np.ndarray,
    duration_limit: float,
) -> list[tuple[float, list[int], float, float]]:
    m, n = S.shape
    candidates: list[tuple[float, list[int], float, float]] = []

    for start in range(n):
        if duration_limit == -1:
            end = n
        else:
            end = int(
                np.searchsorted(
                    timestamps,
                    timestamps[start] + duration_limit,
                    side="right",
                )
            )

        if end - start < m:
            continue

        local_S = S[:, start:end]
        local_score, local_path = _run_dp_on_window(local_S)

        if not local_path or not math.isfinite(local_score):
            continue

        global_path = [start + idx for idx in local_path]
        start_time = float(timestamps[global_path[0]])
        end_time = float(timestamps[global_path[-1]])

        candidates.append(
            (
                float(local_score),
                global_path,
                start_time,
                end_time,
            )
        )

    return candidates


def _suffix_temporal_candidates(
    S: np.ndarray,
    timestamps: np.ndarray,
) -> list[tuple[float, list[int], float, float]]:
    """
    Optimized suffix-DP for unlimited duration searches.
    Vectorizes best-score tracking where possible while preserving exact results.
    """
    m, n = S.shape

    dp = np.full((m, n), -np.inf, dtype=np.float32)
    parent = np.full((m, n), -1, dtype=np.int32)
    dp[m - 1] = S[m - 1]

    for qi in range(m - 2, -1, -1):
        next_scores = dp[qi + 1]
        suffix_scores = np.full(n, -np.inf, dtype=np.float32)
        suffix_indices = np.full(n, -1, dtype=np.int32)

        best_score = -np.inf
        best_idx = -1

        for fj in range(n - 1, -1, -1):
            suffix_scores[fj] = best_score
            suffix_indices[fj] = best_idx

            if next_scores[fj] >= best_score:
                best_score = float(next_scores[fj])
                best_idx = fj

        valid = np.isfinite(suffix_scores)
        dp[qi, valid] = S[qi, valid] + suffix_scores[valid]
        parent[qi, valid] = suffix_indices[valid]

    suffix_best_scores = np.full(n, -np.inf, dtype=np.float32)
    suffix_best_indices = np.full(n, -1, dtype=np.int32)

    best_score = -np.inf
    best_idx = -1

    for start in range(n - 1, -1, -1):
        if dp[0, start] >= best_score:
            best_score = float(dp[0, start])
            best_idx = start

        suffix_best_scores[start] = best_score
        suffix_best_indices[start] = best_idx

    candidates: list[tuple[float, list[int], float, float]] = []

    for start in range(n):
        score = float(suffix_best_scores[start])
        frame_idx = int(suffix_best_indices[start])

        if frame_idx < 0 or not math.isfinite(score):
            continue

        path = [frame_idx]

        for qi in range(m - 1):
            frame_idx = int(parent[qi, frame_idx])

            if frame_idx < 0:
                path = []
                break

            path.append(frame_idx)

        if not path:
            continue

        start_time = float(timestamps[path[0]])
        end_time = float(timestamps[path[-1]])
        candidates.append((score, path, start_time, end_time))

    return candidates


def _numba_candidates_to_list(
    scores, paths_flat, path_offsets, start_times, end_times, path_len,
) -> list[tuple[float, list[int], float, float]]:
    """Unpack flat Numba output arrays into the Python list-of-tuples format."""
    out: list[tuple[float, list[int], float, float]] = []
    pl = int(path_len)
    for i in range(len(scores)):
        off = int(path_offsets[i])
        path = [int(paths_flat[off + j]) for j in range(pl)]
        out.append((float(scores[i]), path, float(start_times[i]), float(end_times[i])))
    return out


def _temporal_topk_dp(
    S: np.ndarray,
    timestamps: np.ndarray,
    duration_limit: float = -1,
    max_sequences: int = 3,
    overlap_threshold: float = 0.6,
) -> list[tuple[float, list[int]]]:
    """
    Sinh nhiều sequence candidate trong cùng một video.

    Cách làm:
    - Mỗi start timestamp tạo một window hợp lệ.
    - Chạy DP trong window đó để lấy best sequence của window.
    - Gom tất cả sequence.
    - Sort theo score giảm dần.
    - Temporal NMS để loại sequence overlap cao.

    When Numba is available, delegates to JIT-compiled kernels for ~5-20× speedup.
    """

    m, n = S.shape

    if m == 0 or n < m:
        return []

    S_f32 = S.astype(np.float32, copy=False)
    ts_f32 = timestamps.astype(np.float32, copy=False)

    # ---- Numba fast path ----
    if _HAS_NUMBA:
        if duration_limit == -1:
            if _numba_has_near_tied(S_f32):
                raw = _numba_window_candidates(S_f32, ts_f32, -1.0)
            else:
                raw = _numba_suffix_candidates(S_f32, ts_f32)
        else:
            raw = _numba_window_candidates(S_f32, ts_f32, float(duration_limit))

        scores, paths_flat, path_offsets, start_times, end_times, path_len = raw

        if len(scores) == 0:
            return []

        sel_indices = _numba_select_nms(
            scores, paths_flat, path_offsets, start_times, end_times,
            int(path_len), max_sequences, overlap_threshold,
        )

        results: list[tuple[float, list[int]]] = []
        pl = int(path_len)
        for idx in sel_indices:
            off = int(path_offsets[idx])
            path = [int(paths_flat[off + j]) for j in range(pl)]
            results.append((float(scores[idx]), path))
        return results

    # ---- Pure-Python fallback (original logic) ----
    if duration_limit == -1:
        if _has_near_tied_suffix_choices(S):
            candidates = _window_temporal_candidates(
                S=S,
                timestamps=timestamps,
                duration_limit=duration_limit,
            )
        else:
            candidates = _suffix_temporal_candidates(S=S, timestamps=timestamps)

        return _select_non_overlapping_candidates(
            candidates=candidates,
            max_sequences=max_sequences,
            overlap_threshold=overlap_threshold,
        )

    candidates = _window_temporal_candidates(
        S=S,
        timestamps=timestamps,
        duration_limit=duration_limit,
    )

    return _select_non_overlapping_candidates(
        candidates=candidates,
        max_sequences=max_sequences,
        overlap_threshold=overlap_threshold,
    )


# ---------------------------------------------------------------------------
# Module-level memmap store (set once at startup via set_memmap_store)
# ---------------------------------------------------------------------------
_memmap_store: EmbeddingMemmapStore | None = None


def set_memmap_store(store: EmbeddingMemmapStore) -> None:
    """Register a pre-built memmap store for O(1) embedding lookups."""
    global _memmap_store
    _memmap_store = store
    logger.info("Memmap embedding store registered (%d embeddings)", len(store))


def get_memmap_store() -> EmbeddingMemmapStore | None:
    return _memmap_store


def _load_embedding_fast(path: str) -> np.ndarray:
    """Load from memmap if available, else fall back to file + LRU cache."""
    if _memmap_store is not None and path in _memmap_store:
        return _memmap_store[path]
    return _load_embedding(path)


def temporal_search_from_candidates(
    query_embeddings: np.ndarray,
    sub_queries: list[str],
    candidate_df: pd.DataFrame,
    duration_limit: float = -1,
    top_k_videos: int = 10,
    max_sequences_per_video: int = 3,
    overlap_threshold: float = 0.6,
    memmap_store: EmbeddingMemmapStore | None = None,
) -> pd.DataFrame:
    """
    query_embeddings: [m_queries, d]
    sub_queries: list query đã được split ở API/main.
    candidate_df cần có:
        video_id, timestamp_sec, embedding_path, keyframe_path, keyframe_id

    Lưu ý:
    - top_k_videos hiện được hiểu là top_k sequence toàn cục.
    - Một video có thể xuất hiện nhiều lần nếu có nhiều sequence tốt.
    - memmap_store: optional EmbeddingMemmapStore for O(1) lookups.
      Falls back to module-level store or per-file loading.
    """

    required = {"video_id", "timestamp_sec", "embedding_path"}
    missing = required - set(candidate_df.columns)

    if missing:
        raise ValueError(f"Missing columns for temporal search: {missing}")

    if len(sub_queries) != len(query_embeddings):
        raise ValueError(
            f"len(sub_queries)={len(sub_queries)} != "
            f"len(query_embeddings)={len(query_embeddings)}"
        )

    query_embeddings = query_embeddings.astype(np.float32)

    # Choose embedding loader: explicit arg > module store > file-based
    store = memmap_store or _memmap_store
    embedding_paths = candidate_df["embedding_path"].dropna().astype(str).unique()

    if store is not None:
        # Batch load from memmap – single indexed read, no per-file I/O
        paths_in_store = [p for p in embedding_paths if p in store]
        paths_fallback = [p for p in embedding_paths if p not in store]

        embedding_cache: dict[str, np.ndarray] = {}
        if paths_in_store:
            batch = store.get_batch(paths_in_store)
            for i, p in enumerate(paths_in_store):
                embedding_cache[p] = batch[i]
        for p in paths_fallback:
            embedding_cache[p] = _load_embedding(p)
    else:
        embedding_cache = {path: _load_embedding(path) for path in embedding_paths}

    results: list[TemporalMatch] = []

    candidate_df = candidate_df.sort_values("timestamp_sec")

    for video_id, video_df in candidate_df.groupby("video_id", sort=False):
        video_df = (
            video_df
            .drop_duplicates(subset=["keyframe_id"], keep="first")
            .reset_index(drop=True)
        )

        if len(video_df) < len(query_embeddings):
            continue

        frame_embeddings = np.stack(
            [
                embedding_cache[path]
                for path in video_df["embedding_path"].astype(str).tolist()
            ]
        )

        S = query_embeddings @ frame_embeddings.T
        timestamps = video_df["timestamp_sec"].to_numpy(dtype=np.float32)

        matches = _temporal_topk_dp(
            S=S,
            timestamps=timestamps,
            duration_limit=duration_limit,
            max_sequences=max_sequences_per_video,
            overlap_threshold=overlap_threshold,
        )

        if not matches:
            continue

        for score, selected_indices in matches:
            selected_rows = video_df.iloc[selected_indices]

            start_time = float(selected_rows["timestamp_sec"].min())
            end_time = float(selected_rows["timestamp_sec"].max())

            selected_keyframes = []
            col_names = list(selected_rows.columns)
            selected_values = selected_rows.values

            for qi in range(len(selected_indices)):
                row_dict = {str(col_names[ci]): _to_python_scalar(selected_values[qi, ci]) for ci in range(len(col_names))}
                frame_local_idx = selected_indices[qi]

                row_dict["sub_query_idx"] = int(qi)
                row_dict["sub_query"] = str(sub_queries[qi])
                row_dict["score"] = float(S[qi, frame_local_idx])

                selected_keyframes.append(row_dict)

            results.append(
                TemporalMatch(
                    video_id=str(video_id),
                    score=float(score),
                    avg_score=float(score / len(query_embeddings)),
                    selected_indices=[int(x) for x in selected_indices],
                    selected_keyframes=selected_keyframes,
                    duration_sec=float(end_time - start_time),
                    start_time=float(start_time),
                    end_time=float(end_time),
                )
            )

    rows = []

    sorted_results = sorted(
        results,
        key=lambda x: x.avg_score,
        reverse=True,
    )[:top_k_videos]

    for rank, item in enumerate(sorted_results, start=1):
        main_frame = item.selected_keyframes[len(item.selected_keyframes) // 2]

        row = _clean_row_dict(dict(main_frame))
        row["rank"] = int(rank)
        row["display_rank"] = int(rank)

        row["video_score"] = float(item.score)
        row["avg_score"] = float(item.avg_score)
        row["retrieval_score"] = float(item.avg_score)

        row["temporal_duration_sec"] = float(item.duration_sec)
        row["temporal_start_time"] = float(item.start_time)
        row["temporal_end_time"] = float(item.end_time)
        row["matched_sequence"] = item.selected_keyframes

        rows.append(row)

    return pd.DataFrame(rows)
