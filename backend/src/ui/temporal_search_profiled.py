from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import logging
import os

import numpy as np
import pandas as pd

try:
    from numba import njit
except ImportError as exc:
    raise ImportError(
        "This module needs numba. Install with: pip install numba"
    ) from exc


LOGGER = logging.getLogger("retrieval.temporal")


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _log_enabled(verbose: bool | None = None) -> bool:
    return _env_flag("TEMPORAL_PROFILE", False) if verbose is None else bool(verbose)


def _log_ms(enabled: bool, message: str, start: float, **fields) -> None:
    if not enabled:
        return

    elapsed_ms = (perf_counter() - start) * 1000.0
    extra = " ".join(f"{k}={v}" for k, v in fields.items())
    LOGGER.info("[TEMPORAL_PROFILE] %s %.2fms%s%s", message, elapsed_ms, " | " if extra else "", extra)


def _configure_profile_logging(enabled: bool) -> None:
    if not enabled:
        return

    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )

    LOGGER.setLevel(logging.INFO)


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


def _load_embedding(path: str | Path) -> np.ndarray:
    path = Path(path)

    if not path.is_absolute():
        backend_dir = Path(__file__).resolve().parents[2]
        path = backend_dir / path

    emb = np.load(path)

    if emb.ndim == 2:
        emb = emb[0]

    emb = emb.astype(np.float32, copy=False)
    norm = np.linalg.norm(emb)

    if norm > 1e-12:
        emb = emb / norm

    return emb.astype(np.float32, copy=False)


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


@njit(cache=True, fastmath=True)
def _time_iou_numba(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    inter_left = start_a if start_a > start_b else start_b
    inter_right = end_a if end_a < end_b else end_b
    inter = inter_right - inter_left

    if inter < 0.0:
        inter = 0.0

    union_left = start_a if start_a < start_b else start_b
    union_right = end_a if end_a > end_b else end_b
    union = union_right - union_left

    if union <= 1e-12:
        return 0.0

    return inter / union


@njit(cache=True, fastmath=True)
def _path_overlap_ratio_numba(path_a: np.ndarray, path_b: np.ndarray, m: int) -> float:
    intersection = 0

    for i in range(m):
        ai = path_a[i]
        for j in range(m):
            if ai == path_b[j]:
                intersection += 1
                break

    if m <= 0:
        return 0.0

    return intersection / m


@njit(cache=True, fastmath=True)
def _run_dp_on_range_numba(
    S: np.ndarray,
    start: int,
    end: int,
) -> tuple[float, np.ndarray]:
    """
    Strict temporal order over a window [start, end):
        query_0 -> frame_i
        query_1 -> frame_j
        ...
    with:
        i < j < ...

    Returns global frame indices, not local window indices.
    """
    m = S.shape[0]
    window_n = end - start

    path = np.full(m, -1, dtype=np.int32)

    if window_n < m:
        return -np.inf, path

    dp = np.full((m, window_n), -np.inf, dtype=np.float32)
    parent = np.full((m, window_n), -1, dtype=np.int32)

    for fj in range(window_n):
        dp[0, fj] = S[0, start + fj]

    for qi in range(1, m):
        best_prev_score = -np.inf
        best_prev_idx = -1

        for fj in range(window_n):
            prev_j = fj - 1

            if prev_j >= 0 and dp[qi - 1, prev_j] > best_prev_score:
                best_prev_score = dp[qi - 1, prev_j]
                best_prev_idx = prev_j

            if best_prev_idx != -1:
                dp[qi, fj] = best_prev_score + S[qi, start + fj]
                parent[qi, fj] = best_prev_idx

    best_last_idx = -1
    best_score = -np.inf

    for fj in range(window_n):
        score = dp[m - 1, fj]
        if score > best_score:
            best_score = score
            best_last_idx = fj

    if best_last_idx < 0 or not np.isfinite(best_score):
        return -np.inf, path

    current = best_last_idx
    path[m - 1] = start + current

    for qi in range(m - 1, 0, -1):
        current = parent[qi, current]

        if current < 0:
            return -np.inf, np.full(m, -1, dtype=np.int32)

        path[qi - 1] = start + current

    return best_score, path


@njit(cache=True, fastmath=True)
def _temporal_topk_dp_numba(
    S: np.ndarray,
    timestamps: np.ndarray,
    duration_limit: float,
    max_sequences: int,
    overlap_threshold: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Numba version of temporal top-k DP + temporal NMS.

    Returns:
        selected_scores: [max_sequences]
        selected_paths:  [max_sequences, m]
        selected_count:  int
    """
    m, n = S.shape

    selected_scores = np.full(max_sequences, -np.inf, dtype=np.float32)
    selected_paths = np.full((max_sequences, m), -1, dtype=np.int32)
    selected_starts = np.full(max_sequences, -1.0, dtype=np.float32)
    selected_ends = np.full(max_sequences, -1.0, dtype=np.float32)

    if n < m or max_sequences <= 0:
        return selected_scores, selected_paths, 0

    candidate_scores = np.full(n, -np.inf, dtype=np.float32)
    candidate_paths = np.full((n, m), -1, dtype=np.int32)
    candidate_starts = np.full(n, -1.0, dtype=np.float32)
    candidate_ends = np.full(n, -1.0, dtype=np.float32)
    candidate_count = 0

    for start in range(n):
        if duration_limit < 0.0:
            end = n
        else:
            limit_time = timestamps[start] + duration_limit
            end = start

            while end < n and timestamps[end] <= limit_time:
                end += 1

        if end - start < m:
            continue

        score, path = _run_dp_on_range_numba(S, start, end)

        if not np.isfinite(score):
            continue

        candidate_scores[candidate_count] = score

        for qi in range(m):
            candidate_paths[candidate_count, qi] = path[qi]

        candidate_starts[candidate_count] = timestamps[path[0]]
        candidate_ends[candidate_count] = timestamps[path[m - 1]]
        candidate_count += 1

    if candidate_count <= 0:
        return selected_scores, selected_paths, 0

    order = np.argsort(candidate_scores[:candidate_count])[::-1]
    selected_count = 0

    for order_i in range(candidate_count):
        cand_idx = order[order_i]
        duplicated = False

        for selected_i in range(selected_count):
            path_overlap = _path_overlap_ratio_numba(
                candidate_paths[cand_idx],
                selected_paths[selected_i],
                m,
            )
            time_overlap = _time_iou_numba(
                candidate_starts[cand_idx],
                candidate_ends[cand_idx],
                selected_starts[selected_i],
                selected_ends[selected_i],
            )

            if path_overlap >= overlap_threshold or time_overlap >= overlap_threshold:
                duplicated = True
                break

        if duplicated:
            continue

        selected_scores[selected_count] = candidate_scores[cand_idx]
        selected_starts[selected_count] = candidate_starts[cand_idx]
        selected_ends[selected_count] = candidate_ends[cand_idx]

        for qi in range(m):
            selected_paths[selected_count, qi] = candidate_paths[cand_idx, qi]

        selected_count += 1

        if selected_count >= max_sequences:
            break

    return selected_scores, selected_paths, selected_count


def _temporal_topk_dp(
    S: np.ndarray,
    timestamps: np.ndarray,
    duration_limit: float = -1,
    max_sequences: int = 3,
    overlap_threshold: float = 0.6,
    profile: bool = False,
    video_id: str | None = None,
) -> list[tuple[float, list[int]]]:
    """
    Python wrapper around the Numba kernel.
    Keeps the old return format: list[(score, path)].
    """
    t_prepare = perf_counter()
    S = np.ascontiguousarray(S, dtype=np.float32)
    timestamps = np.ascontiguousarray(timestamps, dtype=np.float32)
    _log_ms(profile, "dp_prepare", t_prepare, video_id=video_id, S_shape=S.shape)

    t_dp = perf_counter()
    selected_scores, selected_paths, selected_count = _temporal_topk_dp_numba(
        S=S,
        timestamps=timestamps,
        duration_limit=float(duration_limit),
        max_sequences=int(max_sequences),
        overlap_threshold=float(overlap_threshold),
    )
    _log_ms(
        profile,
        "dp_numba",
        t_dp,
        video_id=video_id,
        frames=S.shape[1],
        queries=S.shape[0],
        selected=int(selected_count),
        duration_limit=duration_limit,
    )

    matches: list[tuple[float, list[int]]] = []

    for i in range(int(selected_count)):
        path = [int(x) for x in selected_paths[i].tolist() if int(x) >= 0]

        if path:
            matches.append((float(selected_scores[i]), path))

    return matches


def temporal_search_from_candidates(
    query_embeddings: np.ndarray,
    sub_queries: list[str],
    candidate_df: pd.DataFrame,
    duration_limit: float = -1,
    top_k_videos: int = 10,
    max_sequences_per_video: int = 3,
    overlap_threshold: float = 0.6,
    profile: bool | None = None,
    profile_every_video: bool = True,
) -> pd.DataFrame:
    """
    query_embeddings: [m_queries, d]
    sub_queries: list query đã được split ở API/main.
    candidate_df cần có:
        video_id, timestamp_sec, embedding_path, keyframe_path, keyframe_id

    Lưu ý:
    - top_k_videos hiện được hiểu là top_k sequence toàn cục.
    - Một video có thể xuất hiện nhiều lần nếu có nhiều sequence tốt.
    """
    profile_enabled = _log_enabled(profile)
    _configure_profile_logging(profile_enabled)
    t_total = perf_counter()

    required = {"video_id", "timestamp_sec", "embedding_path"}
    missing = required - set(candidate_df.columns)

    if missing:
        raise ValueError(f"Missing columns for temporal search: {missing}")

    if len(sub_queries) != len(query_embeddings):
        raise ValueError(
            f"len(sub_queries)={len(sub_queries)} != "
            f"len(query_embeddings)={len(query_embeddings)}"
        )

    t_query = perf_counter()
    query_embeddings = np.ascontiguousarray(query_embeddings, dtype=np.float32)
    _log_ms(
        profile_enabled,
        "query_prepare",
        t_query,
        query_shape=query_embeddings.shape,
        sub_queries=len(sub_queries),
        candidates=len(candidate_df),
        videos=candidate_df["video_id"].nunique(),
    )

    results: list[TemporalMatch] = []
    video_count = 0
    skipped_too_short = 0

    t_group_all = perf_counter()
    grouped = candidate_df.groupby("video_id", sort=False)
    _log_ms(profile_enabled, "groupby_init", t_group_all, groups=getattr(grouped, "ngroups", "unknown"))

    for video_id, video_df in grouped:
        video_count += 1
        t_video = perf_counter()
        raw_len = len(video_df)
        t_sort = perf_counter()
        video_df = (
            video_df
            .sort_values("timestamp_sec")
            .drop_duplicates(subset=["keyframe_id"], keep="first")
            .reset_index(drop=True)
        )
        _log_ms(
            profile_enabled and profile_every_video,
            "video_sort_dedup",
            t_sort,
            video_id=video_id,
            raw_frames=raw_len,
            unique_frames=len(video_df),
        )

        if len(video_df) < len(query_embeddings):
            skipped_too_short += 1
            _log_ms(
                profile_enabled and profile_every_video,
                "video_skip_too_short",
                t_video,
                video_id=video_id,
                frames=len(video_df),
                queries=len(query_embeddings),
            )
            continue

        # NOTE: np.load vẫn là Python/I/O bottleneck.
        # Nếu muốn nhanh hơn nữa, nên cache frame embeddings theo video_id.
        t_paths = perf_counter()
        embedding_paths = video_df["embedding_path"].tolist()
        _log_ms(profile_enabled and profile_every_video, "embedding_path_list", t_paths, video_id=video_id, paths=len(embedding_paths))

        t_load = perf_counter()
        frame_embeddings = np.stack(
            [_load_embedding(path) for path in embedding_paths]
        ).astype(np.float32, copy=False)
        _log_ms(
            profile_enabled and profile_every_video,
            "embedding_load_np_stack",
            t_load,
            video_id=video_id,
            frames=len(video_df),
            emb_shape=frame_embeddings.shape,
        )

        t_contig = perf_counter()
        frame_embeddings = np.ascontiguousarray(frame_embeddings, dtype=np.float32)
        _log_ms(profile_enabled and profile_every_video, "embedding_contiguous", t_contig, video_id=video_id)

        t_sim = perf_counter()
        S = np.ascontiguousarray(query_embeddings @ frame_embeddings.T, dtype=np.float32)
        _log_ms(profile_enabled and profile_every_video, "similarity_matmul", t_sim, video_id=video_id, S_shape=S.shape)

        t_ts = perf_counter()
        timestamps = video_df["timestamp_sec"].to_numpy(dtype=np.float32)
        _log_ms(profile_enabled and profile_every_video, "timestamps_to_numpy", t_ts, video_id=video_id, timestamps=len(timestamps))

        matches = _temporal_topk_dp(
            S=S,
            timestamps=timestamps,
            duration_limit=duration_limit,
            max_sequences=max_sequences_per_video,
            overlap_threshold=overlap_threshold,
            profile=profile_enabled and profile_every_video,
            video_id=str(video_id),
        )

        if not matches:
            _log_ms(profile_enabled and profile_every_video, "video_done_no_match", t_video, video_id=video_id, frames=len(video_df))
            continue

        t_build_matches = perf_counter()
        for score, selected_indices in matches:
            selected_rows = video_df.iloc[selected_indices]

            start_time = float(selected_rows["timestamp_sec"].min())
            end_time = float(selected_rows["timestamp_sec"].max())

            selected_keyframes = []

            for qi, (_, selected_row) in enumerate(selected_rows.iterrows()):
                row_dict = _clean_row_dict(selected_row.to_dict())
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

        _log_ms(
            profile_enabled and profile_every_video,
            "build_video_matches",
            t_build_matches,
            video_id=video_id,
            matches=len(matches),
        )
        _log_ms(
            profile_enabled and profile_every_video,
            "video_total",
            t_video,
            video_id=video_id,
            raw_frames=raw_len,
            unique_frames=len(video_df),
            matches=len(matches),
        )

    rows = []

    sorted_results = sorted(
        results,
        key=lambda x: x.avg_score,
        reverse=True,
    )[:top_k_videos]

    t_sort_results = perf_counter()

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

    _log_ms(
        profile_enabled,
        "sort_and_build_output",
        t_sort_results,
        total_sequences=len(results),
        returned=len(rows),
    )

    t_df = perf_counter()
    output_df = pd.DataFrame(rows)
    _log_ms(profile_enabled, "output_dataframe", t_df, rows=len(output_df))
    _log_ms(
        profile_enabled,
        "temporal_search_total",
        t_total,
        videos=video_count,
        skipped_too_short=skipped_too_short,
        sequences=len(results),
        returned=len(output_df),
    )

    return output_df
