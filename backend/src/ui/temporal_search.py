from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


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

    emb = emb.astype(np.float32)
    norm = np.linalg.norm(emb)

    if norm > 1e-12:
        emb = emb / norm

    return emb


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
        best_prev_score = -np.inf
        best_prev_idx = -1

        for fj in range(n):
            if fj > 0 and dp[qi - 1, fj - 1] > best_prev_score:
                best_prev_score = dp[qi - 1, fj - 1]
                best_prev_idx = fj - 1

            if best_prev_idx != -1:
                dp[qi, fj] = best_prev_score + S[qi, fj]
                parent[qi, fj] = best_prev_idx

    last_idx = int(np.argmax(dp[m - 1]))
    best_score = float(dp[m - 1, last_idx])

    if not np.isfinite(best_score):
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
    """

    m, n = S.shape

    if n < m:
        return []

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

        if not local_path or not np.isfinite(local_score):
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


def temporal_search_from_candidates(
    query_embeddings: np.ndarray,
    sub_queries: list[str],
    candidate_df: pd.DataFrame,
    duration_limit: float = -1,
    top_k_videos: int = 10,
    max_sequences_per_video: int = 3,
    overlap_threshold: float = 0.6,
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

    results: list[TemporalMatch] = []

    for video_id, video_df in candidate_df.groupby("video_id"):
        video_df = (
            video_df
            .sort_values("timestamp_sec")
            .drop_duplicates(subset=["keyframe_id"], keep="first")
            .reset_index(drop=True)
        )

        if len(video_df) < len(query_embeddings):
            continue

        frame_embeddings = np.stack(
            [
                _load_embedding(path)
                for path in video_df["embedding_path"].tolist()
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