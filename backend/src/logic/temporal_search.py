from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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


# =====================================================================
# TỐI ƯU C-LEVEL 1: Thuật toán DP Dịch bằng Numba JIT (Siêu Tốc Độ)
# Bỏ qua GIL, không cấp phát mảng trung gian gây tốn RAM.
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
            # Tính max cộng dồn từ các frame trước đó
            if j > 0:
                prev_j = j - 1
                if dp_prev[prev_j] > current_max_val:
                    current_max_val = dp_prev[prev_j]
                    current_max_idx = prev_j
            
            # Nếu có chuỗi hợp lệ thì cộng điểm
            if current_max_idx != -1:
                dp_cur[j] = current_max_val + S[qi, j]
                parents[qi, j] = current_max_idx
                
        dp_prev = dp_cur

    # Tìm điểm kết thúc chuỗi có score cao nhất
    best_score = -np.inf
    last = -1
    for j in range(n):
        if dp_prev[j] > best_score:
            best_score = dp_prev[j]
            last = j

    if last == -1 or not np.isfinite(best_score):
        return -np.inf, np.empty(0, dtype=np.int32)

    # Backtrack tìm đường đi ngược
    path = np.empty(m, dtype=np.int32)
    path[m - 1] = last
    for qi in range(m - 1, 0, -1):
        last = parents[qi, last]
        if last < 0:
            return -np.inf, np.empty(0, dtype=np.int32)
        path[qi - 1] = last

    return float(best_score), path


# Wrapper kết nối Numba Core với code Python
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

            candidates.append((score, path, float(timestamps[path[0]]), float(timestamps[path[-1]])))

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
                candidates.append((score, path, float(timestamps[path[0]]), float(timestamps[path[-1]])))

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


def temporal_search_from_candidates(
    query_embeddings: np.ndarray,
    sub_queries: list[str],
    candidate_df: pd.DataFrame,
    duration_limit: float = -1,
    top_k_videos: int = 10,
    max_sequences_per_video: int = 3,
    overlap_threshold: float = 0.6,
    embedding_matrix: np.ndarray | np.memmap | None = None,
    allow_npy_fallback: bool = False,
) -> pd.DataFrame:
    required = {"video_id", "timestamp_sec"}
    if not required.issubset(candidate_df.columns):
        raise ValueError(f"Missing columns for temporal search: {required - set(candidate_df.columns)}")

    if len(sub_queries) != len(query_embeddings):
        raise ValueError("sub_queries/query_embeddings length mismatch")

    query_embeddings = np.ascontiguousarray(query_embeddings, dtype=np.float32)
    results: list[TemporalMatch] = []

    # =====================================================================
    # TỐI ƯU C-LEVEL 2: Tiêu diệt Overhead của Pandas
    # 1. Map Dict CHỈ 1 LẦN.
    # 2. Xử lý cắt mảng hoàn toàn bằng NumPy Con trỏ (Numpy Slicing)
    # =====================================================================
    all_records = candidate_df.to_dict(orient="records")
    
    v_ids = candidate_df["video_id"].to_numpy()
    unique_vids = np.unique(v_ids)

    faiss_id_col = "_faiss_id" if "_faiss_id" in candidate_df.columns else "keyframe_id"
    faiss_ids = candidate_df[faiss_id_col].to_numpy(dtype=np.int64)
    timestamps_all = pd.to_numeric(candidate_df["timestamp_sec"], errors="coerce").fillna(0).to_numpy(np.float32)
    
    emb_paths_all = candidate_df["embedding_path"].to_numpy() if (allow_npy_fallback and "embedding_path" in candidate_df.columns) else None

    for video_id in unique_vids:
        # Lấy Index của video này bằng Array Mask (O(1) Memory)
        idx_for_vid = np.where(v_ids == video_id)[0]

        # 1. Sort theo timestamp bằng argsort
        sub_times = timestamps_all[idx_for_vid]
        sort_order = np.argsort(sub_times, kind="mergesort")
        idx_for_vid = idx_for_vid[sort_order]

        # 2. Drop duplicates bằng faiss_id (Thay thế DataFrame.drop_duplicates)
        sub_faiss = faiss_ids[idx_for_vid]
        _, unique_local_idx = np.unique(sub_faiss, return_index=True)
        idx_for_vid = idx_for_vid[np.sort(unique_local_idx)]

        if len(idx_for_vid) < len(query_embeddings):
            continue

        sub_faiss_final = faiss_ids[idx_for_vid]
        sub_times_final = timestamps_all[idx_for_vid]

        # 3. Tra cứu Vector
        if embedding_matrix is not None:
            if np.any(sub_faiss_final < 0) or np.any(sub_faiss_final >= len(embedding_matrix)):
                raise IndexError(f"Invalid FAISS ID in temporal candidates for video_id={video_id}")
            frame_embeddings = np.asarray(embedding_matrix[sub_faiss_final], dtype=np.float32)
        elif emb_paths_all is not None:
            frame_embeddings = np.stack([_load_embedding(emb_paths_all[i]) for i in idx_for_vid])
        else:
            raise RuntimeError("Temporal search needs vector cache by _faiss_id.")

        # 4. Nhân Ma trận
        S = (query_embeddings @ frame_embeddings.T).astype(np.float32, copy=False)

        # 5. Gọi Numba DP Tìm đường tốt nhất
        matches = _temporal_topk_dp(S, sub_times_final, duration_limit, max_sequences_per_video, overlap_threshold)
        if not matches:
            continue

        # 6. Rút trích kết quả trả về
        for score, path in matches:
            selected = []
            for qi, local_frame_idx in enumerate(path):
                global_idx = idx_for_vid[local_frame_idx]
                row = _clean_row(all_records[global_idx])
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

    rows = []
    for rank, item in enumerate(results[:top_k_videos], 1):
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