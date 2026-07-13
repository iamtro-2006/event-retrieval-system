from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock

import numpy as np
import pandas as pd

from src.retrieval.retriever.semantic_search.pipeline.search import multi_query_search

try:
    from src.ui.temporal_dp_numba import (
        run_dp_on_window as _numba_run_dp,
        has_near_tied_suffix_choices,
        window_temporal_candidates,
        suffix_temporal_candidates,
        select_non_overlapping,
    )
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False

try:
    from src.ui.embedding_memmap import EmbeddingMemmapStore
except ImportError:
    EmbeddingMemmapStore = None


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


# ---------------------------------------------------------------------------
# Pure-Python fallback (used when Numba is unavailable)
# ---------------------------------------------------------------------------
def _run_dp_on_window_python(S: np.ndarray) -> tuple[float, list[int]]:
    """O(m*n) strict-order DP — pure Python fallback when Numba is unavailable."""
    m, n = S.shape
    if m == 0 or n < m:
        return -np.inf, []

    dp = np.full((m, n), -np.inf, dtype=np.float32)
    parent = np.full((m, n), -1, dtype=np.int32)
    dp[0] = S[0]

    for qi in range(1, m):
        best_prev_score = -np.inf
        best_prev_idx = -1
        for fj in range(n):
            if fj > 0:
                prev_val = dp[qi - 1, fj - 1]
                if prev_val > best_prev_score:
                    best_prev_score = prev_val
                    best_prev_idx = fj - 1
            if best_prev_idx >= 0 and np.isfinite(best_prev_score):
                dp[qi, fj] = best_prev_score + S[qi, fj]
                parent[qi, fj] = best_prev_idx

    best_score = -np.inf
    last = -1
    for j in range(n):
        if dp[m - 1, j] > best_score:
            best_score = dp[m - 1, j]
            last = j

    if last == -1 or not np.isfinite(best_score):
        return -np.inf, []

    path = [0] * m
    path[m - 1] = last
    for qi in range(m - 1, 0, -1):
        last = parent[qi, last]
        if last < 0:
            return -np.inf, []
        path[qi - 1] = last

    return float(best_score), path


def _run_dp_on_window(S: np.ndarray) -> tuple[float, list[int]]:
    if _HAS_NUMBA:
        score, path_arr, path_len = _numba_run_dp(S)
        return float(score), path_arr[:path_len].tolist() if path_len > 0 else []
    return _run_dp_on_window_python(S)


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

    if _HAS_NUMBA:
        # Use full Numba kernel pipeline: window or suffix DP + vectorized NMS
        use_window = duration_limit >= 0 or not has_near_tied_suffix_choices(S)
        if use_window:
            scores, paths_flat, path_offsets, start_times, end_times, path_len = window_temporal_candidates(
                S, timestamps, float(duration_limit)
            )
        else:
            scores, paths_flat, path_offsets, start_times, end_times, path_len = suffix_temporal_candidates(
                S, timestamps
            )

        if len(scores) == 0:
            return []

        sel_indices = select_non_overlapping(
            scores, paths_flat, path_offsets, start_times, end_times,
            path_len, max_sequences, overlap_threshold
        )

        results = []
        for idx in sel_indices:
            offset = path_offsets[idx]
            path = paths_flat[offset: offset + path_len].tolist()
            results.append((float(scores[idx]), path))
        return results

    # Pure-Python fallback
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
    memmap_store=None,
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
    
    has_emb_paths = "embedding_path" in candidate_df.columns
    emb_paths_all = candidate_df["embedding_path"].to_numpy() if has_emb_paths and (memmap_store is not None or allow_npy_fallback) else None

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

        # 3. Tra cứu Vector (memmap_store > embedding_matrix > npy fallback)
        if memmap_store is not None and emb_paths_all is not None:
            paths_for_vid = [emb_paths_all[i] for i in idx_for_vid]
            frame_embeddings = memmap_store.get_batch(paths_for_vid)
        elif embedding_matrix is not None:
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


# =====================================================================
# Orchestration ban dau nam trong `SemanticIndex._build_temporal_candidates`
# va `SemanticIndex.temporal_search` (models/semantic_index.py). Chuyen xuong
# day vi day la logic THUAT TOAN temporal search (khong khoi tao/goi model):
# `all_embeddings`/`event_embeddings` phai duoc SemanticIndex.encode_texts()
# tinh san roi truyen vao, ham o day chi goi FAISS (qua
# `semantic_search.pipeline.search.multi_query_search`) + DP numba o tren.
# =====================================================================
def build_temporal_candidates(
    index,
    search_lock: RLock,
    metadata_records: list[dict],
    events: list[list[str]],
    all_embeddings: np.ndarray,
    candidate_k: int,
) -> pd.DataFrame:
    """Build a unified candidate DataFrame for all temporal events.

    Runs one `multi_query_search` (plain FAISS aggregation) per event, using
    the pre-computed slice of `all_embeddings` for that event's sub-queries,
    then stacks the per-event candidate frames with bookkeeping columns
    (`sub_query_idx`, `candidate_rank`, ...) used later by `_temporal_topk_dp`.
    """
    offset = 0
    frames: list[pd.DataFrame] = []
    for event_idx, event_queries in enumerate(events):
        count = len(event_queries)
        event_emb = all_embeddings[offset:offset + count]
        offset += count

        event_results = multi_query_search(
            index, search_lock, metadata_records, event_queries, event_emb, candidate_k, candidate_k
        )
        if event_results.empty:
            continue

        event_results["candidate_score"] = event_results["retrieval_score"].to_numpy(np.float32)
        event_results["candidate_rank"] = np.arange(1, len(event_results) + 1, dtype=np.int32)
        event_results["sub_query_idx"] = np.int32(event_idx)
        event_results["sub_query"] = event_queries[0]
        frames.append(event_results)

    return pd.concat(frames, ignore_index=True, copy=False) if frames else pd.DataFrame()


def temporal_search_from_events(
    index,
    search_lock: RLock,
    metadata_records: list[dict],
    vector_cache: np.ndarray | np.memmap | None,
    allow_npy_fallback: bool,
    events: list[list[str]],
    all_embeddings: np.ndarray,
    top_k: int = 10,
    candidate_k: int = 500,
    duration_limit: float = -1,
) -> pd.DataFrame:
    """Execute a temporal search across multiple sequential events.

    Full pure-algorithm entry point for temporal search: builds FAISS
    candidates per event via `build_temporal_candidates`, then runs the DP
    sequence alignment via `temporal_search_from_candidates`. `all_embeddings`
    must already be encoded (1:1 with the flattened `events`) by
    `SemanticIndex.encode_texts` — this function never calls a model itself.
    """
    if not events:
        return pd.DataFrame()

    candidate_k = max(int(candidate_k), int(top_k))

    offsets = np.cumsum([0] + [len(event) for event in events[:-1]])
    event_embeddings = all_embeddings[offsets]
    event_queries = [event[0] for event in events]

    candidate_df = build_temporal_candidates(
        index, search_lock, metadata_records, events, all_embeddings, candidate_k
    )
    if candidate_df.empty:
        return pd.DataFrame()

    return temporal_search_from_candidates(
        query_embeddings=event_embeddings,
        sub_queries=event_queries,
        candidate_df=candidate_df,
        duration_limit=duration_limit,
        top_k_videos=top_k,
        max_sequences_per_video=3,
        overlap_threshold=0.6,
        embedding_matrix=vector_cache,
        allow_npy_fallback=allow_npy_fallback,
    )


class SearchPipeline:
    """Temporal (chuỗi event) search trên một `ClipFaissIndex`.

    Entrypoint duy nhất của `temporal_search/` — build qua
    `factory.build_temporal_search_pipeline(index)`. Compose (không kế thừa)
    `ClipFaissIndex` để encode + truy cập FAISS/metadata/vector-cache; toàn bộ
    thuật toán (candidate build + DP alignment) nằm ở các hàm thuần phía trên.
    Cùng hình dạng với `semantic_search.pipeline.search.SearchPipeline` và
    `ocr_search`/`asr_search` `SearchPipeline` để `Orchestrator` gọi thống
    nhất qua `self.temporal_search.search(...)`.
    """

    def __init__(self, index) -> None:
        self.index = index

    def search(
        self,
        events: list[list[str]],
        top_k: int = 10,
        candidate_k: int = 500,
        duration_limit: float = -1,
    ) -> pd.DataFrame:
        """Search một chuỗi event tuần tự (mỗi event là list sub-query),
        dùng cho `mode="temporal"` của orchestrator.

        Thin: encode toàn bộ sub-query của mọi event trong 1 batch (model call
        duy nhất), sau đó giao toàn bộ phần build-candidate + DP alignment cho
        `temporal_search_from_events` ở trên.
        """
        if not events:
            return pd.DataFrame()

        all_queries: list[str] = [query for event in events for query in event]
        all_embeddings: np.ndarray = self.index.encode_texts(all_queries)

        return temporal_search_from_events(
            self.index.index,
            self.index.search_lock,
            self.index.metadata_records,
            self.index.index_vectors,
            self.index.allow_npy_fallback,
            events,
            all_embeddings,
            top_k,
            candidate_k,
            duration_limit,
        )