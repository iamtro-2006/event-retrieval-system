from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Callable

import numpy as np
import pandas as pd
from numba import njit

from src.retrieval.retriever.semantic_search.pipeline.search import clean_queries, multi_query_search


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
# BUG FIX: `FaissIndex.encode_texts()` dedupes its input via `clean_queries()`
# before encoding, but temporal search needs 1 embedding per sub-query
# PER EVENT, positionally aligned with the flattened (non-deduped) event
# list - `all_queries = [q for event in events for q in event]`. If two
# different events happen to share the exact same query text (e.g. event 1
# and event 3 both say "a car"), a naive `index.encode_texts(all_queries)`
# followed by a positional slice silently returns the WRONG embedding for
# every query after the first duplicate (or crashes with a shape mismatch
# further down in `aggregate_multi_query`). Fix: encode once (letting
# `encode_texts` dedupe internally, which is fine/cheaper), then look each
# original query back up by its cleaned text instead of by position.
# =====================================================================
def encode_events_dedup_safe(encode_texts: Callable[[list[str]], np.ndarray], events: list[list[str]]) -> np.ndarray:
    """Encode every sub-query across `events`, returned 1:1 aligned with the
    flattened (non-deduped) query list, safe even when the same query text
    appears in more than one event.

    Args:
        encode_texts: `FaissIndex.encode_texts` (or any callable with the
            same `list[str] -> np.ndarray` contract, already deduping via
            `clean_queries` internally).
        events: `Orchestrator.build_query_plan(...).events` shape.
    """
    all_queries = [query for event in events for query in event]
    if not all_queries:
        return np.empty((0, 0), dtype=np.float32)

    unique_queries = clean_queries(all_queries)
    unique_embeddings = np.asarray(encode_texts(all_queries), dtype=np.float32)
    if len(unique_embeddings) != len(unique_queries):
        raise RuntimeError(
            "encode_texts() returned "
            f"{len(unique_embeddings)} embeddings for {len(unique_queries)} "
            "deduplicated queries - backend/registry contract mismatch."
        )
    lookup = {q.casefold(): unique_embeddings[i] for i, q in enumerate(unique_queries)}

    rows = []
    for query in all_queries:
        cleaned = clean_queries([query])
        key = cleaned[0].casefold() if cleaned else ""
        if key not in lookup:
            raise RuntimeError(f"Query '{query}' was dropped during cleaning/encoding.")
        rows.append(lookup[key])
    return np.stack(rows).astype(np.float32, copy=False)


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


# =====================================================================
# "Advanced search" temporal on/off (Orchestrator.advanced_search): thay vì
# mỗi model tick trong checklist tự chạy nguyên 1 lượt DP alignment độc lập
# trong không gian embedding riêng rồi mới RRF kết quả VIDEO ở cuối, giờ chỉ
# có 1 công tắc "temporal" on/off dùng CHUNG danh sách semantic model đã
# tick: mỗi model build candidate per-event bằng FAISS (thuật toán retrieval
# giữ nguyên, không đổi), các candidate đó được fuse bằng RRF theo
# (video_id, keyframe_id, sub_query_idx) TRƯỚC, rồi DP alignment chỉ chạy
# ĐÚNG 1 LẦN trên candidate pool đã fuse - xem `SearchPipeline.search_combined`.
# =====================================================================
def build_combined_temporal_candidates(
    resolve_index: Callable[[str], object],
    model_keys: list[str],
    events: list[list[str]],
    candidate_k: int,
    rrf_k: int = 60,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Build per-event FAISS candidates for each of `model_keys` independently
    (unchanged single-model retrieval algorithm), then fuse those candidate
    lists into ONE pool per event via Reciprocal Rank Fusion, BEFORE any DP
    alignment runs.

    Candidates are matched across models by CONTENT identity
    (video_id, keyframe_id, sub_query_idx), never by `_faiss_id` - `_faiss_id`
    is just a row index into that model's own FAISS index, so the same
    keyframe has a different `_faiss_id` in each model's index.

    Args:
        resolve_index: `SearchPipeline._resolve` (or any `model_key ->
            FaissIndex` callable) - used to fetch each model's `FaissIndex`.
        model_keys: The ticked semantic model_keys to combine (same list used
            for the semantic checklist - there is no separate temporal
            checklist anymore).
        candidate_k: FAISS candidate pool size per event, per model.
        rrf_k: RRF damping constant (see `common.scoring.reciprocal_rank_fusion`;
            same default/meaning, applied here at the per-event candidate
            level instead of the final cross-method fusion).

    Returns:
        `(combined_df, per_model_candidates)` - `combined_df` has one row per
        (video_id, keyframe_id, sub_query_idx) with a fused `rrf_score`
        column (used IN PLACE OF a real dot-product for DP alignment, since
        different models don't share an embedding space); `per_model_candidates`
        is `{model_key: candidate_df}`, the pre-fuse breakdown for each model,
        for debug/UI display only.
    """
    per_model: dict[str, pd.DataFrame] = {}
    for model_key in model_keys:
        index = resolve_index(model_key)
        embeddings = encode_events_dedup_safe(index.encode_texts, events)
        df = build_temporal_candidates(
            index.index, index.search_lock, index.metadata_records, events, embeddings, candidate_k
        )
        if df.empty:
            continue
        df = df.copy()
        df["model_key"] = model_key
        per_model[model_key] = df

    if not per_model:
        return pd.DataFrame(), {}

    keyframe_col = "keyframe_id_int" if all("keyframe_id_int" in df.columns for df in per_model.values()) else "keyframe_id"

    fused: dict[tuple, dict] = {}
    for model_key, df in per_model.items():
        for row in df.to_dict(orient="records"):
            identity = (str(row.get("video_id")), row.get(keyframe_col), int(row["sub_query_idx"]))
            rank = int(row["candidate_rank"])
            contribution = 1.0 / (rrf_k + rank)
            entry = fused.get(identity)
            if entry is None:
                entry = {"row": row, "rrf_score": 0.0, "source_models": []}
                fused[identity] = entry
            entry["rrf_score"] += contribution
            entry["source_models"].append(model_key)
            # Keep the metadata row from whichever model scored this
            # candidate best (metadata columns are model-agnostic/identical
            # across models for the same keyframe, this only affects which
            # model's `candidate_score` ends up displayed for debug).
            if row.get("candidate_score", -np.inf) > entry["row"].get("candidate_score", -np.inf):
                entry["row"] = {**row, **{k: v for k, v in entry["row"].items() if k in ("model_key",)}}

    rows = []
    for entry in fused.values():
        item = dict(entry["row"])
        item["rrf_score"] = float(entry["rrf_score"])
        item["source_models"] = sorted(set(entry["source_models"]))
        item.pop("model_key", None)  # combined row spans several models now
        rows.append(item)

    combined = pd.DataFrame.from_records(rows)
    return combined, per_model


def temporal_search_from_score_candidates(
    combined_df: pd.DataFrame,
    sub_queries: list[str],
    duration_limit: float = -1,
    top_k_videos: int = 10,
    max_sequences_per_video: int = 3,
    overlap_threshold: float = 0.6,
    score_col: str = "rrf_score",
) -> pd.DataFrame:
    """DP sequence alignment over a candidate pool whose per-(video, frame,
    sub_query_idx) score is ALREADY KNOWN (`score_col`, e.g. the fused
    `rrf_score` from `build_combined_temporal_candidates`), instead of a live
    dot-product against a vector cache.

    Used for the combined multi-model temporal search: since different
    models don't share an embedding space, there is no real similarity to
    compute between a query and a frame that model didn't itself retrieve
    for that event - those cells are left at `-inf` rather than guessed, so
    the DP can only align through frames that were actually retrieved for
    that specific event by at least one ticked model.
    """
    required = {"video_id", "timestamp_sec", "sub_query_idx", score_col}
    missing = required - set(combined_df.columns)
    if missing:
        raise ValueError(f"Missing columns for combined temporal search: {missing}")

    m = len(sub_queries)
    if m == 0 or combined_df.empty:
        return pd.DataFrame()

    keyframe_col = "keyframe_id_int" if "keyframe_id_int" in combined_df.columns else "keyframe_id"
    results: list[TemporalMatch] = []

    for video_id, video_df in combined_df.groupby("video_id", sort=False):
        frames = (
            video_df.drop_duplicates(subset=[keyframe_col])
            .sort_values("timestamp_sec")
            .reset_index(drop=True)
        )
        n = len(frames)
        if n < m:
            continue

        frame_pos = {kf: j for j, kf in enumerate(frames[keyframe_col].tolist())}
        timestamps = pd.to_numeric(frames["timestamp_sec"], errors="coerce").fillna(0).to_numpy(np.float32)
        records = frames.to_dict(orient="records")

        S = np.full((m, n), -np.inf, dtype=np.float32)
        for sub_query_idx, event_df in video_df.groupby("sub_query_idx"):
            qi = int(sub_query_idx)
            if qi < 0 or qi >= m:
                continue
            for kf, score in zip(event_df[keyframe_col], event_df[score_col]):
                j = frame_pos.get(kf)
                if j is not None:
                    S[qi, j] = max(S[qi, j], float(score))

        matches = _temporal_topk_dp(S, timestamps, duration_limit, max_sequences_per_video, overlap_threshold)
        if not matches:
            continue

        for match_score, path in matches:
            selected = []
            for qi, local_frame_idx in enumerate(path):
                row = _clean_row(records[local_frame_idx])
                row.update(sub_query_idx=qi, sub_query=str(sub_queries[qi]), score=float(S[qi, local_frame_idx]))
                selected.append(row)

            start = float(timestamps[path[0]])
            end = float(timestamps[path[-1]])
            results.append(
                TemporalMatch(
                    video_id=str(video_id),
                    score=float(match_score),
                    avg_score=float(match_score / m),
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

    def _resolve(self, model_key: str | None = None):
        """Trả về `FaissIndex` thật sự sẽ dùng cho lần search này (`self.index`
        có thể là 1 `FaissIndex` đơn hoặc 1 `IndexManager` nhiều model)."""
        get = getattr(self.index, "get", None)
        return get(model_key) if callable(get) else self.index

    def search(
        self,
        events: list[list[str]],
        top_k: int = 10,
        candidate_k: int = 500,
        duration_limit: float = -1,
        model_key: str | None = None,
    ) -> pd.DataFrame:
        """Search một chuỗi event tuần tự (mỗi event là list sub-query),
        dùng cho `mode="temporal"` của orchestrator.

        Thin: encode toàn bộ sub-query của mọi event trong 1 batch (model call
        duy nhất), sau đó giao toàn bộ phần build-candidate + DP alignment cho
        `temporal_search_from_events` ở trên.

        Args:
            model_key: Model nào để search (chỉ có ý nghĩa khi pipeline được
                build trên 1 `IndexManager` nhiều model); bỏ trống = model
                mặc định.
        """
        if not events:
            return pd.DataFrame()

        index = self._resolve(model_key)
        # See `encode_events_dedup_safe` docstring: must NOT positionally
        # slice `index.encode_texts(all_queries)` directly, since that
        # dedupes internally and would misalign whenever two events share
        # the exact same query text.
        all_embeddings: np.ndarray = encode_events_dedup_safe(index.encode_texts, events)

        df = temporal_search_from_events(
            index.index,
            index.search_lock,
            index.metadata_records,
            index.index_vectors,
            index.allow_npy_fallback,
            events,
            all_embeddings,
            top_k,
            candidate_k,
            duration_limit,
        )
        if not df.empty:
            df["model_key"] = index.model_key
        return df

    def search_combined(
        self,
        events: list[list[str]],
        model_keys: list[str],
        top_k: int = 10,
        candidate_k: int = 500,
        duration_limit: float = -1,
        rrf_k: int = 60,
    ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        """Temporal search kết hợp NHIỀU model semantic đã tick, dùng cho
        công tắc on/off `temporal` của `Orchestrator.advanced_search` (thay
        cho thiết kế cũ mỗi model chạy 1 lượt DP riêng).

        Mỗi model trong `model_keys` build candidate per-event bằng FAISS
        độc lập (thuật toán retrieval KHÔNG đổi), các candidate đó được fuse
        bằng RRF theo (video_id, keyframe_id, sub_query_idx) TRƯỚC
        (`build_combined_temporal_candidates`), rồi DP alignment chạy ĐÚNG 1
        LẦN trên candidate pool đã fuse (`temporal_search_from_score_candidates`)
        — dùng `rrf_score` thay cho dot-product thật, vì các model không
        chia sẻ chung 1 không gian embedding.

        Args:
            model_keys: model_key đã tick ở checklist semantic — cùng danh
                sách đó được dùng làm input cho temporal khi bật, không còn
                checklist temporal riêng.

        Returns:
            `(fused_df, per_model_candidates)` — `fused_df` shape giống hệt
            `search()`; `per_model_candidates` là `{model_key: candidate_df}`
            breakdown từng model TRƯỚC khi fuse, chỉ để debug/hiển thị (KHÔNG
            tự tham gia RRF cấp method của `Orchestrator.advanced_search`).
        """
        if not events or not model_keys:
            return pd.DataFrame(), {}

        candidate_k = max(int(candidate_k), int(top_k))
        candidate_df, per_model_candidates = build_combined_temporal_candidates(
            self._resolve, model_keys, events, candidate_k, rrf_k=rrf_k
        )
        if candidate_df.empty:
            return pd.DataFrame(), per_model_candidates

        sub_queries = [event[0] for event in events if event]
        fused_df = temporal_search_from_score_candidates(
            candidate_df,
            sub_queries,
            duration_limit=duration_limit,
            top_k_videos=top_k,
            max_sequences_per_video=3,
            overlap_threshold=0.6,
            score_col="rrf_score",
        )
        if not fused_df.empty:
            fused_df["model_key"] = "+".join(model_keys)
            fused_df["search_mode"] = "temporal"
        return fused_df, per_model_candidates