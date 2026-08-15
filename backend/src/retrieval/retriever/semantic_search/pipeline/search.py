# Pipeline "mong" theo dung nghia: file nay chua TOAN BO thuat toan search FAISS
# thuan tuy duoi dang cac ham module rieng le (faiss_search, results_for_queries,
# aggregate_multi_query, multi_query_search, image_similarity_search) — khong ham
# nao trong so nay duoc phep khoi tao FAISS index, load model, hay tu goi model
# encode. Chung chi nhan `index`/`search_lock`/`metadata_records` (state da duoc
# `ClipFaissIndex` khoi tao san) + embedding da encode san lam tham so, roi tra
# ve `pd.DataFrame`.
#
# `SearchPipeline` o cuoi file la entrypoint DUY NHAT cua module nay (khong con
# class `SemanticSearch` rieng nua — xem `models/` da bi xoa). No nhan thang
# `ClipFaissIndex` (giong het cach `ocr_search.pipeline.search.SearchPipeline`
# nhan `OCRRepository` va `asr_search.pipeline.search.SearchPipeline` nhan
# `ASRRepository`), tu goi `index.encode_texts()/encode_image()` de co embedding
# roi giao cho cac ham thuan o tren de tinh toan. Duoc build qua
# `factory.build_semantic_search_pipeline(index)`, luu tren `Orchestrator` la
# `self.semantic_search`, goi thong nhat qua `.search(...)` giong ocr/asr.
from __future__ import annotations

import re
from pathlib import Path
from threading import RLock

import numpy as np
import pandas as pd


def clean_queries(queries: list[str]) -> list[str]:
    """Clean, deduplicate, and normalize a list of query strings.

    Pure string logic used by every search function in this module (and
    re-exported by `semantic_index.py` for backward compatibility) — not a
    model/index concern.
    """
    cleaned: list[str] = []
    seen: set[str] = set()
    for query in queries:
        query = re.sub(r"\s+", " ", str(query or "").strip())
        key = query.casefold()
        if query and key not in seen:
            seen.add(key)
            cleaned.append(query)
    return cleaned


def faiss_search(index, search_lock: RLock, embeddings: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Execute a thread-safe FAISS similarity search against an already-built index.

    Pure search-algorithm step: takes the already-loaded `faiss.Index` and an
    already-computed embedding matrix, does not load/encode anything itself.
    """
    k = min(max(1, int(k)), index.ntotal)
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    with search_lock:
        return index.search(embeddings, k)


def results_for_queries(
    index, search_lock: RLock, embeddings: np.ndarray, candidate_k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Retrieve raw FAISS scores and indices for a batch of query embeddings."""
    if embeddings.size == 0:
        return np.empty((0, 0), np.float32), np.empty((0, 0), np.int64)
    return faiss_search(index, search_lock, embeddings, candidate_k)


def aggregate_multi_query(
    scores: np.ndarray,
    indices: np.ndarray,
    queries: list[str],
    top_k: int,
    metadata_records: list[dict],
) -> pd.DataFrame:
    """Aggregate and rank multi-query FAISS results using vectorized NumPy ops.

    Args:
        scores: Matrix of similarity scores (n_queries, candidate_k).
        indices: Matrix of FAISS indices (n_queries, candidate_k).
        queries: List of original query strings.
        top_k: Number of top results to return.
        metadata_records: `SemanticIndex._metadata_records` — one dict per
            FAISS row, keyed by FAISS id (positional).

    Returns:
        A DataFrame containing the aggregated, ranked metadata records.
    """
    if indices.size == 0:
        return pd.DataFrame()

    n_queries = len(queries)
    valid = indices >= 0
    if not np.any(valid):
        return pd.DataFrame()

    flat_ids = indices[valid].astype(np.int32, copy=False)
    flat_scores = scores[valid].astype(np.float32, copy=False)
    query_ids = np.repeat(np.arange(n_queries, dtype=np.int32), indices.shape[1])[valid.ravel()]

    order = np.lexsort((query_ids, flat_ids))
    ids_sorted = flat_ids[order]
    scores_sorted = flat_scores[order]
    q_sorted = query_ids[order]

    id_change = np.ones(len(ids_sorted), dtype=np.bool_)
    id_change[1:] = ids_sorted[1:] != ids_sorted[:-1]
    starts = np.where(id_change)[0]

    unique_ids = ids_sorted[starts]
    counts = np.diff(np.append(starts, len(ids_sorted)))

    score_sum = np.add.reduceat(scores_sorted, starts)
    max_score = np.maximum.reduceat(scores_sorted, starts)
    avg_score = score_sum / counts

    q_change = np.ones(len(q_sorted), dtype=np.bool_)
    q_change[1:] = q_sorted[1:] != q_sorted[:-1]

    is_new_unique_q = q_change & ~id_change
    extra_unique_q = np.add.reduceat(is_new_unique_q.astype(np.int32), starts)
    matched = 1 + extra_unique_q

    coverage = matched.astype(np.float32) / max(1, n_queries)
    alignment = 0.8 * avg_score + 0.2 * coverage

    rank_order = np.argsort(-alignment, kind="stable")[:top_k]

    rows: list[dict] = []
    for display_rank, pos in enumerate(rank_order, 1):
        idx = int(unique_ids[pos])
        item = metadata_records[idx].copy()
        item["avg_score"] = float(avg_score[pos])
        item["max_score"] = float(max_score[pos])
        item["matched_queries"] = int(matched[pos])
        item["coverage_score"] = float(coverage[pos])
        item["alignment_score"] = float(alignment[pos])
        item["retrieval_score"] = float(alignment[pos])
        item["display_rank"] = display_rank
        item["rank"] = display_rank
        rows.append(item)

    return pd.DataFrame.from_records(rows)


def multi_query_search(
    index,
    search_lock: RLock,
    metadata_records: list[dict],
    queries: list[str],
    embeddings: np.ndarray,
    top_k: int = 10,
    candidate_k: int | None = None,
) -> pd.DataFrame:
    """Execute a multi-query FAISS search and aggregate the results.

    Pure search-algorithm entry point: `embeddings` must already be encoded
    (by `SemanticIndex.encode_texts`) and aligned 1:1 with `queries` — this
    function never calls a model itself.

    IMPORTANT: `queries` must already be deduplicated/cleaned (e.g. via
    `clean_queries()`) by the caller, in the SAME order used to compute
    `embeddings`. This function deliberately does NOT call `clean_queries()`
    again here: doing so previously shrank `queries` while `embeddings`
    (and the `scores`/`indices` derived from it) still had one row per
    ORIGINAL query, breaking the 1:1 alignment `aggregate_multi_query()`
    depends on (`np.repeat(np.arange(len(queries)), ...)` sized for the
    shorter deduped list, boolean-indexed against arrays sized for the
    longer raw one -> `IndexError`). See
    `tests_manual/test_multi_query_dedupe.py` for a repro. Every current
    caller (`SearchPipeline.multi_query_search()`/`.search()`,
    `build_temporal_candidates()`) already dedupes before encoding, so this
    is a no-op for them either way.
    """
    if not queries:
        return pd.DataFrame()
    candidate_k = max(int(candidate_k or top_k), int(top_k))
    scores, indices = results_for_queries(index, search_lock, embeddings, candidate_k)
    return aggregate_multi_query(scores, indices, queries, int(top_k), metadata_records)


def image_similarity_search(
    index,
    search_lock: RLock,
    metadata_records: list[dict],
    image_embedding: np.ndarray,
    image_label: str,
    top_k: int = 20,
) -> pd.DataFrame:
    """Execute an image-to-image similarity search from an already-encoded query embedding."""
    scores, indices = faiss_search(index, search_lock, image_embedding, top_k)
    rows = []
    for rank, idx in enumerate(indices[0], 1):
        if idx < 0:
            continue
        item = dict(metadata_records[int(idx)])
        item.update(
            rank=rank,
            display_rank=rank,
            score=float(scores[0, rank - 1]),
            retrieval_score=float(scores[0, rank - 1]),
            query=str(image_label),
        )
        rows.append(item)
    return pd.DataFrame.from_records(rows)


class SearchPipeline:
    """Semantic (text/image -> keyframe) search trên một hoặc nhiều `FaissIndex`.

    Entrypoint duy nhất của `semantic_search/` — build qua
    `factory.build_semantic_search_pipeline(index)`, với `index` là 1
    `FaissIndex` đơn (tương thích ngược) HOẶC 1 `IndexManager` (nhiều
    model — mỗi method ở đây nhận thêm `model_key` optional để chọn model
    nào search; bỏ trống thì dùng `IndexManager.default_model_key`). Mọi
    "model call" (encode) đi qua `FaissIndex` đã resolve; mọi thuật toán
    FAISS + ranking nằm ở các hàm thuần phía trên trong file này, class này
    chỉ điều phối 2 bước đó — cùng hình dạng với
    `ocr_search.pipeline.search.SearchPipeline` /
    `asr_search.pipeline.search.SearchPipeline` để `Orchestrator` gọi thống
    nhất qua `self.semantic_search.search(...)`.
    """

    def __init__(self, index) -> None:
        self.index = index

    def _resolve(self, model_key: str | None = None):
        """Trả về `FaissIndex` thật sự sẽ dùng cho lần search này."""
        get = getattr(self.index, "get", None)
        return get(model_key) if callable(get) else self.index

    def multi_query_search(
        self,
        queries: list[str],
        top_k: int = 10,
        candidate_k: int | None = None,
        query_embeddings: np.ndarray | None = None,
        model_key: str | None = None,
    ) -> pd.DataFrame:
        index = self._resolve(model_key)
        if query_embeddings is not None:
            # Caller computed these embeddings themselves -> trust they did
            # so for exactly this `queries` list/order (documented
            # precondition, see module-level `multi_query_search()`
            # docstring); don't re-clean underneath them.
            embeddings = query_embeddings
        else:
            # We own the encode step here -> dedupe FIRST so `queries` and
            # the embeddings we compute stay 1:1 aligned (see
            # tests_manual/test_multi_query_dedupe.py for what goes wrong
            # if a duplicate-containing list reaches encode_texts() first).
            queries = clean_queries(queries)
            embeddings = index.encode_texts(queries) if queries else np.empty((0, 0), dtype=np.float32)
        df = multi_query_search(
            index.index,
            index.search_lock,
            index.metadata_records,
            queries,
            embeddings,
            top_k,
            candidate_k,
        )
        if not df.empty:
            df["model_key"] = index.model_key
        return df

    def similarity_search_by_image(
        self, image_path: str | Path, top_k: int = 20, model_key: str | None = None
    ) -> pd.DataFrame:
        index = self._resolve(model_key)
        embedding = index.encode_image(image_path)
        df = image_similarity_search(
            index.index, index.search_lock, index.metadata_records, embedding, str(image_path), top_k
        )
        if not df.empty:
            df["model_key"] = index.model_key
        return df

    def search(
        self,
        events: list[list[str]],
        top_k: int = 10,
        candidate_k: int = 500,
        model_key: str | None = None,
    ) -> pd.DataFrame:
        """Search theo danh sách event (mỗi event là list sub-query), dùng cho
        `mode="semantic"` của orchestrator.

        Args:
            events: Danh sách event, mỗi event là list các sub-query (xem
                `Orchestrator.build_query_plan`).
            top_k: Số kết quả trả về.
            candidate_k: Kích thước candidate pool FAISS trước khi aggregate.
            model_key: Model nào để search (chỉ có ý nghĩa khi pipeline được
                build trên 1 `IndexManager` nhiều model); bỏ trống = model
                mặc định.
        """
        index = self._resolve(model_key)
        queries = clean_queries([query for event in events for query in event])
        embeddings = index.encode_texts(queries)
        return self.multi_query_search(queries, top_k, candidate_k, embeddings, model_key=model_key)
