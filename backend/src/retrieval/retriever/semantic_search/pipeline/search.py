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
from numba import njit, prange


DEFAULT_MULTI_QUERY_BETA = 0.8


def semantic_alignment_scores(
    candidate_similarities: np.ndarray,
    beta: float = DEFAULT_MULTI_QUERY_BETA,
) -> np.ndarray:
    """Apply the canonical semantic scoring policy to a dense query/candidate matrix."""
    similarities = np.asarray(candidate_similarities, dtype=np.float32)
    return np.mean(np.exp(float(beta) * (similarities - 1.0)), axis=0)


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
    candidate_similarities: np.ndarray | None = None,
    beta: float = DEFAULT_MULTI_QUERY_BETA,
) -> pd.DataFrame:
    """Aggregate and rank multi-query FAISS results using vectorized NumPy ops.

    Args:
        scores: Matrix of similarity scores (n_queries, candidate_k).
        indices: Matrix of FAISS indices (n_queries, candidate_k).
        queries: List of original query strings.
        top_k: Number of top results to return.
        metadata_records: `SemanticIndex._metadata_records` — one dict per
            FAISS row, keyed by FAISS id (positional).
        candidate_similarities: Dense matrix with shape
            ``(n_queries, n_unique_candidates)``.  Column order must match
            the ascending FAISS IDs in ``indices``.  Supplying this matrix
            lets every candidate be scored against every query, including
            queries for which it did not enter the FAISS candidate pool.
        beta: Exponential scaling factor in the multi-query score. Defaults
            to ``0.8``.

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

    # Smooth multi-query score requested by the retrieval policy:
    #     score(I) = 1 / |Q| * sum(exp(beta * (sim(q, I) - 1)))  for q in Q
    if candidate_similarities is None:
        raise ValueError(
            "candidate_similarities is required: sparse multi-query scores "
            "are not comparable across candidates"
        )

    candidate_similarities = np.asarray(candidate_similarities, dtype=np.float32)
    expected_shape = (n_queries, len(unique_ids))
    if candidate_similarities.shape != expected_shape:
        raise ValueError(
            "candidate_similarities shape mismatch: "
            f"got {candidate_similarities.shape}, expected {expected_shape}"
        )
    alignment = semantic_alignment_scores(candidate_similarities, beta)

    rank_order = np.argsort(-alignment, kind="stable")[:top_k]

    rows: list[dict] = []
    for display_rank, pos in enumerate(rank_order, 1):
        idx = int(unique_ids[pos])
        item = metadata_records[idx].copy()
        item["avg_score"] = float(avg_score[pos])
        item["max_score"] = float(max_score[pos])
        item["matched_queries"] = int(matched[pos])
        item["coverage_score"] = float(coverage[pos])
        item["exp_mean_score"] = float(alignment[pos])
        item["alignment_score"] = float(alignment[pos])
        item["retrieval_score"] = float(alignment[pos])
        item["display_rank"] = display_rank
        item["rank"] = display_rank
        rows.append(item)

    return pd.DataFrame.from_records(rows)


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def _fill_missing_similarities_numba(
    query_embeddings: np.ndarray,
    candidate_vectors: np.ndarray,
    similarities: np.ndarray,
    known: np.ndarray,
) -> None:
    """Fill unknown query/candidate dot products in parallel, in place."""
    n_queries, dimension = query_embeddings.shape
    n_candidates = candidate_vectors.shape[0]
    for pair_idx in prange(n_queries * n_candidates):
        query_idx = pair_idx // n_candidates
        candidate_idx = pair_idx - query_idx * n_candidates
        if known[query_idx, candidate_idx]:
            continue

        dot = np.float32(0.0)
        for dim_idx in range(dimension):
            dot += (
                query_embeddings[query_idx, dim_idx]
                * candidate_vectors[candidate_idx, dim_idx]
            )
        similarities[query_idx, candidate_idx] = dot


def similarities_for_candidates(
    index,
    search_lock: RLock,
    query_embeddings: np.ndarray,
    candidate_ids: np.ndarray,
    index_vectors: np.ndarray | np.memmap | None = None,
    known_scores: np.ndarray | None = None,
    known_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Compute ``sim(q, I)`` for every query/candidate pair.

    Candidate generation remains approximate/top-k FAISS search, but ranking
    needs a dense score matrix so a candidate is evaluated against all of Q.
    Scores already returned by FAISS are reused. A parallel Numba kernel only
    computes the missing pairs in the query-by-candidate matrix. Prefer the
    configured vector cache and reconstruct only the small union of candidates
    when no cache is available.
    """
    candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
    query_embeddings = np.ascontiguousarray(query_embeddings, dtype=np.float32)
    if candidate_ids.size == 0:
        return np.empty((len(query_embeddings), 0), dtype=np.float32)

    if index_vectors is not None:
        if np.any(candidate_ids < 0) or np.any(candidate_ids >= len(index_vectors)):
            raise IndexError("Invalid FAISS ID while scoring multi-query candidates")
        candidate_vectors = np.asarray(index_vectors[candidate_ids], dtype=np.float32)
    else:
        candidate_vectors = np.empty((len(candidate_ids), index.d), dtype=np.float32)
        with search_lock:
            for row, candidate_id in enumerate(candidate_ids):
                candidate_vectors[row] = index.reconstruct(int(candidate_id))

    candidate_vectors = np.ascontiguousarray(candidate_vectors, dtype=np.float32)
    if candidate_vectors.ndim != 2 or query_embeddings.ndim != 2:
        raise ValueError("query_embeddings and candidate vectors must be 2-D")
    if query_embeddings.shape[1] != candidate_vectors.shape[1]:
        raise ValueError(
            "Embedding dimension mismatch: "
            f"queries={query_embeddings.shape[1]}, candidates={candidate_vectors.shape[1]}"
        )

    similarities = np.empty(
        (len(query_embeddings), len(candidate_ids)), dtype=np.float32
    )
    known = np.zeros(similarities.shape, dtype=np.bool_)

    if known_scores is not None or known_indices is not None:
        if known_scores is None or known_indices is None:
            raise ValueError("known_scores and known_indices must be provided together")
        known_scores = np.asarray(known_scores, dtype=np.float32)
        known_indices = np.asarray(known_indices, dtype=np.int64)
        if known_scores.shape != known_indices.shape:
            raise ValueError("known_scores and known_indices shape mismatch")
        if known_indices.ndim != 2 or known_indices.shape[0] != len(query_embeddings):
            raise ValueError("FAISS result rows must align 1:1 with query embeddings")

        valid = known_indices >= 0
        query_rows = np.broadcast_to(
            np.arange(len(query_embeddings), dtype=np.int64)[:, None],
            known_indices.shape,
        )[valid]
        candidate_columns = np.searchsorted(candidate_ids, known_indices[valid])
        in_union = candidate_columns < len(candidate_ids)
        in_union[in_union] &= (
            candidate_ids[candidate_columns[in_union]] == known_indices[valid][in_union]
        )
        query_rows = query_rows[in_union]
        candidate_columns = candidate_columns[in_union]
        similarities[query_rows, candidate_columns] = known_scores[valid][in_union]
        known[query_rows, candidate_columns] = True

    if not np.all(known):
        _fill_missing_similarities_numba(
            query_embeddings, candidate_vectors, similarities, known
        )
    return similarities


def multi_query_search(
    index,
    search_lock: RLock,
    metadata_records: list[dict],
    queries: list[str],
    embeddings: np.ndarray,
    top_k: int = 10,
    candidate_k: int | None = None,
    index_vectors: np.ndarray | np.memmap | None = None,
    beta: float = DEFAULT_MULTI_QUERY_BETA,
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
    valid_candidate_ids = np.unique(indices[indices >= 0]).astype(np.int64, copy=False)
    candidate_similarities = similarities_for_candidates(
        index,
        search_lock,
        embeddings,
        valid_candidate_ids,
        index_vectors,
        known_scores=scores,
        known_indices=indices,
    )
    return aggregate_multi_query(
        scores,
        indices,
        queries,
        int(top_k),
        metadata_records,
        candidate_similarities=candidate_similarities,
        beta=beta,
    )


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

    def score_candidates(
        self,
        queries: list[str],
        candidates: pd.DataFrame,
        model_key: str | None = None,
        beta: float = DEFAULT_MULTI_QUERY_BETA,
    ) -> pd.DataFrame:
        """Score an externally supplied frame pool with one semantic model.

        This is used for cross-model fusion: candidates are generated by the
        union of all models, then every model evaluates every candidate in its
        own embedding space.  Frame identities are resolved through metadata,
        so FAISS row ordering need not match between model indexes.
        """
        index = self._resolve(model_key)
        queries = clean_queries(queries)
        if not queries or candidates is None or candidates.empty:
            return pd.DataFrame()

        frame_col = "keyframe_id_int" if "keyframe_id_int" in candidates.columns else "keyframe_id"
        if "video_id" not in candidates.columns or frame_col not in candidates.columns:
            raise ValueError("semantic candidates require video_id and keyframe_id")

        rows: list[dict] = []
        candidate_ids: list[int] = []
        seen: set[tuple[str, int]] = set()
        for record in candidates.to_dict(orient="records"):
            try:
                frame_id = int(float(record.get(frame_col)))
            except (TypeError, ValueError):
                continue
            identity = (str(record.get("video_id")), frame_id)
            if identity in seen:
                continue
            row_id = index._row_by_video_frame.get(identity)
            if row_id is None:
                continue
            seen.add(identity)
            candidate_ids.append(int(row_id))
            rows.append(index.metadata_records[int(row_id)].copy())

        if not candidate_ids:
            return pd.DataFrame()
        query_embeddings = index.encode_texts(queries)
        similarities = similarities_for_candidates(
            index.index,
            index.search_lock,
            query_embeddings,
            np.asarray(candidate_ids, dtype=np.int64),
            index.index_vectors,
        )
        # Use exactly the same exponential-mean policy as regular semantic
        # search before applying cross-model weights in the orchestrator.
        fused_scores = semantic_alignment_scores(similarities, beta)
        result = pd.DataFrame.from_records(rows)
        result["retrieval_score"] = fused_scores.astype(float)
        result["alignment_score"] = result["retrieval_score"]
        result["model_key"] = index.model_key
        result["search_mode"] = "semantic"
        return result

    def multi_query_search(
        self,
        queries: list[str],
        top_k: int = 10,
        candidate_k: int | None = None,
        query_embeddings: np.ndarray | None = None,
        model_key: str | None = None,
        beta: float = DEFAULT_MULTI_QUERY_BETA,
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
            index.index_vectors,
            beta,
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
        beta: float = DEFAULT_MULTI_QUERY_BETA,
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
            beta: Hệ số điều chỉnh hàm mũ trong điểm multiple-query; mặc định
                là ``0.8``.
        """
        index = self._resolve(model_key)
        queries = clean_queries([query for event in events for query in event])
        embeddings = index.encode_texts(queries)
        return self.multi_query_search(
            queries, top_k, candidate_k, embeddings, model_key=model_key, beta=beta
        )
