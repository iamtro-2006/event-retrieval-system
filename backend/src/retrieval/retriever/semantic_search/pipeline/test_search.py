from __future__ import annotations

from threading import RLock

import numpy as np
import pytest

from src.retrieval.retriever.semantic_search.pipeline.search import (
    DEFAULT_MULTI_QUERY_BETA,
    _fill_missing_similarities_numba,
    aggregate_multi_query,
    similarities_for_candidates,
)


def test_aggregate_multi_query_uses_beta_scaled_exp_mean_over_every_query() -> None:
    # Candidate 0 is retrieved only by the original query, candidate 2 only
    # by the sub-query, while candidate 1 appears in both candidate lists.
    scores = np.array([[0.9, 0.7], [0.8, 0.6]], dtype=np.float32)
    indices = np.array([[0, 1], [1, 2]], dtype=np.int64)
    dense_similarities = np.array(
        [[0.9, 0.7, 0.1], [0.2, 0.8, 0.6]], dtype=np.float32
    )

    result = aggregate_multi_query(
        scores=scores,
        indices=indices,
        queries=["original query", "sub-query"],
        top_k=3,
        metadata_records=[{"candidate_id": i} for i in range(3)],
        candidate_similarities=dense_similarities,
    )

    expected = np.mean(
        np.exp(DEFAULT_MULTI_QUERY_BETA * (dense_similarities - 1.0)), axis=0
    )
    result_by_id = result.set_index("candidate_id")
    np.testing.assert_allclose(
        result_by_id.loc[[0, 1, 2], "retrieval_score"].to_numpy(),
        expected,
    )
    assert result.iloc[0]["candidate_id"] == 1
    assert result_by_id.loc[0, "matched_queries"] == 1


def test_aggregate_multi_query_accepts_custom_beta() -> None:
    scores = np.array([[0.9, 0.7], [0.8, 0.6]], dtype=np.float32)
    indices = np.array([[0, 1], [1, 2]], dtype=np.int64)
    dense_similarities = np.array(
        [[0.9, 0.7, 0.1], [0.2, 0.8, 0.6]], dtype=np.float32
    )
    beta = 1.25

    result = aggregate_multi_query(
        scores=scores,
        indices=indices,
        queries=["original query", "sub-query"],
        top_k=3,
        metadata_records=[{"candidate_id": i} for i in range(3)],
        candidate_similarities=dense_similarities,
        beta=beta,
    )

    expected = np.mean(np.exp(beta * (dense_similarities - 1.0)), axis=0)
    result_by_id = result.set_index("candidate_id")
    np.testing.assert_allclose(
        result_by_id.loc[[0, 1, 2], "retrieval_score"].to_numpy(),
        expected,
    )


def test_aggregate_multi_query_rejects_sparse_scores() -> None:
    with pytest.raises(ValueError, match="candidate_similarities is required"):
        aggregate_multi_query(
            scores=np.array([[0.9], [0.8]], dtype=np.float32),
            indices=np.array([[0], [0]], dtype=np.int64),
            queries=["query one", "query two"],
            top_k=1,
            metadata_records=[{"candidate_id": 0}],
        )


def test_candidate_similarities_reuse_faiss_scores_and_numba_fills_missing() -> None:
    query_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    index_vectors = np.array(
        [[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]], dtype=np.float32
    )
    candidate_ids = np.array([0, 1, 2], dtype=np.int64)
    indices = np.array([[0, 1], [1, 2]], dtype=np.int64)
    # Deliberately differ slightly from the exact dot products so the test
    # proves that existing FAISS scores are retained rather than recomputed.
    scores = np.array([[0.99, 0.59], [0.79, 0.98]], dtype=np.float32)

    result = similarities_for_candidates(
        index=None,
        search_lock=RLock(),
        query_embeddings=query_embeddings,
        candidate_ids=candidate_ids,
        index_vectors=index_vectors,
        known_scores=scores,
        known_indices=indices,
    )

    expected = np.array([[0.99, 0.59, 0.0], [0.0, 0.79, 0.98]], dtype=np.float32)
    np.testing.assert_allclose(result, expected)
    assert _fill_missing_similarities_numba.nopython_signatures
