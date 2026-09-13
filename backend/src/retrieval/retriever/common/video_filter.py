"""Request-local video allowlist, propagated into worker threads by AnyIO.

Never mutate shared indexes: concurrent searches may use different filters.
"""
from contextvars import ContextVar
import numpy as np

video_ids_context: ContextVar[frozenset[str]] = ContextVar("video_ids", default=frozenset())


def with_video_filter(video_ids, function, *args, **kwargs):
    token = video_ids_context.set(frozenset(str(value).strip() for value in (video_ids or []) if str(value).strip()))
    try:
        return function(*args, **kwargs)
    finally:
        video_ids_context.reset(token)


def filtered_faiss_search(index, search_lock, embeddings, k, metadata_records):
    allowed = video_ids_context.get()
    if allowed:
        if metadata_records is None:
            raise ValueError("Video filtering requires metadata")
        allowed_rows = np.fromiter((str(row.get("video_id")) in allowed for row in metadata_records), dtype=bool)
        if len(allowed_rows) and allowed_rows.all():
            with search_lock:
                return index.search(embeddings, min(k, index.ntotal))
        target = min(k, int(allowed_rows.sum()))
        if not target:
            return np.empty((len(embeddings), 0), np.float32), np.empty((len(embeddings), 0), np.int64)
        # Expand each query independently until its filtered candidate pool is
        # full (or the index is exhausted), before fusion/temporal ranking.
        output_scores = np.full((len(embeddings), target), -np.inf, np.float32)
        output_ids = np.full((len(embeddings), target), -1, np.int64)
        for query_index, embedding in enumerate(embeddings):
            fetch_k = min(max(k, 64), index.ntotal)
            while True:
                with search_lock:
                    scores, rows = index.search(embedding.reshape(1, -1), fetch_k)
                valid = rows[0] >= 0
                positions = np.flatnonzero(valid)
                positions = positions[allowed_rows[rows[0, positions]]][:target]
                if len(positions) >= target or fetch_k >= index.ntotal:
                    output_scores[query_index, :len(positions)] = scores[0, positions]
                    output_ids[query_index, :len(positions)] = rows[0, positions]
                    break
                fetch_k = min(fetch_k * 2, index.ntotal)
        return output_scores, output_ids
    raise ValueError("Filtered search requires a nonempty allowlist")
