"""Request-local video allowlist, propagated into worker threads by AnyIO.

Never mutate shared indexes: concurrent searches may use different filters.
"""
from contextvars import ContextVar
import numpy as np
import faiss

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
        allowed_ids = np.flatnonzero(allowed_rows).astype(np.int64, copy=False)
        target = min(k, len(allowed_ids))
        if target == 0:
            return np.empty((len(embeddings), 0), np.float32), np.empty((len(embeddings), 0), np.int64)
        # Search an exact FAISS flat index built from every allowed row. This
        # makes an in-video search rank all keyframes in that video, including
        # frames absent from the original global result list.
        with search_lock:
            try:
                vectors = np.stack([index.reconstruct(int(row_id)) for row_id in allowed_ids])
            except Exception as exc:
                raise RuntimeError(
                    "Video-filtered search requires reconstructable FAISS vectors"
                ) from exc
            subset_index = faiss.IndexFlat(index.d, index.metric_type)
            subset_index.add(np.ascontiguousarray(vectors, dtype=np.float32))
            scores, local_rows = subset_index.search(embeddings, target)
        valid = local_rows >= 0
        output_ids = np.full(local_rows.shape, -1, dtype=np.int64)
        output_ids[valid] = allowed_ids[local_rows[valid]]
        return scores, output_ids
    raise ValueError("Filtered search requires a nonempty allowlist")
