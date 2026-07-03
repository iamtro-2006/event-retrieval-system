from __future__ import annotations

from src.retrieval.index.clip_faiss_index import ClipFaissIndex
from src.retrieval.retriever.temporal_search.pipeline.search import SearchPipeline


def build_temporal_search_pipeline(index: ClipFaissIndex) -> SearchPipeline:
    """Build a ready-to-use temporal `SearchPipeline` (single entry point,
    cùng hình dạng với `build_ocr_search_pipeline`/`build_asr_search_pipeline`).
    """
    return SearchPipeline(index)
