from __future__ import annotations

from src.retrieval.index.faiss_index import FaissIndex
from src.retrieval.index.index_manager import IndexManager
from src.retrieval.retriever.temporal_search.pipeline.search import SearchPipeline


def build_temporal_search_pipeline(index: FaissIndex | IndexManager) -> SearchPipeline:
    """Build a ready-to-use temporal `SearchPipeline` (single entry point,
    cùng hình dạng với `build_ocr_search_pipeline`/`build_asr_search_pipeline`).

    `index` có thể là 1 `FaissIndex` đơn hoặc 1 `IndexManager` (nhiều model,
    `.search(..., model_key=...)` chọn model nào — xem `SearchPipeline`).
    """
    return SearchPipeline(index)
