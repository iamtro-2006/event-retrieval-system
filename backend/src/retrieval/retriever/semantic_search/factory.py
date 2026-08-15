from __future__ import annotations

from src.retrieval.index.faiss_index import FaissIndex
from src.retrieval.index.index_manager import IndexManager
from src.retrieval.retriever.semantic_search.pipeline.search import SearchPipeline


def build_semantic_search_pipeline(index: FaissIndex | IndexManager) -> SearchPipeline:
    """Build a ready-to-use semantic `SearchPipeline` (single entry point,
    cùng hình dạng với `build_ocr_search_pipeline`/`build_asr_search_pipeline`).

    `index` có thể là 1 `FaissIndex` đơn (tương thích ngược, chỉ 1 model)
    hoặc 1 `IndexManager` (nhiều model, cho phép `.search(..., model_key=...)`
    chọn model nào để search — xem `SearchPipeline`).
    """
    return SearchPipeline(index)
