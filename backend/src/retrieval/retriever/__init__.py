# retriever/ — 4 backend con (semantic_search/temporal_search/ocr_search/asr_search)
# + common/ (Orchestrator dieu phoi). Moi backend co cung 1 hinh dang:
#   factory.py:            build_<mode>_search_pipeline(...) -> SearchPipeline
#   pipeline/search.py:    cac ham thuat toan thuan + class `SearchPipeline`
# Khong co class nghiep vu rieng (SemanticSearch/TemporalSearch) nam ngoai
# pipeline/search.py nua — xem `src.retrieval.system.build_system`.
#
# Package nay CHUA export gi o day; import thang tu submodule can dung, vd:
#   from src.retrieval.retriever.common.orchestrator import Orchestrator, QueryPlan
#   from src.retrieval.retriever.semantic_search.factory import build_semantic_search_pipeline
