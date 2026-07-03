from __future__ import annotations

from pathlib import Path
from typing import Any

from src.retrieval.indexer.elasticsearch.ocr.factory import (
    build_ocr_repository,
    load_ocr_config,
)
from src.retrieval.retriever.ocr_search.pipeline.search import SearchPipeline


def build_ocr_search_pipeline(cfg: dict[str, Any] | None = None, config_path: str | Path = "configs/ocr.yaml") -> SearchPipeline:
    """Build a ready-to-use OCR SearchPipeline (single entry point for API layer)."""
    cfg = cfg or load_ocr_config(config_path)
    repository = build_ocr_repository(cfg)
    return SearchPipeline(repository)
