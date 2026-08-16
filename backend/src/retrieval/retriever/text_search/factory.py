from __future__ import annotations

from pathlib import Path
from typing import Any

from src.retrieval.indexer.elasticsearch.multimodal_text.factory import (
    build_text_repository,
    load_text_config,
)
from src.retrieval.retriever.text_search.pipeline.search import SearchPipeline


def build_text_search_pipeline(
    cfg: dict[str, Any] | None = None, config_path: str | Path = "configs/text.yaml"
) -> SearchPipeline:
    """Build a ready-to-use unified text SearchPipeline (single entry point for API layer)."""
    cfg = cfg or load_text_config(config_path)
    repository = build_text_repository(cfg)
    return SearchPipeline(repository)
