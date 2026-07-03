from __future__ import annotations

from pathlib import Path
from typing import Any

from src.retrieval.indexer.elasticsearch.asr.factory import (
    build_asr_repository,
    load_asr_config,
)
from src.retrieval.retriever.asr_search.pipeline.search import SearchPipeline


def build_asr_search_pipeline(cfg: dict[str, Any] | None = None, config_path: str | Path = "configs/asr.yaml") -> SearchPipeline:
    """Build a ready-to-use ASR SearchPipeline (single entry point for API layer)."""
    cfg = cfg or load_asr_config(config_path)
    repository = build_asr_repository(cfg)
    return SearchPipeline(repository)
