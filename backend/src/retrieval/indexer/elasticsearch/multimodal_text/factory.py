from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.retrieval.indexer.elasticsearch.client import ElasticsearchService
from src.retrieval.indexer.elasticsearch.multimodal_text.repository import (
    MultimodalTextRepository,
)


def load_text_config(config_path: str | Path = "configs/text.yaml") -> dict[str, Any]:
    """Load the unified text-search configuration YAML file."""
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_elasticsearch_service(cfg: dict[str, Any]) -> ElasticsearchService:
    """Build an ElasticsearchService from a loaded text config dict.

    No `search_field` is set: the unified index uses `multi_match` across
    `ocr_text` + `asr_text` (see `MultimodalTextRepository.search`), not the
    single-field `search()` path used by the legacy OCR/ASR services.
    """
    es_cfg = cfg["elasticsearch"]
    return ElasticsearchService(
        host=es_cfg["host"],
        port=int(es_cfg["port"]),
        scheme=es_cfg.get("scheme", "http"),
        index_name=es_cfg["index"],
    )


def build_text_repository(cfg: dict[str, Any]) -> MultimodalTextRepository:
    """Build a MultimodalTextRepository wired to Elasticsearch from config."""
    return MultimodalTextRepository(build_elasticsearch_service(cfg))
