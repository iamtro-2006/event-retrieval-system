from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.retrieval.indexer.elasticsearch.client import ElasticsearchService
from src.retrieval.indexer.elasticsearch.asr.indexing_pipeline import IndexPipeline
from src.retrieval.indexer.elasticsearch.asr.repository import ASRRepository


def load_asr_config(config_path: str | Path = "configs/asr.yaml") -> dict[str, Any]:
    """Load the ASR configuration YAML file."""
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_elasticsearch_service(cfg: dict[str, Any]) -> ElasticsearchService:
    """Build an ElasticsearchService from a loaded ASR config dict."""
    es_cfg = cfg["elasticsearch"]
    return ElasticsearchService(
        host=es_cfg["host"],
        port=int(es_cfg["port"]),
        scheme=es_cfg.get("scheme", "http"),
        index_name=es_cfg["index"],
        search_field="text",
    )


def build_asr_repository(cfg: dict[str, Any]) -> ASRRepository:
    """Build a ASRRepository wired to Elasticsearch from config."""
    return ASRRepository(build_elasticsearch_service(cfg))


def build_asr_index_pipeline(cfg: dict[str, Any] | None = None, config_path: str | Path = "configs/asr.yaml") -> IndexPipeline:
    """Build a ASR IndexPipeline for offline ingestion jobs."""
    cfg = cfg or load_asr_config(config_path)
    repository = build_asr_repository(cfg)
    return IndexPipeline(repository)
