from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.retrieval.indexer.elasticsearch.client import ElasticsearchService
from src.retrieval.indexer.elasticsearch.ocr.indexing_pipeline import IndexPipeline
from src.retrieval.indexer.elasticsearch.ocr.repository import OCRRepository


def load_ocr_config(config_path: str | Path = "configs/ocr.yaml") -> dict[str, Any]:
    """Load the OCR configuration YAML file."""
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_elasticsearch_service(cfg: dict[str, Any]) -> ElasticsearchService:
    """Build an ElasticsearchService from a loaded OCR config dict."""
    es_cfg = cfg["elasticsearch"]
    return ElasticsearchService(
        host=es_cfg["host"],
        port=int(es_cfg["port"]),
        scheme=es_cfg.get("scheme", "http"),
        index_name=es_cfg["index"],
        search_field="texts",
    )


def build_ocr_repository(cfg: dict[str, Any]) -> OCRRepository:
    """Build a OCRRepository wired to Elasticsearch from config."""
    return OCRRepository(build_elasticsearch_service(cfg))


def build_ocr_index_pipeline(cfg: dict[str, Any] | None = None, config_path: str | Path = "configs/ocr.yaml") -> IndexPipeline:
    """Build a OCR IndexPipeline for offline ingestion jobs."""
    cfg = cfg or load_ocr_config(config_path)
    repository = build_ocr_repository(cfg)
    return IndexPipeline(repository)
