from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.ocr.pipelines.indexing import IndexPipeline
from src.ocr.pipelines.repository import OCRRepository
from src.ocr.pipelines.search import SearchPipeline
from src.ocr.services.elastic_search import ElasticsearchService


def load_ocr_config(config_path: str | Path = "configs/ocr.yaml") -> dict[str, Any]:
    """Load the OCR configuration YAML file.

    Args:
        config_path: Path to the OCR config file.

    Returns:
        The parsed configuration as a dictionary.
    """
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_elasticsearch_service(cfg: dict[str, Any]) -> ElasticsearchService:
    """Build an `ElasticsearchService` from a loaded OCR config dict.

    Args:
        cfg: Parsed OCR configuration (must contain an "elasticsearch" section).

    Returns:
        A configured `ElasticsearchService` instance.
    """
    es_cfg = cfg["elasticsearch"]
    return ElasticsearchService(
        host=es_cfg["host"],
        port=int(es_cfg["port"]),
        scheme=es_cfg.get("scheme", "http"),
        index_name=es_cfg["index"],
    )


def build_ocr_repository(cfg: dict[str, Any]) -> OCRRepository:
    """Build an `OCRRepository` wired to Elasticsearch from config."""
    return OCRRepository(build_elasticsearch_service(cfg))


def build_ocr_search_pipeline(cfg: dict[str, Any] | None = None, config_path: str | Path = "configs/ocr.yaml") -> SearchPipeline:
    """Build a ready-to-use OCR `SearchPipeline`.

    This is the single entry point the rest of the application (e.g. the API layer
    and `FaissRetrievalSystem`) should use to obtain OCR search capabilities.

    Args:
        cfg: Optional pre-loaded config dict. If omitted, loads from `config_path`.
        config_path: Path to the OCR config YAML, used only if `cfg` is None.

    Returns:
        A `SearchPipeline` instance backed by Elasticsearch.
    """
    cfg = cfg or load_ocr_config(config_path)
    repository = build_ocr_repository(cfg)
    return SearchPipeline(repository)


def build_ocr_index_pipeline(cfg: dict[str, Any] | None = None, config_path: str | Path = "configs/ocr.yaml") -> IndexPipeline:
    """Build an OCR `IndexPipeline` for offline ingestion jobs (mirrors `build_ocr_search_pipeline`)."""
    cfg = cfg or load_ocr_config(config_path)
    repository = build_ocr_repository(cfg)
    return IndexPipeline(repository)
