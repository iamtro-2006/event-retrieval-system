from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.asr.pipelines.indexing import IndexPipeline
from src.asr.pipelines.repository import ASRRepository
from src.asr.pipelines.search import SearchPipeline
from src.asr.services.elastic_search import ElasticsearchService


def load_asr_config(config_path: str | Path = "configs/asr.yaml") -> dict[str, Any]:
    """Load the ASR configuration YAML file.

    Args:
        config_path: Path to the ASR config file.

    Returns:
        The parsed configuration as a dictionary.
    """
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_elasticsearch_service(cfg: dict[str, Any]) -> ElasticsearchService:
    """Build an `ElasticsearchService` from a loaded ASR config dict.

    Args:
        cfg: Parsed ASR configuration (must contain an "elasticsearch" section).

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


def build_asr_repository(cfg: dict[str, Any]) -> ASRRepository:
    """Build an `ASRRepository` wired to Elasticsearch from config."""
    return ASRRepository(build_elasticsearch_service(cfg))


def build_asr_search_pipeline(cfg: dict[str, Any] | None = None, config_path: str | Path = "configs/asr.yaml") -> SearchPipeline:
    """Build a ready-to-use ASR `SearchPipeline`.

    This is the single entry point the rest of the application (e.g. the API layer
    and `FaissRetrievalSystem`) should use to obtain ASR search capabilities.

    Args:
        cfg: Optional pre-loaded config dict. If omitted, loads from `config_path`.
        config_path: Path to the ASR config YAML, used only if `cfg` is None.

    Returns:
        A `SearchPipeline` instance backed by Elasticsearch.
    """
    cfg = cfg or load_asr_config(config_path)
    repository = build_asr_repository(cfg)
    return SearchPipeline(repository)


def build_asr_index_pipeline(cfg: dict[str, Any] | None = None, config_path: str | Path = "configs/asr.yaml") -> IndexPipeline:
    """Build an ASR `IndexPipeline` for offline ingestion jobs (mirrors `build_asr_search_pipeline`)."""
    cfg = cfg or load_asr_config(config_path)
    repository = build_asr_repository(cfg)
    return IndexPipeline(repository)
