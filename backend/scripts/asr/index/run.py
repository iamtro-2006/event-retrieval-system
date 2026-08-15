from __future__ import annotations

from pathlib import Path

import yaml

from src.retrieval.indexer.elasticsearch.asr.indexing_pipeline import IndexPipeline
from src.retrieval.indexer.elasticsearch.asr.repository import ASRRepository
from src.retrieval.indexer.elasticsearch.client import ElasticsearchService

BACKEND_DIR = Path(__file__).resolve().parents[3]


def resolve_backend_path(path_value: str | Path) -> Path:
    path = Path(str(path_value).replace("\\", "/"))
    return path if path.is_absolute() else BACKEND_DIR / path


def load_config() -> dict:
    config_path = BACKEND_DIR / "configs" / "asr_extraction.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    cfg = load_config()
    es_cfg = cfg["elasticsearch"]
    es = ElasticsearchService(
        host=es_cfg["host"],
        port=int(es_cfg["port"]),
        scheme=es_cfg.get("scheme", "http"),
        index_name=es_cfg["index"],
    )

    print(f"Connected: {es.ping()}")
    repository = ASRRepository(es)
    pipeline = IndexPipeline(repository)
    pipeline.create_index()
    pipeline.index_folder(resolve_backend_path(cfg["dataset"]["root"]))
    print(f"Total documents: {repository.count()}")


if __name__ == "__main__":
    main()
