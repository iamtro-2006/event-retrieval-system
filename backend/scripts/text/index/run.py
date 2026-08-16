from __future__ import annotations

from pathlib import Path

import yaml

from src.retrieval.indexer.elasticsearch.client import ElasticsearchService
from src.retrieval.indexer.elasticsearch.multimodal_text.indexing_pipeline import (
    IndexPipeline,
)
from src.retrieval.indexer.elasticsearch.multimodal_text.repository import (
    MultimodalTextRepository,
)


def load_config() -> dict:
    config_path = Path("configs/text.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    cfg = load_config()

    es = ElasticsearchService(
        host=cfg["elasticsearch"]["host"],
        port=cfg["elasticsearch"]["port"],
        scheme=cfg["elasticsearch"]["scheme"],
        index_name=cfg["elasticsearch"]["index"],
    )

    print(f"Connected: {es.ping()}")

    repository = MultimodalTextRepository(es)

    pipeline = IndexPipeline(
        repository=repository,
        metadata_path=cfg["dataset"]["metadata_path"],
        ocr_root=cfg["dataset"].get("ocr_root"),
        asr_root=cfg["dataset"].get("asr_root"),
    )

    pipeline.create_index()

    total = pipeline.run()

    print(f"Total documents: {repository.count()}")
    print(f"Indexed {total} keyframes")


if __name__ == "__main__":
    main()
