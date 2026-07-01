from __future__ import annotations

from pathlib import Path

import yaml

from src.ocr.pipelines.indexing import IndexPipeline
from src.ocr.pipelines.repository import OCRRepository
from src.ocr.services.elastic_search import ElasticsearchService


def load_config():

    config_path = Path("configs/ocr.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():

    cfg = load_config()

    es = ElasticsearchService(
        host=cfg["elasticsearch"]["host"],
        port=cfg["elasticsearch"]["port"],
        scheme=cfg["elasticsearch"]["scheme"],
        index_name=cfg["elasticsearch"]["index"],
    )

    print(f"Connected: {es.ping()}")

    repository = OCRRepository(es)

    pipeline = IndexPipeline(repository)

    pipeline.create_index()

    pipeline.index_folder(
        cfg["dataset"]["root"]
    )

    print(f"Total documents: {repository.count()}")


if __name__ == "__main__":
    main()