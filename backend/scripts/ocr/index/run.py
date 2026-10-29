from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

from src.retrieval.indexer.elasticsearch.client import ElasticsearchService
from src.retrieval.indexer.elasticsearch.ocr.indexing_pipeline import IndexPipeline
from src.retrieval.indexer.elasticsearch.ocr.repository import OCRRepository

BACKEND_DIR = Path(__file__).resolve().parents[3]


def resolve_backend_path(path_value: str | Path) -> Path:
    path = Path(str(path_value).replace("\\", "/"))
    return path if path.is_absolute() else BACKEND_DIR / path


def load_config(dataset: str) -> dict:
    config_path = BACKEND_DIR / "configs" / "ocr_extraction.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    with Path(os.environ.get("CONFIG_PATH", BACKEND_DIR / "configs" / "app.yaml")).open("r", encoding="utf-8") as f:
        app_cfg = yaml.safe_load(f) or {}
    available = app_cfg.get("datasets", {}).get("available", {})
    if dataset not in available:
        raise ValueError(f"Unknown dataset '{dataset}'. Available: {sorted(available)}")
    spec = available[dataset]
    cfg["elasticsearch"]["index"] = spec["ocr_index"]
    cfg.setdefault("dataset", {})["root"] = str(Path(spec["root"]) / "ocr" / dataset.upper())
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an isolated OCR index for V3C or LHE")
    parser.add_argument("--dataset", choices=["v3c", "lhe"], required=True)
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete the index first. Without this, re-running appends a second copy of every document.",
    )
    args = parser.parse_args()
    cfg = load_config(args.dataset)
    es_cfg = cfg["elasticsearch"]
    es = ElasticsearchService(
        host=es_cfg["host"],
        port=int(es_cfg["port"]),
        scheme=es_cfg.get("scheme", "http"),
        index_name=es_cfg["index"],
    )

    print(f"Connected: {es.ping()}")
    if args.recreate:
        es.delete_index()
        print(f"Dropped index {es.index_name}")
    repository = OCRRepository(es)
    pipeline = IndexPipeline(repository)
    pipeline.create_index()
    pipeline.index_folder(resolve_backend_path(cfg["dataset"]["root"]))
    print(f"Total documents: {repository.count()}")


if __name__ == "__main__":
    main()
