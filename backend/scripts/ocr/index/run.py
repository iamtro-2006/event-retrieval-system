from __future__ import annotations

import argparse
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
    with (BACKEND_DIR / "configs" / "app.yaml").open("r", encoding="utf-8") as f:
        app_cfg = yaml.safe_load(f) or {}
    available = app_cfg.get("datasets", {}).get("available", {})
    if dataset not in available:
        raise ValueError(f"Unknown dataset '{dataset}'. Available: {sorted(available)}")
    spec = available[dataset]
    cfg["elasticsearch"]["index"] = spec["ocr_index"]
    cfg.setdefault("dataset", {})["root"] = str(Path(spec["root"]) / "ocr")
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an isolated OCR index for AIC or CAM")
    parser.add_argument("--dataset", choices=["aic", "cam"], required=True)
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
    repository = OCRRepository(es)
    pipeline = IndexPipeline(repository)
    pipeline.create_index()
    pipeline.index_folder(resolve_backend_path(cfg["dataset"]["root"]))
    print(f"Total documents: {repository.count()}")


if __name__ == "__main__":
    main()
