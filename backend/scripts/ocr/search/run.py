from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.retrieval.indexer.elasticsearch.client import ElasticsearchService
from src.retrieval.indexer.elasticsearch.ocr.repository import OCRRepository
from src.retrieval.retriever.ocr_search.pipeline.search import SearchPipeline

BACKEND_DIR = Path(__file__).resolve().parents[3]


def load_config(dataset: str | None = None) -> dict:
    """Load OCR config, optionally pinned to a dataset's real index.

    Without `dataset` this reads the standalone index name from
    `configs/ocr_extraction.yaml` (e.g. `ocr_db`), which is NOT what the API
    searches. Passing a dataset applies the same override the API and the
    indexing script use, so a manual check hits the same index as the backend.
    """
    config_path = BACKEND_DIR / "configs" / "ocr_extraction.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if dataset:
        with (BACKEND_DIR / "configs" / "app.yaml").open("r", encoding="utf-8") as f:
            app_cfg = yaml.safe_load(f) or {}
        available = app_cfg.get("datasets", {}).get("available", {})
        if dataset not in available:
            raise ValueError(f"Unknown dataset '{dataset}'. Available: {sorted(available)}")
        cfg["elasticsearch"]["index"] = available[dataset]["ocr_index"]
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive OCR search over Elasticsearch")
    parser.add_argument("--dataset", choices=["v3c", "lhe"], default=None)
    args = parser.parse_args()

    cfg = load_config(args.dataset)
    es_cfg = cfg["elasticsearch"]
    es = ElasticsearchService(
        host=es_cfg["host"],
        port=int(es_cfg["port"]),
        scheme=es_cfg.get("scheme", "http"),
        index_name=es_cfg["index"],
        search_field="texts",
    )
    print(f"ES: {es.url} | index: {es.index_name} | {es.health()}")
    pipeline = SearchPipeline(OCRRepository(es))

    while True:
        query = input("\nQuery (type 'exit' to quit): ").strip()
        if query.lower() == "exit":
            break

        results = pipeline.search(query=query, top_k=10)
        if not results:
            print("No result.")
            continue

        for i, result in enumerate(results, start=1):
            print("=" * 60)
            print(f"Rank         : {i}")
            print(f"Score        : {result['score']:.4f}")
            print(f"Dataset      : {result['dataset']}")
            print(f"Video ID     : {result['video_id']}")
            print(f"Keyframe ID  : {result['keyframe_id']}")
            print("Texts:")
            for text in result["texts"]:
                print(f"  - {text}")
        print("=" * 60)


if __name__ == "__main__":
    main()
