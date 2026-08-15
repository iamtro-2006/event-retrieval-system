from __future__ import annotations

from pathlib import Path

import yaml

from src.retrieval.indexer.elasticsearch.asr.repository import ASRRepository
from src.retrieval.indexer.elasticsearch.client import ElasticsearchService
from src.retrieval.retriever.asr_search.pipeline.search import SearchPipeline

BACKEND_DIR = Path(__file__).resolve().parents[3]


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
    pipeline = SearchPipeline(ASRRepository(es))

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
            print(f"Segment ID   : {result['segment_id']}")
            print(f"Start / End  : {result['start_time']:.2f}s - {result['end_time']:.2f}s")
            print(f"Text         : {result['text']}")
        print("=" * 60)


if __name__ == "__main__":
    main()
