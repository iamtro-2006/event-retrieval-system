from __future__ import annotations

from pathlib import Path

import yaml

from src.asr.pipelines.search import SearchPipeline
from src.asr.pipelines.repository import ASRRepository
from src.asr.services.elastic_search import ElasticsearchService


def load_config():

    config_path = Path("configs/asr.yaml")

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

    repository = ASRRepository(es)

    pipeline = SearchPipeline(repository)

    while True:

        query = input("\nQuery (exit để thoát): ").strip()

        if query.lower() == "exit":
            break

        results = pipeline.search(
            query=query,
            top_k=10,
        )

        print()

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
