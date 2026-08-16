"""Standalone dev/test script for the unified `multimodal_text` Elasticsearch index.

Idempotently resets the index and ingests a few mock keyframe documents so the
schema can be validated against real `multi_match` queries. Not part of the
online read path — run it offline (from `backend/`) like the other
`scripts/` CLIs:

    python -m src.retrieval.indexer.elasticsearch.setup_and_ingest

Connects to `localhost:9200` (matches `docker-compose.yaml`). For a non-local
host, pass the host/port via the `--host`/`--port` flags or edit the defaults.
"""

from __future__ import annotations

from src.retrieval.indexer.elasticsearch.client import ElasticsearchService
from src.retrieval.indexer.elasticsearch.multimodal_text.repository import (
    MultimodalTextRepository,
)
from src.retrieval.indexer.elasticsearch.multimodal_text.schemas import (
    MultimodalTextDocument,
)

INDEX_NAME = "multimodal_text"

# Default dev connection (matches backend/docker-compose.yaml).
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 9200


def build_repository(
    host: str = DEFAULT_HOST, port: int = DEFAULT_PORT
) -> MultimodalTextRepository:
    """Build a MultimodalTextRepository wired to the given Elasticsearch host."""
    service = ElasticsearchService(
        host=host,
        port=port,
        index_name=INDEX_NAME,
    )
    return MultimodalTextRepository(service)


def delete_index(repository: MultimodalTextRepository) -> None:
    """Delete the `multimodal_text` index if it exists (idempotent)."""
    repository.delete_index()
    print(f"[setup] deleted index '{INDEX_NAME}' (if it existed)")


def create_index(repository: MultimodalTextRepository) -> None:
    """Create the `multimodal_text` index with the unified OCR+ASR mapping."""
    repository.create_index()
    print(f"[setup] created index '{INDEX_NAME}' with mapping:")
    for field, spec in MultimodalTextRepository.mapping()["properties"].items():
        print(f"         {field}: {spec}")


def bulk_ingest_mock_docs(repository: MultimodalTextRepository) -> None:
    """Bulk-ingest 3 mock keyframe documents to exercise the schema."""
    mock_docs: list[MultimodalTextDocument] = [
        MultimodalTextDocument(
            dataset="test",
            video_id="video_001",
            keyframe_id="1",
            timestamp_sec=12.5,
            ocr_text="BREAKING NEWS: City marathon results",
            asr_text="Welcome back, the marathon results are in",
        ),
        MultimodalTextDocument(
            dataset="test",
            video_id="video_001",
            keyframe_id="2",
            timestamp_sec=25.0,
            ocr_text="Weather forecast: heavy rain tonight",
            asr_text="Expect heavy rainfall throughout the evening",
        ),
        MultimodalTextDocument(
            dataset="test",
            video_id="video_002",
            keyframe_id="1",
            timestamp_sec=5.0,
            ocr_text="Goal! 2-1 in the final minutes",
            asr_text="An incredible goal in the dying minutes of the match",
        ),
    ]
    repository.bulk_insert(mock_docs)
    print(f"[setup] bulk-ingested {len(mock_docs)} mock documents")


def main(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    """Reset and re-ingest the `multimodal_text` index end-to-end."""
    repository = build_repository(host=host, port=port)

    delete_index(repository)
    create_index(repository)
    bulk_ingest_mock_docs(repository)

    count = repository.count()
    print(f"[setup] done — index '{INDEX_NAME}' now holds {count} documents")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Set up the multimodal_text ES index with mock data."
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Elasticsearch host")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="Elasticsearch port"
    )
    args = parser.parse_args()

    main(host=args.host, port=args.port)
