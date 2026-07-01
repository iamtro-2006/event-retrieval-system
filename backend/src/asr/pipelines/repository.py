from __future__ import annotations

from typing import Any

from src.asr.schemas.document import ASRDocument
from src.asr.services.elastic_search import ElasticsearchService


class ASRRepository:
    """Data-access layer for ASR (speech transcript) documents, backed by Elasticsearch."""

    def __init__(self, elasticsearch_service: ElasticsearchService) -> None:
        self.es = elasticsearch_service

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    @staticmethod
    def mapping() -> dict[str, Any]:
        """Elasticsearch index mapping for ASR documents."""
        return {
            "properties": {
                "dataset": {"type": "keyword"},
                "video_id": {"type": "keyword"},
                "segment_id": {"type": "keyword"},
                "start_time": {"type": "float"},
                "end_time": {"type": "float"},
                "text": {"type": "text"},
            }
        }

    def create_index(self) -> None:
        """Create the ASR index if it does not already exist."""
        self.es.create_index(mapping=self.mapping())

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def insert(self, document: ASRDocument) -> None:
        """Insert a single ASR document."""
        self.es.insert(document.model_dump())

    def bulk_insert(self, documents: list[ASRDocument]) -> None:
        """Bulk-insert a batch of ASR documents."""
        self.es.bulk_insert([doc.model_dump() for doc in documents])

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, text: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Run a full-text search against the `text` field.

        Returns:
            Raw Elasticsearch hits (list of `{"_score": ..., "_source": {...}}`).
        """
        return self.es.search(query=text, size=top_k)

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return the total number of indexed ASR documents."""
        return self.es.count()
