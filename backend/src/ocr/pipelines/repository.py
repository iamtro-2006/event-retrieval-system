from __future__ import annotations

from typing import Any

from src.ocr.schemas.document import OCRDocument
from src.ocr.services.elastic_search import ElasticsearchService


class OCRRepository:
    """Data-access layer for OCR documents, backed by Elasticsearch."""

    def __init__(self, elasticsearch_service: ElasticsearchService) -> None:
        self.es = elasticsearch_service

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    @staticmethod
    def mapping() -> dict[str, Any]:
        """Elasticsearch index mapping for OCR documents."""
        return {
            "properties": {
                "dataset": {"type": "keyword"},
                "video_id": {"type": "keyword"},
                "keyframe_id": {"type": "keyword"},
                "texts": {"type": "text"},
            }
        }

    def create_index(self) -> None:
        """Create the OCR index if it does not already exist."""
        self.es.create_index(mapping=self.mapping())

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def insert(self, document: OCRDocument) -> None:
        """Insert a single OCR document."""
        self.es.insert(document.model_dump())

    def bulk_insert(self, documents: list[OCRDocument]) -> None:
        """Bulk-insert a batch of OCR documents."""
        self.es.bulk_insert([doc.model_dump() for doc in documents])

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, text: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Run a full-text search against the `texts` field.

        Returns:
            Raw Elasticsearch hits (list of `{"_score": ..., "_source": {...}}`).
        """
        return self.es.search(query=text, size=top_k)

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return the total number of indexed OCR documents."""
        return self.es.count()
