from __future__ import annotations

from typing import Any

from src.retrieval.indexer.elasticsearch.client import ElasticsearchService
from src.retrieval.indexer.elasticsearch.multimodal_text.schemas import (
    MultimodalTextDocument,
)


class MultimodalTextRepository:
    """Data-access layer for unified OCR+ASR documents, backed by Elasticsearch.

    Each document is keyframe-level and carries both ``ocr_text`` and
    ``asr_text``, so a single ``multi_match`` query can fuse the two signals
    (replacing the legacy split into separate ``ocr``/``asr`` indices).
    """

    # Per-field boost weights for the unified multi_match query.
    # OCR is boosted 3x over ASR: on-screen text is a stronger, more precise
    # signal than speech transcripts (which are noisier and time-snapped).
    # Tune these together with `fuzziness` below.
    SEARCH_FIELDS: list[str] = ["ocr_text^3", "asr_text^1"]

    # Fields to return matched fragments for (populates matched_ocr/matched_asr).
    HIGHLIGHT_FIELDS: list[str] = ["ocr_text", "asr_text"]

    def __init__(self, elasticsearch_service: ElasticsearchService) -> None:
        self.es = elasticsearch_service

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    @staticmethod
    def mapping() -> dict[str, Any]:
        """Elasticsearch index mapping for unified multimodal-text documents."""
        return {
            "properties": {
                "dataset": {"type": "keyword"},
                "video_id": {"type": "keyword"},
                "keyframe_id": {"type": "keyword"},
                "timestamp_sec": {"type": "float"},
                "ocr_text": {"type": "text", "analyzer": "standard"},
                "asr_text": {"type": "text", "analyzer": "standard"},
            }
        }

    def create_index(self) -> None:
        """Create the `multimodal_text` index if it does not already exist."""
        self.es.create_index(mapping=self.mapping())

    def delete_index(self) -> None:
        """Delete the `multimodal_text` index if it exists (idempotent)."""
        self.es.delete_index()

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def insert(self, document: MultimodalTextDocument) -> None:
        """Insert a single multimodal-text document."""
        self.es.insert(document.model_dump())

    def bulk_insert(self, documents: list[MultimodalTextDocument]) -> None:
        """Bulk-insert a batch of multimodal-text documents."""
        self.es.bulk_insert([doc.model_dump() for doc in documents])

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, text: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Run a boosted, fuzzy `multi_match` across OCR + ASR text.

        Returns:
            Raw Elasticsearch hits (list of ``{"_score": ..., "_source": {...},
            "highlight": {...}}``).
        """
        return self.es.multi_match_search(
            query=text,
            fields=self.SEARCH_FIELDS,
            size=top_k,
            fuzziness="AUTO",  # typo / OCR-noise tolerance
            match_type="best_fields",
            highlight_fields=self.HIGHLIGHT_FIELDS,
        )

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return the total number of indexed multimodal-text documents."""
        return self.es.count()
