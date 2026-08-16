from __future__ import annotations

from typing import Any, TypedDict

from src.retrieval.indexer.elasticsearch.multimodal_text.repository import (
    MultimodalTextRepository,
)


class TextHit(TypedDict):
    """A single normalized unified-text search hit (one keyframe), before being
    enriched with FAISS metadata by the orchestrator.

    Carries BOTH the matched OCR and ASR fragments so downstream fusion with
    Milvus (semantic search) can reason about which modality contributed.
    """

    es_score: float
    dataset: str
    video_id: str
    keyframe_id: str
    timestamp_sec: float
    matched_ocr: str
    matched_asr: str


class SearchPipeline:
    """Unified full-text search over OCR + ASR (Elasticsearch-backed).

    Same shape as `ocr_search`/`asr_search` `SearchPipeline`: constructed with
    a repository, exposes a single `.search(query, top_k)` entrypoint, and
    returns normalized hits (no raw Elasticsearch JSON leaks out).
    """

    def __init__(self, repository: MultimodalTextRepository) -> None:
        self.repository = repository

    def search(self, query: str, top_k: int = 10) -> list[TextHit]:
        """Run a boosted, fuzzy `multi_match` search and return normalized hits.

        The query is a `multi_match` of type ``best_fields`` across
        ``ocr_text^3`` and ``asr_text^1`` with ``fuzziness: AUTO`` (see
        `MultimodalTextRepository.SEARCH_FIELDS`). OCR is weighted 3x over ASR
        because on-screen text is a stronger, more precise signal than speech
        transcripts; tune the boost weights there to rebalance.

        Args:
            query: Free-text search query (matched against OCR + ASR text).
            top_k: Maximum number of hits to return.

        Returns:
            A list of normalized text hits, ordered by descending relevance
            score. Each hit carries the matched OCR/ASR fragments (from ES
            `highlight`, falling back to the full source field).
        """
        query = str(query or "").strip()
        if not query:
            return []

        raw_hits: list[dict[str, Any]] = self.repository.search(text=query, top_k=top_k)

        hits: list[TextHit] = []
        for hit in raw_hits:
            source = hit.get("_source", {})
            highlight = hit.get("highlight") or {}

            # ES highlight returns a list of matched fragments per field; join
            # them into one string. Fall back to the full source text when no
            # fragment was highlighted (e.g. fuzzy match with no exact span).
            ocr_fragments = highlight.get("ocr_text")
            matched_ocr = (
                " ... ".join(ocr_fragments)
                if ocr_fragments
                else str(source.get("ocr_text", ""))
            )

            asr_fragments = highlight.get("asr_text")
            matched_asr = (
                " ... ".join(asr_fragments)
                if asr_fragments
                else str(source.get("asr_text", ""))
            )

            hits.append(
                TextHit(
                    es_score=float(hit.get("_score") or 0.0),
                    dataset=source.get("dataset", ""),
                    video_id=source.get("video_id", ""),
                    keyframe_id=source.get("keyframe_id", ""),
                    timestamp_sec=float(source.get("timestamp_sec", 0.0)),
                    matched_ocr=matched_ocr,
                    matched_asr=matched_asr,
                )
            )
        return hits
