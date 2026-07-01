from __future__ import annotations

from typing import Any, TypedDict

from src.ocr.pipelines.repository import OCRRepository


class OCRHit(TypedDict):
    """A single normalized OCR search hit, before being enriched with FAISS metadata."""

    score: float
    dataset: str
    video_id: str
    keyframe_id: str
    texts: list[str]


class SearchPipeline:
    """Full-text OCR search over indexed keyframe texts (Elasticsearch-backed)."""

    def __init__(self, repository: OCRRepository) -> None:
        self.repository = repository

    def search(self, query: str, top_k: int = 10) -> list[OCRHit]:
        """Run a full-text OCR search and return normalized hits.

        Args:
            query: Free-text search query (matched against OCR'd texts).
            top_k: Maximum number of hits to return.

        Returns:
            A list of normalized OCR hits, ordered by descending relevance score.
        """
        query = str(query or "").strip()
        if not query:
            return []

        raw_hits: list[dict[str, Any]] = self.repository.search(text=query, top_k=top_k)

        hits: list[OCRHit] = []
        for hit in raw_hits:
            source = hit["_source"]
            hits.append(
                OCRHit(
                    score=float(hit["_score"]),
                    dataset=source["dataset"],
                    video_id=source["video_id"],
                    keyframe_id=source["keyframe_id"],
                    texts=list(source.get("texts", [])),
                )
            )
        return hits
