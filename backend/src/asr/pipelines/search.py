from __future__ import annotations

from typing import Any, TypedDict

from src.asr.pipelines.repository import ASRRepository


class ASRHit(TypedDict):
    """A single normalized ASR search hit (one speech segment), before being
    expanded to the set of keyframes that fall inside its [start_time, end_time]
    window."""

    score: float
    dataset: str
    video_id: str
    segment_id: str
    start_time: float
    end_time: float
    text: str


class SearchPipeline:
    """Full-text ASR search over indexed speech transcript segments (Elasticsearch-backed)."""

    def __init__(self, repository: ASRRepository) -> None:
        self.repository = repository

    def search(self, query: str, top_k: int = 10) -> list[ASRHit]:
        """Run a full-text ASR search and return normalized hits.

        Args:
            query: Free-text search query (matched against transcript text).
            top_k: Maximum number of hits to return.

        Returns:
            A list of normalized ASR hits (one per matched segment), ordered by
            descending relevance score.
        """
        query = str(query or "").strip()
        if not query:
            return []

        raw_hits: list[dict[str, Any]] = self.repository.search(text=query, top_k=top_k)

        hits: list[ASRHit] = []
        for hit in raw_hits:
            source = hit["_source"]
            hits.append(
                ASRHit(
                    score=float(hit["_score"]),
                    dataset=source["dataset"],
                    video_id=source["video_id"],
                    segment_id=source["segment_id"],
                    start_time=float(source.get("start_time", 0.0)),
                    end_time=float(source.get("end_time", 0.0)),
                    text=str(source.get("text", "")),
                )
            )
        return hits
