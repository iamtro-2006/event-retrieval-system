from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SearchModeIn = Literal["semantic", "temporal", "ocr", "auto"]


class SearchRequest(BaseModel):
    """Request body for POST /api/search."""

    query: str = Field(..., min_length=1, description="Search query text.")
    mode: SearchModeIn = Field("semantic", description="Search mode to execute.")
    top_k: int = Field(20, ge=1, le=200, description="Number of results to return.")
    use_split: bool = Field(True, description="Split comma/period-separated sub-queries (semantic/temporal only).")
    candidate_multiplier: int = Field(5, ge=1, le=50, description="Candidate oversampling factor (semantic/temporal only).")
    duration_limit: float = Field(-1, description="Max event duration in seconds for temporal search (-1 = unlimited).")


class SearchResultItem(BaseModel):
    """A single result row, shaped identically regardless of search mode."""

    dataset: str | None = None
    video_id: str | None = None
    keyframe_id: str | None = None
    frame_idx: int | None = None
    timestamp_sec: float | None = None
    fps: float | None = None
    keyframe_path: str | None = None
    rank: int
    retrieval_score: float

    # OCR-only field: on-screen text strings that matched the query.
    matched_texts: list[str] | None = None

    model_config = {"extra": "allow"}


class SearchResponse(BaseModel):
    """Response body for POST /api/search."""

    query: str
    mode: str
    effective_mode: str
    total_results: int
    results: list[SearchResultItem]
