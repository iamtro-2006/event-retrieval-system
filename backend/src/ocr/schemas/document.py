from __future__ import annotations

from pydantic import BaseModel, Field


class OCRDocument(BaseModel):

    dataset: str = Field(...)

    video_id: str = Field(...)

    keyframe_id: str = Field(...)

    bboxs: list[list[float]] = Field(...)

    texts: list[str] = Field(default_factory=list)