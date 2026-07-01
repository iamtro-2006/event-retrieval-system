from __future__ import annotations

from pydantic import BaseModel, Field


class ASRDocument(BaseModel):

    dataset: str = Field(...)

    video_id: str = Field(...)

    segment_id: str = Field(...)

    start_time: float = Field(...)

    end_time: float = Field(...)

    text: str = Field(default="")
