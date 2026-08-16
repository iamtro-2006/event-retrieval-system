from __future__ import annotations

from pydantic import BaseModel, Field


class MultimodalTextDocument(BaseModel):
    """A single keyframe document in the unified `multimodal_text` index.

    Unlike the legacy split (OCR index per keyframe, ASR index per speech
    segment), this document is *keyframe-level* and carries BOTH the on-screen
    text and the speech transcript snapped to that keyframe, so a single
    `multi_match` query can fuse the two signals with per-field boosting.
    """

    dataset: str = Field(...)

    video_id: str = Field(...)

    keyframe_id: str = Field(...)

    timestamp_sec: float = Field(...)

    ocr_text: str = Field(default="")

    asr_text: str = Field(default="")
