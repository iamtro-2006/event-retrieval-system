from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel

from src.index.query_planning import SearchMode


class SearchRequest(BaseModel):
    query: str
    top_k: int | None = None
    candidate_multiplier: int | None = None
    use_split: bool | None = None
    use_translate: bool | None = None
    search_mode: SearchMode | None = "semantic"
    duration_limit: float | None = -1
    search_ef: int | None = None


class DresLoginRequest(BaseModel):
    dres_url: str
    username: str
    password: str


class DresSubmitRequest(BaseModel):
    dres_url: str
    session_id: str
    evaluation_id: str | None = None
    video_id: str
    frame_id: int
    timestamp: float | None = None


class SimilaritySearchRequest(BaseModel):
    video_id: str
    frame_id: int
    top_k: int | None = None
