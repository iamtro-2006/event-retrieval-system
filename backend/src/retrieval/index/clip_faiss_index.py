"""Backward-compat shim.

This module used to contain `ClipFaissIndex`, hardcoded to open_clip only.
It has been generalized and moved to `faiss_index.py` (class `FaissIndex`,
loads any backend registered in `embedding_extraction.models.registry`).

Keep importing from here if you have old code (`from
src.retrieval.index.clip_faiss_index import ClipFaissIndex`) — it still
works, `ClipFaissIndex` is just an alias for `FaissIndex` now. New code
should import `FaissIndex` from `src.retrieval.index.faiss_index` directly.
"""
from __future__ import annotations

from src.retrieval.index.faiss_index import (  # noqa: F401
    METADATA_DISPLAY_COLUMNS,
    FaissIndex,
    resolve_device,
)

# Old name kept as an alias so existing imports don't break.
ClipFaissIndex = FaissIndex
