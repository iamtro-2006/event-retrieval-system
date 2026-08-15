from __future__ import annotations

from typing import Any

from src.retrieval.index.faiss_index import FaissIndex
from src.retrieval.index.index_manager import IndexManager, build_index_manager


def build_faiss_index(config: dict) -> FaissIndex:
    """Build a single `FaissIndex` từ config dict (index_path, metadata_path,
    model_name, backend, pretrained, device, precision, ... — xem
    `FaissIndex.__init__`). Dùng khi chỉ cần đúng 1 model; để chọn được
    model nào load cho semantic/temporal, dùng `build_index_manager(...)`.
    """
    return FaissIndex(**config)


def build_clip_faiss_index(config: dict) -> FaissIndex:
    """Deprecated alias, kept for backward compatibility."""
    return build_faiss_index(config)


def build_index_manager_from_config(config: dict[str, Any] | list[dict[str, Any]]) -> IndexManager:
    """Build an `IndexManager` (multiple models) — see `index_manager.py`."""
    return build_index_manager(config)
