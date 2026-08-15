"""FastAPI dependencies dùng chung cho mọi router legacy — đọc lại state đã
được `api/main.py` (lifespan) build 1 lần (`request.app.state.*`), KHÔNG tự
build gì thêm ở đây (giống pattern `get_retrieval_system` trong
`routers/search.py` của nhánh mới).
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Request

from src.api.legacy.paths import LegacyPaths
from src.retrieval.index.faiss_index import FaissIndex
from src.retrieval.system import RetrievalSystem


def get_legacy_system(request: Request) -> RetrievalSystem:
    """Giống hệt `get_retrieval_system` trong `routers/search.py`: 503 thay
    vì AttributeError nếu app chưa khởi động xong hoặc `build_system()` đã
    fail lúc lifespan."""

    system = getattr(request.app.state, "retrieval_system", None)
    if system is None:
        raise HTTPException(
            status_code=503,
            detail="Retrieval system chưa sẵn sàng (chưa khởi tạo xong hoặc khởi tạo thất bại).",
        )
    return system


def get_legacy_index(request: Request) -> FaissIndex:
    """`clip_index` tương đương bản gốc — model mặc định (xem
    `Orchestrator.index`, luôn resolve về 1 `FaissIndex` cụ thể kể cả khi
    backend load nhiều model)."""

    return get_legacy_system(request).orchestrator.index


def get_cfg(request: Request) -> dict[str, Any]:
    """Toàn bộ `configs/app.yaml` đã parse, lưu ở lifespan."""

    return request.app.state.cfg


def get_legacy_paths(request: Request) -> LegacyPaths:
    return request.app.state.legacy_paths
