"""Router duy nhất gọi vào `RetrievalSystem` (xem `src/retrieval/system.py`
— "facade DUY NHẤT giữa `api/` và `retrieval/retriever/*`").

Router KHÔNG tự chọn retriever, KHÔNG biết gì về `Orchestrator` — chỉ:
  1. Validate HTTP input qua `api/schemas/search.py`.
  2. Gọi đúng 1 hàm `RetrievalSystem.search_*`.
  3. Serialize DataFrame kết quả -> `SearchResultItem` (xem `_records_from_df`).

`RetrievalSystem` được build 1 lần lúc app khởi động (`api/main.py` lifespan)
và lấy lại qua `request.app.state.retrieval_system` (`get_retrieval_system`
dependency bên dưới) — router không tự gọi `build_system(...)`.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from src.api.schemas.search import (
    AdvancedSearchRequest,
    AdvancedSearchResponse,
    AsrSearchRequest,
    AutoSearchRequest,
    AvailableModelsResponse,
    OcrSearchRequest,
    QueryPlanOut,
    SearchResponse,
    SearchResultItem,
    SemanticSearchRequest,
    TemporalSearchRequest,
)
from src.retrieval.retriever.common.orchestrator import QueryPlan
from src.retrieval.system import RetrievalSystem

router = APIRouter(prefix="/api/search", tags=["search"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_retrieval_system(request: Request) -> RetrievalSystem:
    """Lấy `RetrievalSystem` singleton đã build lúc lifespan (xem
    `api/main.py`). 503 thay vì AttributeError nếu app chưa khởi động xong
    hoặc build_system từng fail — tránh lộ stacktrace nội bộ ra client."""

    system = getattr(request.app.state, "retrieval_system", None)
    if system is None:
        raise HTTPException(
            status_code=503,
            detail="Retrieval system chưa sẵn sàng (chưa khởi tạo xong hoặc khởi tạo thất bại).",
        )
    return system


def _sanitize(value: Any) -> Any:
    """Chuẩn hoá 1 giá trị lấy từ DataFrame/dict lồng nhau về kiểu JSON-safe
    thuần Python (FastAPI/Pydantic serialize numpy scalar hoặc NaN sẽ lỗi
    hoặc ra JSON không hợp lệ `NaN`)."""

    if isinstance(value, (np.generic,)):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    if value is pd.NaT:
        return None
    if isinstance(value, np.ndarray):
        return [_sanitize(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize(v) for v in value]
    return value


def _records_from_df(df: pd.DataFrame) -> list[dict[str, Any]]:
    """DataFrame -> list[dict] JSON-safe. Không hard-code tên cột: mỗi
    search mode trả về 1 tập cột khác nhau trên cùng 1 shape tối thiểu
    (xem docstring `api/schemas/search.py`), nên serialize NGUYÊN mọi cột
    thay vì chọn lọc, để không lỡ làm rớt field mới thêm về sau."""

    if df is None or df.empty:
        return []
    records = df.to_dict(orient="records")
    return [{k: _sanitize(v) for k, v in row.items()} for row in records]


def _query_plan_out(plan: QueryPlan) -> QueryPlanOut:
    return QueryPlanOut(mode=plan.mode, use_split=plan.use_split, events=plan.events)


def _run_or_503(fn, *, feature_name: str):
    """OCR/ASR là subsystem optional (xem `RetrievalSystem`/`Orchestrator`)
    — khi chưa được cấu hình, `RuntimeError` được raise ra từ orchestrator.
    Chuyển thành 503 (thay vì 500) để client phân biệt được "tính năng
    chưa bật" với "lỗi thật".
    """

    try:
        return fn()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=f"{feature_name} không khả dụng: {exc}") from exc
    except KeyError as exc:
        # model_key không tồn tại trong IndexManager, xem IndexManager.get().
        raise HTTPException(status_code=400, detail=str(exc).strip('"')) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/models", response_model=AvailableModelsResponse)
def list_available_models(request: Request) -> AvailableModelsResponse:
    system = get_retrieval_system(request)
    return AvailableModelsResponse(models=system.available_models())


@router.post("/semantic", response_model=SearchResponse)
def search_semantic(payload: SemanticSearchRequest, request: Request) -> SearchResponse:
    system = get_retrieval_system(request)
    df, plan = _run_or_503(
        lambda: system.search_semantic(
            payload.query,
            top_k=payload.top_k,
            use_split=payload.use_split,
            candidate_multiplier=payload.candidate_multiplier,
            model_key=payload.model_key,
            translate=payload.translate,
        ),
        feature_name="Semantic search",
    )
    records = _records_from_df(df)
    return SearchResponse(
        query=payload.query,
        mode="semantic",
        count=len(records),
        results=[SearchResultItem(**r) for r in records],
        query_plan=_query_plan_out(plan),
    )


@router.post("/temporal", response_model=SearchResponse)
def search_temporal(payload: TemporalSearchRequest, request: Request) -> SearchResponse:
    system = get_retrieval_system(request)
    df, plan = _run_or_503(
        lambda: system.search_temporal(
            payload.query,
            top_k=payload.top_k,
            use_split=payload.use_split,
            candidate_multiplier=payload.candidate_multiplier,
            duration_limit=payload.duration_limit,
            model_key=payload.model_key,
            translate=payload.translate,
        ),
        feature_name="Temporal search",
    )
    records = _records_from_df(df)
    return SearchResponse(
        query=payload.query,
        mode="temporal",
        count=len(records),
        results=[SearchResultItem(**r) for r in records],
        query_plan=_query_plan_out(plan),
    )


@router.post("/ocr", response_model=SearchResponse)
def search_ocr(payload: OcrSearchRequest, request: Request) -> SearchResponse:
    system = get_retrieval_system(request)
    df, plan = _run_or_503(
        lambda: system.search_ocr(payload.query, top_k=payload.top_k),
        feature_name="OCR search",
    )
    records = _records_from_df(df)
    return SearchResponse(
        query=payload.query,
        mode="ocr",
        count=len(records),
        results=[SearchResultItem(**r) for r in records],
        query_plan=_query_plan_out(plan),
    )


@router.post("/asr", response_model=SearchResponse)
def search_asr(payload: AsrSearchRequest, request: Request) -> SearchResponse:
    system = get_retrieval_system(request)
    df, plan = _run_or_503(
        lambda: system.search_asr(payload.query, top_k=payload.top_k),
        feature_name="ASR search",
    )
    records = _records_from_df(df)
    return SearchResponse(
        query=payload.query,
        mode="asr",
        count=len(records),
        results=[SearchResultItem(**r) for r in records],
        query_plan=_query_plan_out(plan),
    )


@router.post("/auto", response_model=SearchResponse)
def search_auto(payload: AutoSearchRequest, request: Request) -> SearchResponse:
    system = get_retrieval_system(request)
    df, plan = _run_or_503(
        lambda: system.search_auto(
            payload.query,
            top_k=payload.top_k,
            use_split=payload.use_split,
            candidate_multiplier=payload.candidate_multiplier,
            translate=payload.translate,
        ),
        feature_name="Auto search",
    )
    records = _records_from_df(df)
    return SearchResponse(
        query=payload.query,
        mode="auto",
        count=len(records),
        results=[SearchResultItem(**r) for r in records],
        query_plan=_query_plan_out(plan),
    )


@router.post("/advanced", response_model=AdvancedSearchResponse)
def search_advanced(payload: AdvancedSearchRequest, request: Request) -> AdvancedSearchResponse:
    """Wire trực tiếp `Orchestrator.advanced_search` qua
    `RetrievalSystem.search_advanced` — xem docstring 2 hàm đó cho contract
    đầy đủ (semantic_models + temporal on/off dùng chung list, KHÔNG có
    `temporal_models` riêng)."""

    system = get_retrieval_system(request)
    fused_df, per_source = _run_or_503(
        lambda: system.search_advanced(
            payload.query,
            semantic_models=payload.semantic_models,
            temporal=payload.temporal,
            use_ocr=payload.use_ocr,
            use_asr=payload.use_asr,
            top_k=payload.top_k,
            use_split=payload.use_split,
            candidate_multiplier=payload.candidate_multiplier,
            duration_limit=payload.duration_limit,
            weights=payload.weights,
        ),
        feature_name="Advanced search",
    )

    records = _records_from_df(fused_df)
    per_source_out = None
    if payload.include_per_source:
        per_source_out = {
            label: [SearchResultItem(**r) for r in _records_from_df(df)] for label, df in per_source.items()
        }

    return AdvancedSearchResponse(
        query=payload.query,
        count=len(records),
        results=[SearchResultItem(**r) for r in records],
        per_source=per_source_out,
    )
