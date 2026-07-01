from __future__ import annotations

import math

import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from api.schemas.search import SearchRequest, SearchResponse
from src.retrieval.models.retrieval_system import FaissRetrievalSystem

router = APIRouter(prefix="/api/search", tags=["search"])


def get_retrieval_system(request: Request) -> FaissRetrievalSystem:
    """Fetch the shared `FaissRetrievalSystem` instance from app state.

    The instance is built once at startup (see `api/main.py`) and reused across
    requests since it holds the FAISS index, CLIP model, and vector cache in memory.
    """
    system: FaissRetrievalSystem | None = getattr(request.app.state, "retrieval_system", None)
    if system is None:
        raise HTTPException(status_code=503, detail="Retrieval system is not initialized yet.")
    return system


def _dataframe_to_results(df: pd.DataFrame) -> list[dict]:
    """Convert a results DataFrame into JSON-safe records for the API response.

    Replaces NaN/inf (which are not valid JSON) with None, and keeps only
    JSON-serializable primitive types.
    """
    if df.empty:
        return []

    df = df.where(pd.notnull(df), None)
    records = df.to_dict(orient="records")

    for record in records:
        for key, value in record.items():
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                record[key] = None
            elif isinstance(value, (pd.Timestamp,)):
                record[key] = str(value)
    return records


@router.post("", response_model=SearchResponse)
def search(payload: SearchRequest, request: Request) -> SearchResponse:
    """Unified search endpoint.

    Supports semantic (CLIP), temporal (multi-event), and OCR (on-screen text)
    search behind a single contract, so the frontend can render results the
    same way regardless of `mode`.
    """
    system = get_retrieval_system(request)

    try:
        if payload.mode == "ocr":
            # OCR search uses the raw query text directly (Elasticsearch handles
            # tokenization/fuzzy matching); the semantic/temporal query-splitting
            # logic does not apply here.
            df = system.ocr_search(payload.query, top_k=payload.top_k)
            effective_mode = "ocr"
        else:
            df, plan = system.run_search(
                query=payload.query,
                mode=payload.mode,
                use_split=payload.use_split,
                top_k=payload.top_k,
                candidate_multiplier=payload.candidate_multiplier,
                duration_limit=payload.duration_limit,
            )
            if payload.mode == "auto":
                effective_mode = "temporal" if len(plan.events) > 1 else "semantic"
            else:
                effective_mode = payload.mode
    except RuntimeError as exc:
        # e.g. OCR backend not configured/injected.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    results = _dataframe_to_results(df)
    return SearchResponse(
        query=payload.query,
        mode=payload.mode,
        effective_mode=effective_mode,
        total_results=len(results),
        results=results,
    )
