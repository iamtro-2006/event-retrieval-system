"""API search GỐC (`main.py` cũ) — port nguyên contract, KHÔNG đổi trừ khi
thật sự cần thiết. Đây là endpoint frontend cũ đang gọi thẳng:

  - `POST /api/search`: endpoint search HỢP NHẤT, chọn mode qua
    `payload.search_mode` ("semantic" | "temporal" | "ocr" | "asr" | "auto").
    Khác với nhánh MỚI (`src/api/routers/search.py`, prefix `/api/search/...`
    — mỗi mode 1 endpoint riêng, response shape gọn/strict-typed hơn), route
    này trả nguyên `dict_to_result_FAST(...)` (field `image_url`,
    `video_url`, `map_url`, `matched_sequence`, `temporal`, ... mà UI cũ
    render trực tiếp) — 2 nhánh cùng tồn tại song song, không nhánh nào thay
    thế nhánh nào.
  - `GET /api/frame-info`, `GET /api/surrounding-frames`,
    `POST /api/similarity-search`: không có tương đương ở nhánh search mới,
    giữ nguyên như bản gốc.

Chỉ khác bản gốc 1 điểm bắt buộc: lấy `RetrievalSystem`/`Orchestrator` qua
`request.app.state` (build 1 lần trong `api/main.py` lifespan) thay vì tự
`build_system()` ở top-level module như file `main.py` cũ.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from src.api.legacy.deps import get_cfg, get_legacy_index, get_legacy_paths, get_legacy_system
from src.api.legacy.paths import LegacyPaths
from src.api.legacy.serializers import (
    dict_to_result_FAST,
    find_metadata_row,
    resolve_keyframe_path_from_dict,
    safe_int,
)
from src.api.legacy.translate import translate_query_if_needed
from src.api.schemas.legacy import FusionSearchRequest, SearchRequest, SimilaritySearchRequest
from src.retrieval.index.faiss_index import FaissIndex
from src.retrieval.system import RetrievalSystem

router = APIRouter(tags=["legacy-search"])


@router.get("/api/frame-info")
async def get_frame_info(
    video_id: str,
    keyframe_id: int,
    request: Request,
    clip_index: FaissIndex = Depends(get_legacy_index),
    paths: LegacyPaths = Depends(get_legacy_paths),
):
    """Retrieve detailed metadata for a specific video frame."""

    def _fetch():
        row = find_metadata_row(clip_index, video_id, keyframe_id)
        return dict_to_result_FAST(row.to_dict(), paths.keyframes_root, paths.backend_dir)

    return await run_in_threadpool(_fetch)


@router.get("/api/video-preview")
async def get_video_preview(
    video_id: str,
    request: Request,
    frame_id: int | None = None,
    frame_idx: int | None = None,
    timestamp_ms: int | None = None,
    clip_index: FaissIndex = Depends(get_legacy_index),
    paths: LegacyPaths = Depends(get_legacy_paths),
):
    """Resolve a video plus frame_id or milliseconds to the normal video result shape."""
    if frame_id is not None and frame_idx is not None:
        raise HTTPException(status_code=400, detail="Provide only one of frame_id or frame_idx")
    requested_frame = frame_idx if frame_idx is not None else frame_id
    if (requested_frame is None) == (timestamp_ms is None):
        raise HTTPException(status_code=400, detail="Provide exactly one of frame_id/frame_idx or timestamp_ms")
    if timestamp_ms is not None and timestamp_ms < 0:
        raise HTTPException(status_code=400, detail="timestamp_ms must be non-negative")

    def _fetch():
        metadata = clip_index.metadata
        rows = metadata[metadata["video_id"].astype(str) == str(video_id)].copy()
        if rows.empty:
            raise HTTPException(status_code=404, detail=f"Video not found: {video_id}")
        if requested_frame is not None:
            # `frame_id` in the preview UI is the decoded video frame index,
            # not the sparse keyframe id used by the retrieval index.
            # Metadata may contain one row per keyframe, so select the row
            # whose actual frame_idx is closest to the requested frame.
            frame_col = next((c for c in ("frame_idx", "frame_id") if c in rows.columns), None)
            if frame_col:
                frame_values = pd.to_numeric(rows[frame_col], errors="coerce")
                valid = frame_values.notna()
                if valid.any():
                    row = rows.loc[(frame_values[valid] - int(requested_frame)).abs().idxmin()]
                else:
                    raise HTTPException(status_code=422, detail="Frame index metadata is unavailable for this video")
            else:
                # Backward compatibility for old indexes that only expose
                # keyframe ids.
                row = find_metadata_row(clip_index, video_id, int(requested_frame))
            fps = pd.to_numeric(row.get("fps", 0), errors="coerce")
            if float(fps) > 0:
                result_timestamp = float(requested_frame) / float(fps)
            else:
                result_timestamp = pd.to_numeric(
                    row.get("timestamp_sec", row.get("timestamp", 0)), errors="coerce"
                )
            # Applied after serializing below so the video seeks to the
            # requested real frame rather than the nearest sparse keyframe.
        else:
            time_col = next((c for c in ("timestamp_sec", "timestamp", "time_sec") if c in rows.columns), None)
            if not time_col:
                raise HTTPException(status_code=422, detail="Timestamp metadata is unavailable for this video")
            values = pd.to_numeric(rows[time_col], errors="coerce")
            target_sec = float(timestamp_ms) / 1000.0
            row = rows.loc[(values - target_sec).abs().idxmin()]
        result = dict_to_result_FAST(row.to_dict(), paths.keyframes_root, paths.backend_dir)
        if timestamp_ms is not None:
            result["timestamp"] = float(timestamp_ms) / 1000.0
            # `frame_id` in the index is a keyframe id. The frame id requested
            # by the video player is the actual decoded video frame, derived
            # from the playback timestamp and that video's FPS.
            fps = pd.to_numeric(row.get("fps", 0), errors="coerce")
            if float(fps) > 0:
                result["frame_idx"] = int(round(float(timestamp_ms) / 1000.0 * float(fps)))
        elif requested_frame is not None:
            result["frame_idx"] = int(requested_frame)
            if "result_timestamp" in locals() and pd.notna(result_timestamp):
                result["timestamp"] = float(result_timestamp)
        return result

    return await run_in_threadpool(_fetch)


@router.post("/api/search")
async def search_api(
    payload: SearchRequest,
    request: Request,
    system: RetrievalSystem = Depends(get_legacy_system),
    cfg: dict[str, Any] = Depends(get_cfg),
    paths: LegacyPaths = Depends(get_legacy_paths),
):
    """Execute a multi-modal retrieval search based on the query payload.

    `payload.search_mode` selects the retrieval strategy: "semantic" (CLIP),
    "temporal" (multi-event), "ocr" (on-screen text via Elasticsearch), "asr"
    (speech transcript via Elasticsearch), or "auto" (semantic/temporal
    auto-detected from query structure). All modes return results through the
    same `dict_to_result_FAST` serializer, so the frontend renders them with
    the same result-card component, but the *shape* of each result differs by
    mode:
      - "ocr": one row per matched keyframe, no `matched_sequence` -> renders
        identically to "semantic" (a plain result grid), with a non-empty
        `matched_texts` field showing the on-screen text that matched.
      - "asr": one row per matched speech segment, with a populated
        `matched_sequence` (every keyframe spoken during that segment) -> the
        frontend's existing temporal-sequence detection renders it identically
        to "temporal" (a horizontal frame-chain strip), with `matched_texts`
        carrying the segment's transcript.
    """
    orchestrator = system.orchestrator

    original_query = payload.query.strip()
    if not original_query:
        raise HTTPException(status_code=400, detail="Query is empty")

    max_top_k = int(cfg["search"].get("max_top_k", 200))
    default_top_k = int(cfg["search"].get("default_top_k", 20))
    top_k = max(1, min(int(payload.top_k or default_top_k), max_top_k))

    candidate_multiplier = max(1, int(payload.candidate_multiplier or cfg["search"].get("candidate_multiplier", 5)))
    use_split = True if payload.use_split is None else bool(payload.use_split)
    use_translate = (
        bool(cfg.get("translate", {}).get("enabled_default", False))
        if payload.use_translate is None
        else bool(payload.use_translate)
    )

    mode = payload.search_mode
    duration_limit = -1.0 if payload.duration_limit is None or payload.duration_limit == 0 else float(payload.duration_limit)

    # OCR/ASR search match raw text via Elasticsearch (its own tokenizer/fuzzy
    # matching, over on-screen text or Vietnamese speech transcripts), so
    # translation is skipped for those modes: translating the query would
    # work against literal signage/labels or the transcript language.
    should_translate = use_translate and mode not in ("ocr", "asr")

    start = time.perf_counter()

    try:
        search_query = await run_in_threadpool(
            translate_query_if_needed,
            query=original_query,
            use_translate=should_translate,
            cfg=cfg,
            backend_dir=paths.backend_dir,
            provider=payload.translate_provider,
            api_key=payload.translate_api_key,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Translate failed: {type(exc).__name__}: {exc}")

    try:
        results_df, query_plan = await run_in_threadpool(
            orchestrator.run_search, query=search_query, mode=mode, use_split=use_split,
            reasoning=bool(payload.reasoning),
            top_k=top_k, candidate_multiplier=candidate_multiplier, duration_limit=duration_limit,
        )
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc))
    except RuntimeError as exc:
        # e.g. OCR requested but Elasticsearch is unavailable/not configured.
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Search failed: {type(exc).__name__}: {exc}")

    latency_ms = round((time.perf_counter() - start) * 1000)
    candidate_k = max(top_k * candidate_multiplier, top_k)

    response_base = {
        "original_query": original_query, "query": search_query,
        "translated_query": search_query if should_translate else None,
        "use_translate": should_translate, "use_split": use_split, "reasoning": payload.reasoning,
        "mode": mode, "search_mode": mode,
        "duration_limit": duration_limit, "top_k": top_k, "candidate_multiplier": candidate_multiplier,
        "candidate_k": candidate_k, "latency_ms": latency_ms,
        "events": query_plan.events, "event_queries": query_plan.event_queries, "sub_queries": query_plan.flat_queries,
    }

    if results_df.empty:
        return {**response_base, "count": 0, "results": []}

    try:
        def _serialize_results():
            cols_to_keep = [c for c in [
                "video_id", "keyframe_id", "keyframe_id_int", "frame_idx", "dataset",
                "keyframe_path", "video_path", "timestamp_sec", "timestamp", "fps",
                "score", "retrieval_score", "avg_score", "caption", "matched_sequence",
                "temporal_start_time", "temporal_end_time", "temporal_duration_sec", "video_score",
                # OCR-specific columns (see FaissRetrievalSystem._enrich_ocr_hits)
                "matched_texts", "ocr_score",
                # ASR-specific columns (see FaissRetrievalSystem._enrich_asr_hits)
                "asr_score", "segment_id",
            ] if c in results_df.columns]
            records = results_df[cols_to_keep].to_dict(orient="records")
            return [dict_to_result_FAST(rec, paths.keyframes_root, paths.backend_dir) for rec in records]

        results = await run_in_threadpool(_serialize_results)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Serialize failed: {type(exc).__name__}: {exc}")

    return {**response_base, "count": len(results), "results": results}


@router.post("/api/search/fusion")
async def search_fusion_api(
    payload: FusionSearchRequest,
    request: Request,
    system: RetrievalSystem = Depends(get_legacy_system),
    cfg: dict[str, Any] = Depends(get_cfg),
    paths: LegacyPaths = Depends(get_legacy_paths),
):
    """Advanced/fusion search (RRF trên nhiều semantic model đã tick +
    temporal/ocr/asr on-off, mỗi nguồn có weight riêng qua `payload.weights`)
    — xem `RetrievalSystem.search_advanced`/`Orchestrator.advanced_search`
    cho contract đầy đủ. Trả về CÙNG shape với `POST /api/search` (dùng lại
    `dict_to_result_FAST`) để frontend render bằng đúng result-card hiện có,
    không cần thêm 1 nhánh UI riêng cho fusion.
    """

    original_query = payload.query.strip()
    if not original_query:
        raise HTTPException(status_code=400, detail="Query is empty")

    max_top_k = int(cfg["search"].get("max_top_k", 200))
    default_top_k = int(cfg["search"].get("default_top_k", 20))
    top_k = max(1, min(int(payload.top_k or default_top_k), max_top_k))

    candidate_multiplier = max(1, int(payload.candidate_multiplier or cfg["search"].get("candidate_multiplier", 5)))
    use_split = True if payload.use_split is None else bool(payload.use_split)
    requested_translate = (
        bool(cfg.get("translate", {}).get("enabled_default", False))
        if payload.use_translate is None
        else bool(payload.use_translate)
    )
    # Semantic/temporal and OCR/ASR now receive separate query paths in the
    # backend: semantic gets the translated query, OCR/ASR get original text.
    use_translate = requested_translate
    duration_limit = -1.0 if payload.duration_limit is None or payload.duration_limit == 0 else float(payload.duration_limit)

    start = time.perf_counter()

    try:
        search_query = original_query
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Translate failed: {type(exc).__name__}: {exc}")

    try:
        fused_df, _per_source = await run_in_threadpool(
            system.search_advanced,
            original_query,
            semantic_models=payload.semantic_models,
            temporal=payload.temporal,
            use_ocr=payload.use_ocr,
            use_asr=payload.use_asr,
            top_k=top_k,
            use_split=use_split,
            candidate_multiplier=candidate_multiplier,
            duration_limit=duration_limit,
            weights=payload.weights,
            translate=use_translate,
            translate_provider=payload.translate_provider,
            translate_api_key=payload.translate_api_key,
            reasoning=payload.reasoning,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc).strip('"'))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Fusion search failed: {type(exc).__name__}: {exc}")

    latency_ms = round((time.perf_counter() - start) * 1000)

    translated_query = fused_df.attrs.get("translated_query") if fused_df is not None else None
    response_base = {
        "original_query": original_query, "query": translated_query or search_query,
        "translated_query": translated_query,
        "use_translate": use_translate, "use_split": use_split, "reasoning": payload.reasoning,
        "mode": "fusion", "search_mode": "fusion",
        "duration_limit": duration_limit, "top_k": top_k, "candidate_multiplier": candidate_multiplier,
        "latency_ms": latency_ms,
        "semantic_models": payload.semantic_models, "temporal": payload.temporal,
        "use_ocr": payload.use_ocr, "use_asr": payload.use_asr,
        # Expose the effective plan/weights used by the backend.  The UI can
        # distinguish an actual LLM rewrite from regex/default fallback.
        "weights": (fused_df.attrs.get("weights") if fused_df is not None else None) or payload.weights or {},
        "events": fused_df.attrs.get("events", []) if fused_df is not None else [],
        "event_queries": fused_df.attrs.get("event_queries", []) if fused_df is not None else [],
        "focused_queries": fused_df.attrs.get("focused_queries", {}) if fused_df is not None else {},
    }

    if fused_df is None or fused_df.empty:
        return {**response_base, "count": 0, "results": []}

    try:
        def _serialize_results():
            records = fused_df.to_dict(orient="records")
            return [dict_to_result_FAST(rec, paths.keyframes_root, paths.backend_dir) for rec in records]

        results = await run_in_threadpool(_serialize_results)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Serialize failed: {type(exc).__name__}: {exc}")

    return {**response_base, "count": len(results), "results": results}


@router.get("/api/surrounding-frames")
async def get_surrounding_frames(
    video_id: str,
    keyframe_id: int,
    request: Request,
    radius: int = 10,
    clip_index: FaissIndex = Depends(get_legacy_index),
    paths: LegacyPaths = Depends(get_legacy_paths),
):
    """Retrieve a sequence of frames surrounding a target keyframe."""

    def _fetch_surround():
        metadata_df = clip_index.metadata
        radius_val = max(1, min(int(radius), 50))
        target_keyframe_id = int(keyframe_id)

        row_ids = clip_index._rows_by_video.get(str(video_id))
        if row_ids is None or len(row_ids) == 0:
            raise HTTPException(status_code=404, detail=f"Video not found: {video_id}")

        video_df = metadata_df.iloc[row_ids].copy()
        if video_df.empty:
            raise HTTPException(status_code=404, detail=f"Video not found: {video_id}")

        # Vectorized sorting key extraction (replaces slow .apply(lambda))
        sort_col = next((c for c in ["keyframe_id_int", "keyframe_id", "frame_idx"] if c in video_df.columns), None)
        if not sort_col:
            raise HTTPException(status_code=500, detail="Metadata missing keyframe identifier column")

        video_df["_keyframe_sort"] = pd.to_numeric(video_df[sort_col], errors="coerce").fillna(0).astype(np.int32)
        video_df = video_df.sort_values("_keyframe_sort", kind="stable").reset_index(drop=True)

        center_matches = video_df[video_df["_keyframe_sort"] == target_keyframe_id]
        if center_matches.empty:
            raise HTTPException(status_code=404, detail=f"Frame not found: {video_id}/{keyframe_id}")

        center_pos = int(center_matches.index[0])
        start_pos = max(0, center_pos - radius_val)
        end_pos = min(len(video_df), center_pos + radius_val + 1)

        surrounding_df = video_df.iloc[start_pos:end_pos]
        records = surrounding_df.to_dict(orient="records")

        frames = []
        for rec in records:
            item = dict_to_result_FAST(rec, paths.keyframes_root, paths.backend_dir)
            item["is_surround_center"] = safe_int(item.get("frame_id"), -1) == target_keyframe_id
            item["surround_offset"] = safe_int(item.get("frame_id"), 0) - target_keyframe_id
            frames.append(item)

        return radius_val, frames

    r_val, frames_result = await run_in_threadpool(_fetch_surround)
    return {
        "video_id": video_id, "center_frame_id": int(keyframe_id),
        "radius": r_val, "count": len(frames_result), "frames": frames_result,
    }


@router.post("/api/similarity-search")
async def similarity_search_api(
    payload: SimilaritySearchRequest,
    request: Request,
    system: RetrievalSystem = Depends(get_legacy_system),
    clip_index: FaissIndex = Depends(get_legacy_index),
    paths: LegacyPaths = Depends(get_legacy_paths),
):
    """Execute an image-to-image similarity search based on a source frame."""
    orchestrator = system.orchestrator

    def _run_sim_search():
        top_k = 20

        row = find_metadata_row(clip_index, payload.video_id, payload.frame_id)
        source_dict = row.to_dict()
        image_path = resolve_keyframe_path_from_dict(source_dict, paths.keyframes_root, paths.backend_dir)

        results_df = orchestrator.semantic_search.similarity_search_by_image(image_path=Path(image_path), top_k=top_k)
        return source_dict, results_df

    start = time.perf_counter()
    try:
        source_dict_data, df_results = await run_in_threadpool(_run_sim_search)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Similarity search failed: {type(exc).__name__}: {exc}")

    latency_ms = round((time.perf_counter() - start) * 1000)

    if df_results.empty:
        return {
            "query": f"similarity:{payload.video_id}/{payload.frame_id:06d}",
            "search_mode": "similarity", "latency_ms": latency_ms, "count": 0, "results": [],
        }

    def _parse():
        return [dict_to_result_FAST(rec, paths.keyframes_root, paths.backend_dir) for rec in df_results.to_dict(orient="records")]

    results = await run_in_threadpool(_parse)

    return {
        "query": f"similarity:{payload.video_id}/{payload.frame_id:06d}",
        "search_mode": "similarity", "latency_ms": latency_ms, "count": len(results),
        "source": dict_to_result_FAST(source_dict_data, paths.keyframes_root, paths.backend_dir), "results": results,
    }
