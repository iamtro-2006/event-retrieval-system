from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import requests
from deep_translator import GoogleTranslator
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import asyncio
from fastapi.concurrency import run_in_threadpool
from tempfile import NamedTemporaryFile
from faster_whisper import WhisperModel

import os
from dotenv import load_dotenv

import cProfile
import pstats
import io

from src.api.models import (
    DresLoginRequest,
    DresSubmitRequest,
    SearchRequest,
    SimilaritySearchRequest,
)
from src.api.serialization import (
    resolve_keyframe_path,
    safe_int,
    serialize as dict_to_result_FAST,
)
from src.api.dres import DresClient
from src.config.retrieval_config import RetrievalConfig, load_retrieval_config, to_dict
from src.index.retrieval_backend import create_retrieval_backend
from src.index.query_planning import SearchMode

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    print("[ENV] HF_TOKEN loaded")

BACKEND_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BACKEND_DIR / "configs" / "app.yaml"


def normalize_path_text(path_value: str | Path) -> str:
    return str(path_value or "").replace("\\", "/")


def resolve_backend_path(path_value: str | Path) -> Path:
    path = Path(normalize_path_text(path_value))

    if path.is_absolute():
        return path

    return BACKEND_DIR / path


CFG: RetrievalConfig = load_retrieval_config(CONFIG_PATH)
CFG_DICT = to_dict(CFG)

SHOULD_PROFILE = CFG.debug.profile

BACKEND_NAME = CFG.backend

FAISS_INDEX_PATH = (
    resolve_backend_path(CFG.faiss.index_path) if BACKEND_NAME == "faiss" else None
)
METADATA_PATH = (
    resolve_backend_path(CFG.faiss.metadata_path) if BACKEND_NAME == "faiss" else None
)
VECTOR_CACHE_PATH = (
    resolve_backend_path(CFG.faiss.vector_cache_path or "")
    if BACKEND_NAME == "faiss"
    else None
)

KEYFRAMES_ROOT = resolve_backend_path(CFG.paths.keyframes_root)
VIDEOS_ROOT = resolve_backend_path(CFG.paths.videos_root)
MAP_KEYFRAME_ROOT = resolve_backend_path(CFG.paths.map_keyframe_path)


app = FastAPI(title="Event Retrieval API", version="1.0.0")


# --- MIDDLEWARE PROFILING ---
@app.middleware("http")
async def profile_and_bottleneck_tracker(request: Request, call_next):
    if not SHOULD_PROFILE or not request.url.path.startswith("/api/"):
        return await call_next(request)

    pr = cProfile.Profile()
    pr.enable()

    start_time = time.perf_counter()
    response = await call_next(request)
    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(30)
    profile_content = s.getvalue()

    report_text = (
        "\n" + "=" * 40 + " BOTTLENECK PROFILE REPORT " + "=" * 40 + "\n"
        f"[REQ] {request.method} {request.url.path} | Total Latency: {latency_ms}ms\n"
        f"{'-' * 107}\n"
        f"{profile_content}"
        f"{'=' * 107}\n"
    )

    print(report_text)

    # Đẩy việc ghi log file ra một thread phụ để luồng chính không bị kẹt khi đang ghi đĩa
    def _write_log():
        log_file = BACKEND_DIR / "search_profile_log.txt"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(report_text)

    asyncio.create_task(asyncio.to_thread(_write_log))

    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

if KEYFRAMES_ROOT.exists():
    app.mount(
        "/static/keyframes",
        StaticFiles(directory=str(KEYFRAMES_ROOT)),
        name="keyframes",
    )

if VIDEOS_ROOT.exists():
    app.mount(
        "/static/videos",
        StaticFiles(directory=str(VIDEOS_ROOT)),
        name="videos",
    )

if MAP_KEYFRAME_ROOT.exists():
    app.mount(
        "/static/map-keyframes",
        StaticFiles(directory=str(MAP_KEYFRAME_ROOT)),
        name="map_keyframes",
    )


retrieval_system = create_retrieval_backend(CFG_DICT)

speech_model = None


def get_speech_model():
    global speech_model
    if speech_model is None:
        speech_model = WhisperModel(
            CFG.speech.model_size,
            device=CFG.speech.device,
            compute_type=CFG.speech.compute_type,
        )
    return speech_model


def find_metadata_row(video_id: str, keyframe_id: int) -> dict:
    row = retrieval_system.get_frame(str(video_id), int(keyframe_id))
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Frame not found: {video_id}/{keyframe_id}",
        )
    return row


def translate_query_if_needed(query: str, use_translate: bool) -> str:
    if not use_translate:
        return query

    return GoogleTranslator(
        source=CFG.translate.source, target=CFG.translate.target
    ).translate(query)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "backend": BACKEND_NAME,
        "backend_dir": str(BACKEND_DIR),
        "config_path": str(CONFIG_PATH),
        "backend_info": retrieval_system.cache_info,
        "keyframes_root": str(KEYFRAMES_ROOT),
        "videos_root": str(VIDEOS_ROOT),
        "map_keyframe_path": str(MAP_KEYFRAME_ROOT),
        "keyframes_root_exists": KEYFRAMES_ROOT.exists(),
        "videos_root_exists": VIDEOS_ROOT.exists(),
        "map_keyframe_root_exists": MAP_KEYFRAME_ROOT.exists(),
        "model": CFG_DICT["model"],
    }


@app.get("/api/config")
def get_public_config():
    return {
        "backend": BACKEND_NAME,
        "search": {
            "default_top_k": CFG.search.default_top_k,
            "max_top_k": CFG.search.max_top_k,
            "candidate_multiplier": CFG.search.candidate_multiplier,
            "available_modes": ["semantic", "temporal", "ocr", "asr", "auto"],
            "default_search_mode": "semantic",
            "default_duration_limit": -1,
        },
        "ui": {
            "surrounding_radius": CFG.ui.surrounding_radius,
            "max_surrounding_radius": CFG.ui.max_surrounding_radius,
        },
        "translate": {
            "enabled_default": CFG.translate.enabled_default,
            "source": CFG.translate.source,
            "target": CFG.translate.target,
        },
        "model": {
            "name": CFG.model.name,
            "pretrained": CFG.model.pretrained,
            "device": CFG.model.device,
            "precision": CFG.model.precision,
            "normalize": CFG.model.normalize,
        },
        "milvus": {
            "host": CFG.milvus.host,
            "port": CFG.milvus.port,
            "collection_name": CFG.milvus.collection_name,
            "consistency_level": CFG.milvus.consistency_level,
            "metric_type": CFG.milvus.search_params.metric_type,
            "default_ef": CFG.milvus.search_params.ef,
            "num_entities": retrieval_system.num_entities,
        },
    }


# Lưu ý: Các hàm DRES dùng requests nên để def thông thường, FastAPI sẽ tự động nạp nó vào worker thread phụ để không block Event Loop
@app.post("/api/dres/login")
def dres_login(payload: DresLoginRequest):
    if not payload.username.strip() or not payload.password:
        raise HTTPException(status_code=400, detail="Missing username or password")
    try:
        client = DresClient(payload.dres_url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        return client.login(payload.username, payload.password)
    except requests.RequestException as e:
        raise HTTPException(
            status_code=502, detail=f"DRES login connection failed: {e}"
        )
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/api/dres/submit")
def dres_submit(payload: DresSubmitRequest):
    try:
        client = DresClient(payload.dres_url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        return client.submit(
            session_id=payload.session_id,
            evaluation_id=payload.evaluation_id,
            video_id=payload.video_id,
            frame_id=payload.frame_id,
            timestamp=payload.timestamp,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except requests.RequestException as e:
        raise HTTPException(
            status_code=502, detail=f"Connection to DRES submit failed: {e}"
        )


@app.get("/api/frame-info")
async def get_frame_info(video_id: str, keyframe_id: int):
    def _fetch():
        row = find_metadata_row(video_id, keyframe_id)
        return dict_to_result_FAST(row)

    return await run_in_threadpool(_fetch)


# ====================================================================
# KIẾN TRÚC TỐI ƯU CỐT LÕI: Non-blocking Threadpool cho mọi tác vụ CPU
# ====================================================================


@app.post("/api/search")
async def search_api(payload: SearchRequest):
    original_query = payload.query.strip()
    if not original_query:
        raise HTTPException(status_code=400, detail="Query is empty")

    max_top_k = CFG.search.max_top_k
    default_top_k = CFG.search.default_top_k

    top_k = payload.top_k or default_top_k
    top_k = max(1, min(int(top_k), max_top_k))

    candidate_multiplier = (
        payload.candidate_multiplier or CFG.search.candidate_multiplier
    )
    candidate_multiplier = max(1, int(candidate_multiplier))

    use_split = True if payload.use_split is None else bool(payload.use_split)

    use_translate = (
        CFG.translate.enabled_default
        if payload.use_translate is None
        else bool(payload.use_translate)
    )

    mode = payload.search_mode or "semantic"
    duration_limit = (
        -1.0 if payload.duration_limit is None else float(payload.duration_limit)
    )
    search_ef = payload.search_ef
    if duration_limit == 0:
        duration_limit = -1.0

    start = time.perf_counter()

    try:
        # Bất đồng bộ hóa dịch thuật Google API (Mạng I/O)
        search_query = await run_in_threadpool(
            translate_query_if_needed,
            query=original_query,
            use_translate=use_translate,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Translate failed: {type(exc).__name__}: {exc}"
        )

    try:
        results_df, query_plan = await run_in_threadpool(
            retrieval_system.run_search,
            query=search_query,
            mode=mode,
            use_split=use_split,
            top_k=top_k,
            candidate_multiplier=candidate_multiplier,
            duration_limit=duration_limit,
            search_ef=search_ef,
        )
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Search failed: {type(exc).__name__}: {exc}"
        )

    latency_ms = round((time.perf_counter() - start) * 1000)
    candidate_k = max(top_k * candidate_multiplier, top_k)

    response_base = {
        "original_query": original_query,
        "query": search_query,
        "translated_query": search_query if use_translate else None,
        "use_translate": use_translate,
        "use_split": use_split,
        "mode": mode,
        "search_mode": mode,
        "duration_limit": duration_limit,
        "top_k": top_k,
        "candidate_multiplier": candidate_multiplier,
        "candidate_k": candidate_k,
        "latency_ms": latency_ms,
        "events": query_plan.events,
        "event_queries": query_plan.event_queries,
        "sub_queries": query_plan.flat_queries,
    }

    if results_df.empty:
        return {**response_base, "count": 0, "results": []}

    try:
        # Chạy ép kiểu DataFrame sang C-Dict ngoài luồng Event Loop để tránh overhead bộ nhớ
        def _serialize_results():
            cols_to_keep = [
                c
                for c in [
                    "video_id",
                    "keyframe_id",
                    "keyframe_id_int",
                    "frame_idx",
                    "dataset",
                    "keyframe_path",
                    "video_path",
                    "timestamp_sec",
                    "timestamp",
                    "score",
                    "retrieval_score",
                    "avg_score",
                    "caption",
                    "matched_sequence",
                    "temporal_start_time",
                    "temporal_end_time",
                    "temporal_duration_sec",
                    "video_score",
                ]
                if c in results_df.columns
            ]
            optimized_df = results_df[cols_to_keep]
            records = optimized_df.to_dict(orient="records")
            return [dict_to_result_FAST(rec) for rec in records]

        results = await run_in_threadpool(_serialize_results)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Serialize results failed: {type(exc).__name__}: {exc}",
        )

    return {**response_base, "count": len(results), "results": results}


@app.get("/api/surrounding-frames")
async def get_surrounding_frames(video_id: str, keyframe_id: int, radius: int = 10):
    def _fetch_surround():
        radius_val = max(1, min(int(radius), 50))
        target_keyframe_id = int(keyframe_id)

        frames_data = retrieval_system.get_video_frames(str(video_id))
        if not frames_data:
            raise HTTPException(status_code=404, detail=f"Video not found: {video_id}")

        frames_data.sort(key=lambda r: safe_int(r.get("keyframe_id_int", 0), 0))

        center_idx = None
        for i, rec in enumerate(frames_data):
            if safe_int(rec.get("keyframe_id_int", 0), 0) == target_keyframe_id:
                center_idx = i
                break

        if center_idx is None:
            raise HTTPException(
                status_code=404, detail=f"Frame not found: {video_id}/{keyframe_id}"
            )

        start_pos = max(0, center_idx - radius_val)
        end_pos = min(len(frames_data), center_idx + radius_val + 1)
        surrounding = frames_data[start_pos:end_pos]

        frames = []
        for rec in surrounding:
            item = dict_to_result_FAST(rec)
            item["is_surround_center"] = (
                safe_int(item.get("frame_id"), -1) == target_keyframe_id
            )
            item["surround_offset"] = (
                safe_int(item.get("frame_id"), 0) - target_keyframe_id
            )
            frames.append(item)

        return radius_val, frames

    r_val, frames_result = await run_in_threadpool(_fetch_surround)
    return {
        "video_id": video_id,
        "center_frame_id": int(keyframe_id),
        "radius": r_val,
        "count": len(frames_result),
        "frames": frames_result,
    }


@app.post("/api/similarity-search")
async def similarity_search_api(payload: SimilaritySearchRequest):
    def _run_sim_search():
        max_top_k = CFG.search.max_top_k
        default_top_k = CFG.search.default_top_k

        top_k = payload.top_k or default_top_k
        top_k = max(1, min(int(top_k), max_top_k))

        row = find_metadata_row(payload.video_id, payload.frame_id)
        source_dict = row

        image_path = resolve_keyframe_path(source_dict)

        results_df = retrieval_system.similarity_search_by_image(
            image_path=Path(image_path),
            top_k=top_k,
        )
        return source_dict, results_df

    start = time.perf_counter()

    try:
        source_dict_data, df_results = await run_in_threadpool(_run_sim_search)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Similarity search failed: {type(exc).__name__}: {exc}",
        )

    latency_ms = round((time.perf_counter() - start) * 1000)

    if df_results.empty:
        return {
            "query": f"similarity:{payload.video_id}/{payload.frame_id:06d}",
            "search_mode": "similarity",
            "latency_ms": latency_ms,
            "count": 0,
            "results": [],
        }

    # Ép kiểu siêu tốc
    def _parse():
        return [
            dict_to_result_FAST(rec) for rec in df_results.to_dict(orient="records")
        ]

    results = await run_in_threadpool(_parse)

    return {
        "query": f"similarity:{payload.video_id}/{payload.frame_id:06d}",
        "search_mode": "similarity",
        "latency_ms": latency_ms,
        "count": len(results),
        "source": dict_to_result_FAST(source_dict_data),
        "results": results,
    }


@app.post("/api/speech/transcribe")
async def transcribe_speech(file: UploadFile = File(...)):
    suffix = Path(file.filename or "audio.webm").suffix or ".webm"

    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # Bọc model AI nặng vào Worker Thread
        def _run_whisper():
            return get_speech_model().transcribe(
                tmp_path,
                beam_size=1,
                language="vi",
                vad_filter=True,
            )

        segments, info = await run_in_threadpool(_run_whisper)
        text = " ".join(seg.text.strip() for seg in segments).strip()

        return {
            "text": text,
            "language": info.language,
            "duration": info.duration,
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)
