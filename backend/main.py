from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import yaml
from deep_translator import GoogleTranslator
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import asyncio
from fastapi.concurrency import run_in_threadpool  # Hỗ trợ non-blocking threadpool

from src.index.retrieval_system import FaissRetrievalSystem, SearchMode
from src.ui import get_timestamp_from_row

from fastapi import UploadFile, File
from tempfile import NamedTemporaryFile
from faster_whisper import WhisperModel

import os 
from dotenv import load_dotenv

# Thư viện phục vụ Profiling điểm nghẽn
import cProfile
import pstats
import io

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    print("[ENV] HF_TOKEN loaded")

BACKEND_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BACKEND_DIR / "configs" / "app.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_path_text(path_value: str | Path) -> str:
    return str(path_value or "").replace("\\", "/")


def resolve_backend_path(path_value: str | Path) -> Path:
    path = Path(normalize_path_text(path_value))

    if path.is_absolute():
        return path

    return BACKEND_DIR / path


CFG = load_yaml(CONFIG_PATH)

# CHUẨN ĐƯỜNG DẪN: Đọc cấu hình điều khiển profile từ CFG["debug"]["profile"]
SHOULD_PROFILE = bool(CFG.get("debug", {}).get("profile", False))

FAISS_INDEX_PATH = resolve_backend_path(CFG["faiss"]["index_path"])
METADATA_PATH = resolve_backend_path(CFG["faiss"]["metadata_path"])
VECTOR_CACHE_PATH = resolve_backend_path(
    CFG.get("faiss", {}).get(
        "vector_cache_path",
        "data/database/faiss_hnsw_clip_vitl16_siglip_256/vectors_fp16.npy",
    )
)

KEYFRAMES_ROOT = resolve_backend_path(CFG["paths"]["keyframes_root"])
VIDEOS_ROOT = resolve_backend_path(CFG["paths"]["videos_root"])
MAP_KEYFRAME_ROOT = resolve_backend_path(CFG["paths"]["map_keyframe_path"])


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
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(30)
    profile_content = s.getvalue()

    report_text = (
        f"\n" + "="*40 + " BOTTLENECK PROFILE REPORT " + "="*40 + "\n"
        f"[REQ] {request.method} {request.url.path} | Total Latency: {latency_ms}ms\n"
        f"{'-'*107}\n"
        f"{profile_content}"
        f"{'='*107}\n"
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


retrieval_system = FaissRetrievalSystem(
    index_path=str(FAISS_INDEX_PATH),
    metadata_path=str(METADATA_PATH),
    model_name=CFG["model"]["name"],
    pretrained=CFG["model"]["pretrained"],
    device=CFG["model"].get("device", "auto"),
    precision=CFG["model"].get("precision", "fp32"),
    normalize=bool(CFG["model"].get("normalize", True)),
    ef_search=int(CFG.get("faiss", {}).get("ef_search", 64)),
    faiss_threads=CFG.get("faiss", {}).get("threads"),
    cache_index_vectors=CFG.get("faiss", {}).get("cache_index_vectors", None),
    vector_cache_mode=CFG.get("faiss", {}).get("vector_cache_mode", None),
    vector_cache_dtype=CFG.get("faiss", {}).get("vector_cache_dtype", "float32"),
    vector_cache_path=str(VECTOR_CACHE_PATH),
    allow_npy_fallback=bool(CFG.get("faiss", {}).get("allow_npy_fallback", False)),
    compile_model=bool(CFG.get("model", {}).get("compile", False)),
)

speech_model = None

def get_speech_model():
    global speech_model
    if speech_model is None:
        speech_cfg = CFG.get("speech", {})
        speech_model = WhisperModel(
            speech_cfg.get("model_size", "base"),
            device=speech_cfg.get("device", "cpu"),
            compute_type=speech_cfg.get("compute_type", "int8"),
        )
    return speech_model

class SearchRequest(BaseModel):
    query: str
    top_k: int | None = None
    candidate_multiplier: int | None = None
    use_split: bool | None = None
    use_translate: bool | None = None
    search_mode: SearchMode | None = "semantic"
    duration_limit: float | None = -1


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


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    try:
        if pd.isna(value): return None
    except Exception:
        pass
    return value


def format_keyframe_id_from_dict(item: dict[str, Any]) -> str:
    value = item.get("keyframe_id", None)
    if value is not None:
        value_text = str(value)
        if value_text.isdigit():
            return value_text.zfill(6)
        return value_text

    if "keyframe_id_int" in item:
        return f"{safe_int(item.get('keyframe_id_int'), 0):06d}"
    if "frame_idx" in item:
        return f"{safe_int(item.get('frame_idx'), 0):06d}"
    return "000000"


def normalize_rel_path(path_value: str | Path) -> str:
    return normalize_path_text(path_value).lstrip("/")


# 0ms I/O: Cắt URL logic siêu tốc
def resolve_keyframe_path_from_dict(item: dict[str, Any]) -> str:
    keyframe_path = str(item.get("keyframe_path", "") or "")
    if keyframe_path:
        return keyframe_path.replace("\\", "/")

    dataset = str(item.get("dataset", "") or "")
    video_id = str(item.get("video_id", "") or "")
    frame_id_text = format_keyframe_id_from_dict(item)

    if dataset and video_id:
        return f"data/processed/keyframes/{dataset}/{video_id}/{frame_id_text}.jpg"
    return f"data/processed/keyframes/{video_id}/{frame_id_text}.jpg"


def find_video_path_from_dict(item: dict[str, Any]) -> str:
    video_path = str(item.get("video_path", "") or "")
    if video_path:
        return video_path.replace("\\", "/")

    dataset = str(item.get("dataset", "") or "")
    video_id = str(item.get("video_id", "") or "")

    if not video_id:
        return ""

    if dataset:
        return f"data/processed/videos/{dataset}/{video_id}.mp4"
    return f"data/processed/videos/{video_id}.mp4"


def translate_query_if_needed(query: str, use_translate: bool) -> str:
    if not use_translate:
        return query

    translate_cfg = CFG.get("translate", {})
    source = translate_cfg.get("source", "vi")
    target = translate_cfg.get("target", "en")

    return GoogleTranslator(source=source, target=target).translate(query)


def serialize_matched_sequence(sequence: Any) -> list[dict[str, Any]]:
    if not isinstance(sequence, list):
        return []

    rows: list[dict[str, Any]] = []

    for idx, item in enumerate(sequence):
        if not isinstance(item, dict):
            continue

        raw_k_path = resolve_keyframe_path_from_dict(item)
        if "keyframes/" in raw_k_path:
            image_rel_path = raw_k_path.split("keyframes/", 1)[1]
        else:
            dataset = str(item.get("dataset", "") or "")
            video_id = str(item.get("video_id", "") or "")
            frame_id_text = format_keyframe_id_from_dict(item)
            image_rel_path = f"{dataset}/{video_id}/{frame_id_text}.jpg" if dataset else f"{video_id}/{frame_id_text}.jpg"

        frame_id_text = format_keyframe_id_from_dict(item)
        timestamp = safe_float(item.get("timestamp_sec", item.get("timestamp", 0.0)), 0.0)
        score = safe_float(item.get("score", item.get("candidate_score", 0.0)), 0.0)

        rows.append(
            {
                "video_id": str(json_safe(item.get("video_id", "")) or ""),
                "source_name": str(json_safe(item.get("source_name", "")) or ""),
                "keyframe_id": frame_id_text,
                "frame_id": safe_int(frame_id_text, 0),
                "frame_idx": safe_int(item.get("frame_idx", 0), 0),
                "timestamp_sec": timestamp,
                "fps": safe_float(item.get("fps", 0), 0.0),
                "keyframe_path": raw_k_path,
                "image_url": f"/static/keyframes/{image_rel_path}" if image_rel_path else "",
                "image_rel_path": image_rel_path,
                "sub_query_idx": safe_int(item.get("sub_query_idx", idx), idx),
                "sub_query": str(json_safe(item.get("sub_query", "")) or ""),
                "score": score,
                "candidate_score": safe_float(item.get("candidate_score", score), score),
                "candidate_rank": safe_int(item.get("candidate_rank", 0), 0),
            }
        )

    return rows


# TỐI ƯU C-LEVEL: Hàm xử lý Mảng -> Dict siêu tốc độ
def dict_to_result_FAST(item: dict[str, Any]) -> dict[str, Any]:
    dataset = str(item.get("dataset", "") or "")
    video_id = str(item.get("video_id", "unknown_video") or "unknown_video")

    frame_id_text = format_keyframe_id_from_dict(item)
    frame_id_number = safe_int(frame_id_text, 0)

    # Trích xuất chuẩn xác đuôi tương đối (Xử lý dứt điểm lỗi 404)
    raw_k_path = resolve_keyframe_path_from_dict(item)
    if "keyframes/" in raw_k_path:
        image_rel_path = raw_k_path.split("keyframes/", 1)[1]
    else:
        image_rel_path = f"{dataset}/{video_id}/{frame_id_text}.jpg" if dataset else f"{video_id}/{frame_id_text}.jpg"
    image_url = f"/static/keyframes/{image_rel_path}"

    raw_v_path = find_video_path_from_dict(item)
    if "videos/" in raw_v_path:
        video_rel_path = raw_v_path.split("videos/", 1)[1]
    else:
        video_rel_path = f"{dataset}/{video_id}.mp4" if dataset else f"{video_id}.mp4"
    video_url = f"/static/videos/{video_rel_path}"

    map_rel_path = f"{dataset}/{video_id}.csv" if dataset else f"{video_id}.csv"
    map_url = f"/static/map-keyframes/{map_rel_path}"

    timestamp = safe_float(item.get("timestamp_sec", item.get("timestamp", 0.0)), 0.0)
    if timestamp == 0.0 and "pts_time" in item:
        timestamp = safe_float(item.get("pts_time"), 0.0)

    retrieval_score = safe_float(
        item.get(
            "retrieval_score",
            item.get("alignment_score", item.get("avg_score", item.get("score", 0.0))),
        ),
        0.0,
    )

    avg_score = safe_float(item.get("avg_score", retrieval_score), retrieval_score)
    matched_sequence = serialize_matched_sequence(item.get("matched_sequence", []))

    temporal_start = safe_float(item.get("temporal_start_time", timestamp), timestamp)
    temporal_end = safe_float(item.get("temporal_end_time", timestamp), timestamp)
    temporal_duration = safe_float(
        item.get("temporal_duration_sec", max(0.0, temporal_end - temporal_start)),
        max(0.0, temporal_end - temporal_start),
    )

    return {
        "id": f"{video_id}_{frame_id_text}",
        "video_id": video_id,
        "frame_id": frame_id_number,
        "frame_name": f"{frame_id_text}.jpg",
        "path": f"{video_id}/{frame_id_text}",
        "keyframe_path": raw_k_path,
        "image_url": image_url,
        "image_rel_path": image_rel_path,
        "video_url": video_url,
        "video_rel_path": video_rel_path,
        "map_url": map_url,
        "map_rel_path": map_rel_path,
        "timestamp": timestamp,
        "similarity": retrieval_score,
        "caption": str(item.get("caption", "") or ""),
        "rank": safe_int(item.get("display_rank", item.get("rank", 0)), 0),
        "matched_sequence": matched_sequence,
        "temporal": {
            "video_score": safe_float(item.get("video_score", 0), 0.0),
            "start_time": temporal_start,
            "end_time": temporal_end,
            "duration_sec": temporal_duration,
            "avg_score": avg_score,
        },
        "raw": item,
    }


def find_metadata_row(video_id: str, keyframe_id: int) -> pd.Series:
    row_idx = retrieval_system._row_by_video_frame.get((str(video_id), int(keyframe_id)))
    if row_idx is None:
        raise HTTPException(
            status_code=404,
            detail=f"Frame not found: {video_id}/{keyframe_id}",
        )
    return retrieval_system.metadata.iloc[int(row_idx)]


DRES_HEADERS = {"ngrok-skip-browser-warning": "true"}

def clean_external_url(url: str) -> str:
    cleaned = str(url or "").strip().rstrip("/")
    if not cleaned: raise HTTPException(status_code=400, detail="Missing DRES URL")
    if not (cleaned.startswith("http://") or cleaned.startswith("https://")):
        raise HTTPException(status_code=400, detail="DRES URL must start with http:// or https://")
    return cleaned

def fetch_dres_evaluations(dres_url: str, session_id: str) -> list[dict]:
    try:
        res = requests.get(f"{dres_url}/api/v2/client/evaluation/list", params={"session": session_id}, headers=DRES_HEADERS, timeout=10)
        if res.ok: return res.json()
        return []
    except Exception: return []

def pick_active_evaluation_id(evaluations: list) -> str | None:
    if not evaluations or not isinstance(evaluations, list): return None
    for ev in evaluations:
        if ev.get("status") == "ACTIVE": return ev.get("id")
    return evaluations[0].get("id")

def normalize_dres_verdict(response: requests.Response) -> dict[str, Any]:
    try: data = response.json()
    except Exception: data = {"raw": response.text or ""}

    raw_text = (response.text or "").lower()
    if response.status_code == 412: return {"status": "wrong", "message": "Wrong Answer", "data": data}
    if not response.ok: return {"status": "error", "message": data.get("description", f"HTTP Error {response.status_code}"), "data": data}
    if response.status_code == 202: return {"status": "pending", "message": "Submitted, waiting for verdict", "data": data}

    verdict = str(data.get("submission", "")).upper()
    if "CORRECT" in verdict or "CORRECT" in raw_text: return {"status": "correct", "message": "Correct!", "data": data}
    if "WRONG" in verdict or "WRONG" in raw_text: return {"status": "wrong", "message": "Wrong Answer", "data": data}
    return {"status": "pending", "message": "Submitted", "data": data}

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "backend_dir": str(BACKEND_DIR),
        "config_path": str(CONFIG_PATH),
        "faiss_index_path": str(FAISS_INDEX_PATH),
        "metadata_path": str(METADATA_PATH),
        "vector_cache_path": str(VECTOR_CACHE_PATH),
        "vector_cache_exists": VECTOR_CACHE_PATH.exists(),
        "vector_cache": retrieval_system.cache_info,
        "keyframes_root": str(KEYFRAMES_ROOT),
        "videos_root": str(VIDEOS_ROOT),
        "map_keyframe_path": str(MAP_KEYFRAME_ROOT),
        "keyframes_root_exists": KEYFRAMES_ROOT.exists(),
        "videos_root_exists": VIDEOS_ROOT.exists(),
        "map_keyframe_root_exists": MAP_KEYFRAME_ROOT.exists(),
        "faiss_index_exists": FAISS_INDEX_PATH.exists(),
        "metadata_exists": METADATA_PATH.exists(),
        "model": CFG["model"],
    }

@app.get("/api/config")
def get_public_config():
    return {
        "search": {
            "default_top_k": int(CFG["search"].get("default_top_k", 20)),
            "max_top_k": int(CFG["search"].get("max_top_k", 200)),
            "candidate_multiplier": int(CFG["search"].get("candidate_multiplier", 1)),
            "available_modes": ["semantic", "temporal", "ocr", "asr", "auto"],
            "default_search_mode": "semantic",
            "default_duration_limit": -1,
        },
        "ui": {
            "surrounding_radius": int(CFG["ui"].get("surrounding_radius", 5)),
            "max_surrounding_radius": int(CFG["ui"].get("max_surrounding_radius", 10)),
        },
        "translate": {
            "enabled_default": bool(CFG.get("translate", {}).get("enabled_default", False)),
            "source": CFG.get("translate", {}).get("source", "vi"),
            "target": CFG.get("translate", {}).get("target", "en"),
        },
        "model": {
            "name": CFG["model"]["name"],
            "pretrained": CFG["model"]["pretrained"],
            "device": CFG["model"].get("device", "auto"),
            "precision": CFG["model"].get("precision", "fp32"),
            "normalize": bool(CFG["model"].get("normalize", True)),
        },
        "faiss": {
            "ef_search": int(CFG.get("faiss", {}).get("ef_search", 64)),
            "threads": CFG.get("faiss", {}).get("threads"),
            "vector_cache_mode": CFG.get("faiss", {}).get("vector_cache_mode", None),
            "vector_cache_dtype": CFG.get("faiss", {}).get("vector_cache_dtype", "float32"),
            "vector_cache_path": str(VECTOR_CACHE_PATH),
            "vector_cache_available": retrieval_system.cache_info.get("available", False),
            "allow_npy_fallback": bool(CFG.get("faiss", {}).get("allow_npy_fallback", False)),
        },
    }

# Lưu ý: Các hàm DRES dùng requests nên để def thông thường, FastAPI sẽ tự động nạp nó vào worker thread phụ để không block Event Loop
@app.post("/api/dres/login")
def dres_login(payload: DresLoginRequest):
    dres_url = clean_external_url(payload.dres_url)
    if not payload.username.strip() or not payload.password: raise HTTPException(status_code=400, detail="Missing username or password")
    session = requests.Session()
    session.headers.update(DRES_HEADERS)
    try:
        login_res = session.post(f"{dres_url}/api/v2/login", json={"username": payload.username, "password": payload.password}, timeout=15)
    except requests.RequestException as e: raise HTTPException(status_code=502, detail=f"DRES login connection failed: {e}")

    if not login_res.ok:
        try: err_desc = login_res.json().get("description", "Login failed")
        except Exception: err_desc = login_res.text or "Login failed"
        raise HTTPException(status_code=login_res.status_code, detail=err_desc)

    try: sess_res = session.get(f"{dres_url}/api/v2/user/session", timeout=15)
    except requests.RequestException as e: raise HTTPException(status_code=502, detail=f"DRES session fetch failed: {e}")
    if not sess_res.ok: raise HTTPException(status_code=sess_res.status_code, detail="Cannot fetch DRES session")

    session_id = sess_res.text.strip().strip('"')
    evaluations = fetch_dres_evaluations(dres_url, session_id)
    evaluation_id = pick_active_evaluation_id(evaluations)

    return {"status": "ok", "session_id": session_id, "evaluation_id": evaluation_id, "evaluations": evaluations, "user": login_res.json() if login_res.text else {}}

@app.post("/api/dres/submit")
def dres_submit(payload: DresSubmitRequest):
    dres_url = clean_external_url(payload.dres_url)
    if not payload.session_id.strip(): raise HTTPException(status_code=400, detail="Missing active session_id")

    evaluation_id = payload.evaluation_id
    if not evaluation_id:
        evals = fetch_dres_evaluations(dres_url, payload.session_id)
        evaluation_id = pick_active_evaluation_id(evals)

    if not evaluation_id: raise HTTPException(status_code=400, detail="No active DRES evaluation found to submit")

    if payload.timestamp is not None and payload.timestamp >= 0:
        time_ms = int(round(payload.timestamp * 1000))
    else:
        time_ms = int(payload.frame_id)

    submit_payload = {"answerSets": [{"answers": [{"mediaItemName": str(payload.video_id).strip(), "start": time_ms, "end": time_ms, "text": None, "mediaItemCollectionName": None}]}]}
    submit_url = f"{dres_url}/api/v2/submit/{evaluation_id}"
    params = {"session": payload.session_id}

    try:
        res = requests.post(submit_url, params=params, json=submit_payload, timeout=15)
        return normalize_dres_verdict(res)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Connection to DRES submit failed: {e}")

@app.get("/api/frame-info")
async def get_frame_info(video_id: str, keyframe_id: int):
    # Dùng threadpool cho các truy vấn pandas nhanh
    def _fetch():
        row = find_metadata_row(video_id, keyframe_id)
        return dict_to_result_FAST(row.to_dict())
    return await run_in_threadpool(_fetch)


# ====================================================================
# KIẾN TRÚC TỐI ƯU CỐT LÕI: Non-blocking Threadpool cho mọi tác vụ CPU
# ====================================================================

@app.post("/api/search")
async def search_api(payload: SearchRequest):
    original_query = payload.query.strip()
    if not original_query:
        raise HTTPException(status_code=400, detail="Query is empty")

    max_top_k = int(CFG["search"].get("max_top_k", 200))
    default_top_k = int(CFG["search"].get("default_top_k", 20))

    top_k = payload.top_k or default_top_k
    top_k = max(1, min(int(top_k), max_top_k))

    candidate_multiplier = payload.candidate_multiplier or int(CFG["search"].get("candidate_multiplier", 5))
    candidate_multiplier = max(1, int(candidate_multiplier))

    use_split = True if payload.use_split is None else bool(payload.use_split)

    use_translate = (
        bool(CFG.get("translate", {}).get("enabled_default", False))
        if payload.use_translate is None else bool(payload.use_translate)
    )

    mode = payload.search_mode
    duration_limit = -1.0 if payload.duration_limit is None else float(payload.duration_limit)
    if duration_limit == 0: duration_limit = -1.0

    start = time.perf_counter()

    try:
        # Bất đồng bộ hóa dịch thuật Google API (Mạng I/O)
        search_query = await run_in_threadpool(
            translate_query_if_needed,
            query=original_query,
            use_translate=use_translate,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Translate failed: {type(exc).__name__}: {exc}")

    try:
        # Bất đồng bộ hóa Model Vector Inference & Faiss Tìm kiếm nặng (CPU/GPU I/O)
        results_df, query_plan = await run_in_threadpool(
            retrieval_system.run_search,
            query=search_query,
            mode=mode,
            use_split=use_split,
            top_k=top_k,
            candidate_multiplier=candidate_multiplier,
            duration_limit=duration_limit,
        )
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Search failed: {type(exc).__name__}: {exc}")

    latency_ms = round((time.perf_counter() - start) * 1000)
    candidate_k = max(top_k * candidate_multiplier, top_k)

    response_base = {
        "original_query": original_query, "query": search_query, "translated_query": search_query if use_translate else None,
        "use_translate": use_translate, "use_split": use_split, "mode": mode, "search_mode": mode,
        "duration_limit": duration_limit, "top_k": top_k, "candidate_multiplier": candidate_multiplier,
        "candidate_k": candidate_k, "latency_ms": latency_ms,
        "events": query_plan.events, "event_queries": query_plan.event_queries, "sub_queries": query_plan.flat_queries,
    }

    if results_df.empty:
        return {**response_base, "count": 0, "results": []}

    try:
        # Chạy ép kiểu DataFrame sang C-Dict ngoài luồng Event Loop để tránh overhead bộ nhớ
        def _serialize_results():
            cols_to_keep = [c for c in ["video_id", "keyframe_id", "keyframe_id_int", "frame_idx", "dataset", "keyframe_path", "video_path", "timestamp_sec", "timestamp", "score", "retrieval_score", "avg_score", "caption", "matched_sequence", "temporal_start_time", "temporal_end_time", "temporal_duration_sec", "video_score"] if c in results_df.columns]
            optimized_df = results_df[cols_to_keep]
            records = optimized_df.to_dict(orient="records")
            return [dict_to_result_FAST(rec) for rec in records]
        
        results = await run_in_threadpool(_serialize_results)
        
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Serialize results failed: {type(exc).__name__}: {exc}")

    return {**response_base, "count": len(results), "results": results}


@app.get("/api/surrounding-frames")
async def get_surrounding_frames(video_id: str, keyframe_id: int, radius: int = 10):
    def _fetch_surround():
        metadata_df = retrieval_system.metadata
        radius_val = max(1, min(int(radius), 50))
        target_keyframe_id = int(keyframe_id)

        row_ids = retrieval_system._rows_by_video.get(str(video_id))
        if row_ids is None or len(row_ids) == 0:
            raise HTTPException(status_code=404, detail=f"Video not found: {video_id}")
        video_df = metadata_df.iloc[row_ids].copy()

        if video_df.empty:
            raise HTTPException(status_code=404, detail=f"Video not found: {video_id}")

        if "keyframe_id_int" in video_df.columns:
            video_df["_keyframe_sort"] = video_df["keyframe_id_int"].apply(lambda x: safe_int(x, 0))
        elif "keyframe_id" in video_df.columns:
            video_df["_keyframe_sort"] = video_df["keyframe_id"].apply(lambda x: safe_int(x, 0))
        elif "frame_idx" in video_df.columns:
            video_df["_keyframe_sort"] = video_df["frame_idx"].apply(lambda x: safe_int(x, 0))
        else:
            raise HTTPException(status_code=500, detail="metadata missing keyframe_id column")

        video_df = video_df.sort_values("_keyframe_sort").reset_index(drop=True)
        center_matches = video_df[video_df["_keyframe_sort"].astype(int) == target_keyframe_id]

        if center_matches.empty:
            raise HTTPException(status_code=404, detail=f"Frame not found: {video_id}/{keyframe_id}")

        center_pos = int(center_matches.index[0])
        start_pos = max(0, center_pos - radius_val)
        end_pos = min(len(video_df), center_pos + radius_val + 1)

        surrounding_df = video_df.iloc[start_pos:end_pos].copy()
        
        records = surrounding_df.to_dict(orient="records")
        frames = []

        for rec in records:
            item = dict_to_result_FAST(rec)
            item["is_surround_center"] = safe_int(item.get("frame_id"), -1) == target_keyframe_id
            item["surround_offset"] = safe_int(item.get("frame_id"), 0) - target_keyframe_id
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
        max_top_k = int(CFG["search"].get("max_top_k", 200))
        default_top_k = int(CFG["search"].get("default_top_k", 20))

        top_k = payload.top_k or default_top_k
        top_k = max(1, min(int(top_k), max_top_k))

        row = find_metadata_row(payload.video_id, payload.frame_id)
        source_dict = row.to_dict()
        
        image_path = resolve_keyframe_path_from_dict(source_dict)

        results_df = retrieval_system.similarity_search_by_image(
            image_path=Path(image_path),
            top_k=top_k,
        )
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
            "search_mode": "similarity",
            "latency_ms": latency_ms,
            "count": 0,
            "results": [],
        }

    # Ép kiểu siêu tốc
    def _parse():
        return [dict_to_result_FAST(rec) for rec in df_results.to_dict(orient="records")]
    
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