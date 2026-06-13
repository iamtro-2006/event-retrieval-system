from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import requests

import numpy as np
import pandas as pd
import yaml
from deep_translator import GoogleTranslator
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from src.index.retrieval_system import FaissRetrievalSystem
from src.ui import (
    split_query,
    rerank_multi_query,
    get_timestamp_from_row,
)


# ============================================================
# 1. Directory + Config
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BACKEND_DIR / "configs" / "app.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_backend_path(path_value: str) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return BACKEND_DIR / path


CFG = load_yaml(CONFIG_PATH)

FAISS_INDEX_PATH = resolve_backend_path(CFG["faiss"]["index_path"])
METADATA_PATH = resolve_backend_path(CFG["faiss"]["metadata_path"])

KEYFRAMES_ROOT = resolve_backend_path(CFG["paths"]["keyframes_root"])
VIDEOS_ROOT = resolve_backend_path(CFG["paths"]["videos_root"])
MAP_KEYFRAME_ROOT = resolve_backend_path(CFG["paths"]["map_keyframe_path"])


# ============================================================
# 2. FastAPI App
# ============================================================

app = FastAPI(
    title="Event Retrieval API",
    version="1.0.0",
)


@app.middleware("http")
async def normalize_double_slash(request: Request, call_next):
    path = request.url.path

    if path.startswith("//"):
        normalized_path = "/" + path.lstrip("/")
        normalized_url = request.url.replace(path=normalized_path)
        return RedirectResponse(str(normalized_url), status_code=307)

    return await call_next(request)

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


# ============================================================
# 3. Retrieval System
# ============================================================

retrieval_system = FaissRetrievalSystem(
    index_path=str(FAISS_INDEX_PATH),
    metadata_path=str(METADATA_PATH),
    model_name=CFG["model"]["name"],
    pretrained=CFG["model"]["pretrained"],
    device=CFG["model"].get("device", "auto"),
    precision=CFG["model"].get("precision", "fp32"),
    normalize=bool(CFG["model"].get("normalize", True)),
)


# ============================================================
# 4. Request Schema
# ============================================================

class SearchRequest(BaseModel):
    query: str
    top_k: int | None = None
    candidate_multiplier: int | None = None
    use_split: bool | None = None
    use_translate: bool | None = None
    search_mode: str | None = "semantic"
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



# ============================================================
# 5. Helper Functions
# ============================================================

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        if pd.isna(value):
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
        if pd.isna(value):
            return None
    except Exception:
        pass

    return value


def format_keyframe_id(row: pd.Series) -> str:
    if "keyframe_id" in row:
        value = str(row["keyframe_id"])

        if value.isdigit():
            return value.zfill(6)

        return value

    if "keyframe_id_int" in row:
        return f"{safe_int(row['keyframe_id_int'], 0):06d}"

    if "frame_idx" in row:
        return f"{safe_int(row['frame_idx'], 0):06d}"

    return "000000"


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


def make_static_keyframe_url(keyframe_path: str | Path) -> str:
    raw_path = str(keyframe_path or "")

    if not raw_path:
        return ""

    path = Path(raw_path)

    if not path.is_absolute():
        path = BACKEND_DIR / path

    try:
        rel = path.resolve().relative_to(KEYFRAMES_ROOT.resolve())
        return f"/static/keyframes/{rel.as_posix()}"
    except Exception:
        return ""


def find_video_path(row: pd.Series) -> Path | None:
    if "video_path" in row and isinstance(row["video_path"], str):
        video_path = Path(row["video_path"])

        if not video_path.is_absolute():
            video_path = BACKEND_DIR / video_path

        if video_path.exists():
            return video_path

    dataset = str(row.get("dataset", "") or "")
    video_id = str(row.get("video_id", "") or "")

    if not video_id:
        return None

    for ext in [".mp4", ".mkv", ".avi", ".mov", ".webm"]:
        candidates: list[Path] = []

        if dataset:
            candidates.append(VIDEOS_ROOT / dataset / f"{video_id}{ext}")

        candidates.append(VIDEOS_ROOT / f"{video_id}{ext}")

        for candidate in candidates:
            if candidate.exists():
                return candidate

    return None


def make_static_video_url(video_path: Path | None) -> str:
    if video_path is None:
        return "#"

    try:
        rel = video_path.resolve().relative_to(VIDEOS_ROOT.resolve())
        return f"/static/videos/{rel.as_posix()}"
    except Exception:
        return "#"


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

        keyframe_path = str(json_safe(item.get("keyframe_path", "")) or "")
        frame_id_text = format_keyframe_id_from_dict(item)
        timestamp = safe_float(
            item.get("timestamp_sec", item.get("timestamp", 0.0)),
            0.0,
        )
        score = safe_float(
            item.get("score", item.get("candidate_score", 0.0)),
            0.0,
        )

        rows.append(
            {
                "video_id": str(json_safe(item.get("video_id", "")) or ""),
                "source_name": str(json_safe(item.get("source_name", "")) or ""),
                "keyframe_id": frame_id_text,
                "frame_id": safe_int(frame_id_text, 0),
                "frame_idx": safe_int(item.get("frame_idx", 0), 0),
                "timestamp_sec": timestamp,
                "fps": safe_float(item.get("fps", 0), 0.0),
                "keyframe_path": keyframe_path,
                "image_url": make_static_keyframe_url(keyframe_path),
                "sub_query_idx": safe_int(item.get("sub_query_idx", idx), idx),
                "sub_query": str(json_safe(item.get("sub_query", "")) or ""),
                "score": score,
                "candidate_score": safe_float(item.get("candidate_score", score), score),
                "candidate_rank": safe_int(item.get("candidate_rank", 0), 0),
            }
        )

    return rows


def row_to_result(row: pd.Series) -> dict[str, Any]:
    video_id = str(row.get("video_id", "unknown_video") or "unknown_video")

    frame_id_text = format_keyframe_id(row)
    frame_id_number = safe_int(frame_id_text, 0)

    keyframe_path = str(row.get("keyframe_path", "") or "")
    image_url = make_static_keyframe_url(keyframe_path)

    timestamp = safe_float(get_timestamp_from_row(row), 0.0)

    video_path = find_video_path(row)
    video_url = make_static_video_url(video_path)

    retrieval_score = safe_float(
        row.get(
            "retrieval_score",
            row.get("alignment_score", row.get("avg_score", 0)),
        ),
        0.0,
    )

    avg_score = safe_float(
        row.get("avg_score", retrieval_score),
        retrieval_score,
    )

    matched_sequence = serialize_matched_sequence(
        row.get("matched_sequence", [])
    )

    temporal_start = safe_float(
        row.get("temporal_start_time", timestamp),
        timestamp,
    )
    temporal_end = safe_float(
        row.get("temporal_end_time", timestamp),
        timestamp,
    )
    temporal_duration = safe_float(
        row.get("temporal_duration_sec", max(0.0, temporal_end - temporal_start)),
        max(0.0, temporal_end - temporal_start),
    )

    return {
        "id": f"{video_id}_{frame_id_text}",
        "video_id": video_id,
        "frame_id": frame_id_number,
        "frame_name": f"{frame_id_text}.jpg",
        "path": f"{video_id}/{frame_id_text}",
        "keyframe_path": keyframe_path,
        "image_url": image_url,
        "video_url": video_url,
        "timestamp": timestamp,
        "similarity": retrieval_score,
        "caption": str(row.get("caption", "") or ""),
        "rank": safe_int(row.get("display_rank", row.get("rank", 0)), 0),

        "matched_sequence": matched_sequence,
        "temporal": {
            "video_score": safe_float(row.get("video_score", 0), 0.0),
            "start_time": temporal_start,
            "end_time": temporal_end,
            "duration_sec": temporal_duration,
            "avg_score": avg_score,
        },

        "raw": {
            "dataset": str(row.get("dataset", "") or ""),
            "source_name": str(row.get("source_name", "") or ""),
            "avg_score": avg_score,
            "retrieval_score": retrieval_score,
            "alignment_score": safe_float(
                row.get("alignment_score", retrieval_score),
                retrieval_score,
            ),
            "frame_idx": safe_int(
                row.get("frame_idx", frame_id_number),
                frame_id_number,
            ),
            "video_score": safe_float(row.get("video_score", 0), 0.0),
            "temporal_start_time": temporal_start,
            "temporal_end_time": temporal_end,
            "temporal_duration_sec": temporal_duration,
            "matched_sequence": matched_sequence,
        },
    }


def normalize_search_mode(search_mode: str | None) -> str:
    mode = (search_mode or "semantic").strip().lower()

    aliases = {
        "text": "semantic",
        "normal": "semantic",
        "semantic": "semantic",
        "temporal": "temporal",
    }

    if mode not in aliases:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported search_mode: {search_mode}",
        )

    return aliases[mode]


def run_search(
    sub_queries: list[str],
    top_k: int,
    candidate_multiplier: int,
    search_mode: str = "semantic",
    duration_limit: float = -1,
) -> pd.DataFrame:
    if not sub_queries:
        return pd.DataFrame()

    candidate_k = max(top_k * candidate_multiplier, top_k)

    if search_mode == "temporal":
        return retrieval_system.temporal_search(
            sub_queries=sub_queries[1:],
            top_k=top_k,
            candidate_k=candidate_k,
            duration_limit=duration_limit,
        )

    all_results = []

    for sub_query in sub_queries:
        df = retrieval_system.search(sub_query, top_k=candidate_k)

        if not df.empty:
            all_results.append(df)

    results = rerank_multi_query(all_results)

    if results.empty:
        return results

    if "alignment_score" in results.columns and "retrieval_score" not in results.columns:
        results["retrieval_score"] = results["alignment_score"]

    results = results.head(top_k).copy()
    results["display_rank"] = range(1, len(results) + 1)

    return results



# ============================================================
# 6. DRES Proxy Helpers
# ============================================================

DRES_HEADERS = {"ngrok-skip-browser-warning": "true"}


def clean_external_url(url: str) -> str:
    cleaned = str(url or "").strip().rstrip("/")
    if not cleaned:
        raise HTTPException(status_code=400, detail="Missing DRES URL")
    if not (cleaned.startswith("http://") or cleaned.startswith("https://")):
        raise HTTPException(
            status_code=400,
            detail="DRES URL must start with http:// or https://",
        )
    return cleaned


def fetch_dres_evaluations(dres_url: str, session_id: str) -> list[dict]:
    """Gọi endpoint GET /api/v2/client/evaluation/list để lấy danh sách trận đấu"""
    try:
        res = requests.get(
            f"{dres_url}/api/v2/client/evaluation/list",
            params={"session": session_id},
            headers=DRES_HEADERS,
            timeout=10,
        )
        if res.ok:
            return res.json()
        return []
    except Exception:
        return []


def pick_active_evaluation_id(evaluations: list) -> str | None:
    """Tự động tìm kiếm evaluation đang 'ACTIVE' từ danh sách trả về của DRES"""
    if not evaluations or not isinstance(evaluations, list):
        return None
    for ev in evaluations:
        # Check theo Schema ApiClientEvaluationInfo, trạng thái nằm trong trường 'status'
        if ev.get("status") == "ACTIVE":
            return ev.get("id")
    # Dự phòng nếu không thấy cụ thể trạng thái ACTIVE thì bốc đại ID đầu tiên
    return evaluations[0].get("id")


def normalize_dres_verdict(response: requests.Response) -> dict[str, Any]:
    """Chuẩn hóa trạng thái trả về theo SuccessfulSubmissionsStatus hoặc ErrorStatus"""
    try:
        data = response.json()
    except Exception:
        data = {"raw": response.text or ""}

    raw_text = (response.text or "").lower()

    # DRES trả về 412 khi Submission bị Reject (Sai kết quả)
    if response.status_code == 412:
        return {"status": "wrong", "message": "Wrong Answer", "data": data}

    if not response.ok:
        desc = data.get("description", f"HTTP Error {response.status_code}")
        return {"status": "error", "message": desc, "data": data}

    # Trường hợp 202: Server đã nhận bài nhưng chưa có kết quả chấm lập tức
    if response.status_code == 202:
        return {
            "status": "pending",
            "message": "Submitted, waiting for verdict",
            "data": data,
        }

    # Trường hợp 200: Có kết quả trả về luôn (Dựa vào schema ApiVerdictStatus)
    verdict = str(data.get("submission", "")).upper()
    if "CORRECT" in verdict or "CORRECT" in raw_text:
        return {"status": "correct", "message": "Correct!", "data": data}
    if "WRONG" in verdict or "WRONG" in raw_text:
        return {"status": "wrong", "message": "Wrong Answer", "data": data}

    return {"status": "pending", "message": "Submitted", "data": data}
def timestamp_to_milliseconds(timestamp: float | None, fallback_frame_id: int) -> int:
    if timestamp is None:
        return int(fallback_frame_id)

    value = safe_float(timestamp, -1.0)

    if value < 0:
        return int(fallback_frame_id)

    return int(round(value * 1000))

# ============================================================
# 6. API Routes
# ============================================================

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "backend_dir": str(BACKEND_DIR),
        "config_path": str(CONFIG_PATH),
        "faiss_index_path": str(FAISS_INDEX_PATH),
        "metadata_path": str(METADATA_PATH),
        "keyframes_root": str(KEYFRAMES_ROOT),
        "videos_root": str(VIDEOS_ROOT),
        "map_keyframe_path": str(MAP_KEYFRAME_ROOT),
        "keyframes_root_exists": KEYFRAMES_ROOT.exists(),
        "videos_root_exists": VIDEOS_ROOT.exists(),
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
            "candidate_multiplier": int(CFG["search"].get("candidate_multiplier", 5)),
            "available_modes": ["semantic", "temporal"],
            "default_search_mode": "semantic",
            "default_duration_limit": -1,
        },
        "ui": {
            "surrounding_radius": int(CFG["ui"].get("surrounding_radius", 5)),
            "max_surrounding_radius": int(CFG["ui"].get("max_surrounding_radius", 10)),
        },
        "translate": {
            "enabled_default": bool(
                CFG.get("translate", {}).get("enabled_default", False)
            ),
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
    }



@app.post("/api/dres/login")
def dres_login(payload: DresLoginRequest):
    dres_url = clean_external_url(payload.dres_url)

    if not payload.username.strip() or not payload.password:
        raise HTTPException(
            status_code=400, detail="Missing username or password"
        )

    session = requests.Session()
    session.headers.update(DRES_HEADERS)

    # 1. Thực hiện đăng nhập thông qua POST /api/v2/login
    try:
        login_res = session.post(
            f"{dres_url}/api/v2/login",
            json={"username": payload.username, "password": payload.password},
            timeout=15,
        )
    except requests.RequestException as e:
        raise HTTPException(
            status_code=502, detail=f"DRES login connection failed: {e}"
        )

    if not login_res.ok:
        try:
            err_desc = login_res.json().get("description", "Login failed")
        except Exception:
            err_desc = login_res.text or "Login failed"
        raise HTTPException(status_code=login_res.status_code, detail=err_desc)

    # 2. Lấy Session ID từ endpoint GET /api/v2/user/session
    try:
        sess_res = session.get(f"{dres_url}/api/v2/user/session", timeout=15)
    except requests.RequestException as e:
        raise HTTPException(
            status_code=502, detail=f"DRES session fetch failed: {e}"
        )

    if not sess_res.ok:
        raise HTTPException(
            status_code=sess_res.status_code, detail="Cannot fetch DRES session"
        )

    # Làm sạch chuỗi session_id trả về dạng plain text
    session_id = sess_res.text.strip().strip('"')

    # 3. Tự động lấy danh sách cuộc thi và bốc active evaluationId ra cho Client UI
    evaluations = fetch_dres_evaluations(dres_url, session_id)
    evaluation_id = pick_active_evaluation_id(evaluations)

    return {
        "status": "ok",
        "session_id": session_id,
        "evaluation_id": evaluation_id,
        "evaluations": evaluations,
        "user": login_res.json() if login_res.text else {},
    }


@app.post("/api/dres/submit")
def dres_submit(payload: DresSubmitRequest):
    dres_url = clean_external_url(payload.dres_url)

    if not payload.session_id.strip():
        raise HTTPException(status_code=400, detail="Missing active session_id")

    # Nếu Front-end không truyền lên evaluation_id, tự động quét tìm lại để tránh lỗi sập hệ thống
    evaluation_id = payload.evaluation_id
    if not evaluation_id:
        evals = fetch_dres_evaluations(dres_url, payload.session_id)
        evaluation_id = pick_active_evaluation_id(evals)

    if not evaluation_id:
        raise HTTPException(
            status_code=400, detail="No active DRES evaluation found to submit"
        )

    # Quy đổi thời gian sang mili-giây theo đúng mô tả dữ liệu ApiClientAnswer (start/end: int64 ms)
    # Nếu UI không có timestamp (ví dụ ảnh đơn lẻ), xử lý fallback an toàn theo frame_id
    if payload.timestamp is not None and payload.timestamp >= 0:
        time_ms = int(round(payload.timestamp * 1000))
    else:
        # Trường hợp giả định frame_id tương ứng mili-giây nếu UI bắn frame index trực tiếp
        time_ms = int(payload.frame_id)

    # Xây dựng Payload Object khớp 100% cấu trúc ApiClientSubmission trong OpenAPI docs của bạn
    submit_payload = {
        "answerSets": [
            {
                "answers": [
                    {
                        "mediaItemName": str(payload.video_id).strip(),
                        "start": time_ms,
                        "end": time_ms,
                        "text": None,
                        "mediaItemCollectionName": None,
                    }
                ]
            }
        ]
    }

    # Gọi chính xác cổng POST /api/v2/submit/{evaluationId} kèm tham số query session
    submit_url = f"{dres_url}/api/v2/submit/{evaluation_id}"
    params = {"session": payload.session_id}

    try:
        res = requests.post(
            submit_url, params=params, json=submit_payload, timeout=15
        )
        # Trả trạng thái chuẩn hóa về cho giao diện hiển thị (Correct, Wrong, Pending)
        return normalize_dres_verdict(res)

    except requests.RequestException as e:
        raise HTTPException(
            status_code=502, detail=f"Connection to DRES submit failed: {e}"
        )

@app.get("/api/frame-info")
def get_frame_info(video_id: str, keyframe_id: int):
    metadata_df = retrieval_system.metadata

    if "keyframe_id_int" in metadata_df.columns:
        mask = (
            (metadata_df["video_id"].astype(str) == str(video_id))
            & (metadata_df["keyframe_id_int"].astype(int) == int(keyframe_id))
        )
    else:
        mask = (
            (metadata_df["video_id"].astype(str) == str(video_id))
            & (metadata_df["keyframe_id"].astype(int) == int(keyframe_id))
        )

    matched = metadata_df[mask]

    if matched.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Frame not found: {video_id}/{keyframe_id}",
        )

    return row_to_result(matched.iloc[0])

@app.post("/api/search")
def search_api(payload: SearchRequest):
    original_query = payload.query.strip()

    if not original_query:
        raise HTTPException(status_code=400, detail="Query is empty")

    max_top_k = int(CFG["search"].get("max_top_k", 200))
    default_top_k = int(CFG["search"].get("default_top_k", 20))

    top_k = payload.top_k or default_top_k
    top_k = max(1, min(top_k, max_top_k))

    candidate_multiplier = payload.candidate_multiplier or int(
        CFG["search"].get("candidate_multiplier", 5)
    )
    candidate_multiplier = max(1, int(candidate_multiplier))

    use_split = True if payload.use_split is None else payload.use_split

    use_translate = (
        bool(CFG.get("translate", {}).get("enabled_default", False))
        if payload.use_translate is None
        else payload.use_translate
    )

    search_mode = normalize_search_mode(payload.search_mode)

    duration_limit = -1.0 if payload.duration_limit is None else float(payload.duration_limit)
    if duration_limit == 0:
        duration_limit = -1.0

    start = time.perf_counter()

    try:
        search_query = translate_query_if_needed(
            query=original_query,
            use_translate=use_translate,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Translate failed: {type(exc).__name__}: {exc}",
        )

    sub_queries = split_query(search_query) if use_split else [search_query]
    sub_queries = [q.strip() for q in sub_queries if q.strip()]

    if not sub_queries:
        raise HTTPException(status_code=400, detail="No valid sub queries")

    try:
        results_df = run_search(
            sub_queries=sub_queries,
            top_k=top_k,
            candidate_multiplier=candidate_multiplier,
            search_mode=search_mode,
            duration_limit=duration_limit,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {type(exc).__name__}: {exc}",
        )

    latency_ms = round((time.perf_counter() - start) * 1000)

    candidate_k = max(top_k * candidate_multiplier, top_k)

    response_base = {
        "original_query": original_query,
        "query": search_query,
        "translated_query": search_query if use_translate else None,
        "use_translate": use_translate,
        "use_split": use_split,
        "sub_queries": sub_queries,
        "search_mode": search_mode,
        "duration_limit": duration_limit,
        "top_k": top_k,
        "candidate_multiplier": candidate_multiplier,
        "candidate_k": candidate_k,
        "latency_ms": latency_ms,
    }

    if results_df.empty:
        return {
            **response_base,
            "count": 0,
            "results": [],
        }

    try:
        results = [row_to_result(row) for _, row in results_df.iterrows()]
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Serialize results failed: {type(exc).__name__}: {exc}",
        )

    return {
        **response_base,
        "count": len(results),
        "results": results,
    }