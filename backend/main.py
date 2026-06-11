from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from deep_translator import GoogleTranslator

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
    """
    Resolve relative path theo backend/.

    Ví dụ YAML:
        data/processed/keyframes

    Sẽ thành:
        D:/event-retrieval-system/backend/data/processed/keyframes
    """
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

# Thêm cấu hình allow host/origin linh hoạt cho ngrok
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://exemplifiable-gauntly-naomi.ngrok-free.dev", # Domain ngrok Frontend hiện tại của bạn
        "*" # Dấu "*" cho phép TẤT CẢ các origin truy cập vào (Khuyên dùng khi bạn thường xuyên đổi ngrok ngẫu nhiên)
    ],
    allow_credentials=True,
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

# ============================================================
# 5. Helper Functions
# ============================================================

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def format_keyframe_id(row: pd.Series) -> str:
    """
    Ưu tiên:
    1. keyframe_id
    2. keyframe_id_int
    3. frame_idx
    """
    if "keyframe_id" in row:
        value = str(row["keyframe_id"])

        if value.isdigit():
            return value.zfill(6)

        return value

    if "keyframe_id_int" in row:
        return f"{int(row['keyframe_id_int']):06d}"

    if "frame_idx" in row:
        return f"{int(row['frame_idx']):06d}"

    return "000000"


def make_static_keyframe_url(keyframe_path: str | Path) -> str:
    """
    Convert keyframe_path trong metadata thành URL static cho frontend.

    Ví dụ metadata:
        data/processed/keyframes/L21_a/L21_V001/000001.jpg

    API trả:
        /static/keyframes/L21_a/L21_V001/000001.jpg
    """
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
    """
    Tìm video gốc dựa trên:
    1. row["video_path"] nếu có
    2. videos_root/dataset/video_id.ext
    3. videos_root/video_id.ext
    """
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
        "rank": safe_int(row.get("display_rank", 0), 0),
        "raw": {
            "dataset": str(row.get("dataset", "") or ""),
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
        },
    }


def run_search(
    query: str,
    top_k: int,
    candidate_multiplier: int,
    use_split: bool,
    use_translate: bool = False,

):
    sub_queries = split_query(query) if use_split else [query]
    candidate_k = top_k * candidate_multiplier

    all_results = []

    for sub_query in sub_queries:
        df = retrieval_system.search(sub_query, top_k=candidate_k)

        if not df.empty:
            all_results.append(df)

    results = rerank_multi_query(all_results)

    if results.empty:
        return results, sub_queries

    if "alignment_score" in results.columns and "retrieval_score" not in results.columns:
        results["retrieval_score"] = results["alignment_score"]

    results = results.head(top_k).copy()
    results["display_rank"] = range(1, len(results) + 1)

    return results, sub_queries


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

    use_split = True if payload.use_split is None else payload.use_split

    use_translate = (
        bool(CFG.get("translate", {}).get("enabled_default", False))
        if payload.use_translate is None
        else payload.use_translate
    )

    start = time.perf_counter()

    try:
        search_query = translate_query_if_needed(
            query=original_query,
            use_translate=use_translate,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Translate failed: {exc}",
        )

    results_df, sub_queries = run_search(
        query=search_query,
        top_k=top_k,
        candidate_multiplier=candidate_multiplier,
        use_split=use_split,
    )

    latency_ms = round((time.perf_counter() - start) * 1000)

    response_base = {
        "original_query": original_query,
        "query": search_query,
        "translated_query": search_query if use_translate else None,
        "use_translate": use_translate,
        "use_split": use_split,
        "sub_queries": sub_queries,
        "latency_ms": latency_ms,
    }

    if results_df.empty:
        return {
            **response_base,
            "count": 0,
            "results": [],
        }

    results = [row_to_result(row) for _, row in results_df.iterrows()]

    return {
        **response_base,
        "count": len(results),
        "results": results,
    }