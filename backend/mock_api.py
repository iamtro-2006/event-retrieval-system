from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.api.models import (
    DresLoginRequest,
    DresSubmitRequest,
    SearchRequest,
    SimilaritySearchRequest,
)
from src.api.serialization import (
    format_keyframe_id,
    safe_float,
    safe_int,
)


# =========================
# CONFIG
# =========================

BACKEND_DIR = Path(__file__).resolve().parent

KEYFRAMES_ROOT = BACKEND_DIR / "data" / "processed" / "keyframes"
VIDEOS_ROOT = BACKEND_DIR / "data" / "raw" / "videos"
MAP_KEYFRAME_ROOT = BACKEND_DIR / "data" / "processed" / "map_keyframes"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTS = [".mp4", ".mkv", ".avi", ".mov", ".webm"]

DEFAULT_TOP_K = 20
MAX_TOP_K = 200
DEFAULT_SURROUNDING_RADIUS = 5
MAX_SURROUNDING_RADIUS = 10


# =========================
# LOGGING
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)

logger = logging.getLogger("mock-api")


# =========================
# APP
# =========================

app = FastAPI(title="Mock Event Retrieval API", version="mock-1.0.0")


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

if MAP_KEYFRAME_ROOT.exists():
    app.mount(
        "/static/map-keyframes",
        StaticFiles(directory=str(MAP_KEYFRAME_ROOT)),
        name="map_keyframes",
    )


# =========================
# CACHE
# =========================

IMAGE_CACHE: list[Path] = []
VIDEO_FRAME_INDEX: dict[str, list[Path]] = {}
FRAME_LOOKUP: dict[tuple[str, int], Path] = {}


# =========================
# UTILS
# =========================


def clamp_top_k(top_k: int | None) -> int:
    return max(1, min(int(top_k or DEFAULT_TOP_K), MAX_TOP_K))


def format_frame_id(frame_id: int | str) -> str:
    value = str(frame_id)

    if value.isdigit():
        return value.zfill(6)

    return value


def parse_image_identity(image_path: Path) -> dict[str, Any]:
    rel = image_path.relative_to(KEYFRAMES_ROOT)
    parts = rel.parts

    dataset = ""
    video_id = "unknown_video"

    if len(parts) >= 3:
        dataset = parts[0]
        video_id = parts[1]
    elif len(parts) >= 2:
        video_id = parts[0]

    frame_stem = image_path.stem
    frame_id = safe_int(frame_stem, 0)

    return {
        "dataset": dataset,
        "video_id": video_id,
        "frame_id": frame_id,
        "frame_id_text": format_frame_id(frame_id),
        "image_rel_path": rel.as_posix(),
    }


def find_video_url(dataset: str, video_id: str) -> tuple[str, str]:
    candidates: list[Path] = []

    for ext in VIDEO_EXTS:
        if dataset:
            candidates.append(VIDEOS_ROOT / dataset / f"{video_id}{ext}")
        candidates.append(VIDEOS_ROOT / f"{video_id}{ext}")

    for candidate in candidates:
        if candidate.exists():
            rel = candidate.relative_to(VIDEOS_ROOT).as_posix()
            return f"/static/videos/{rel}", rel

    return "#", ""


def find_map_url(dataset: str, video_id: str) -> tuple[str, str]:
    candidates: list[Path] = []

    if dataset:
        candidates.append(MAP_KEYFRAME_ROOT / dataset / f"{video_id}.csv")

    candidates.append(MAP_KEYFRAME_ROOT / f"{video_id}.csv")

    for candidate in candidates:
        if candidate.exists():
            rel = candidate.relative_to(MAP_KEYFRAME_ROOT).as_posix()
            return f"/static/map-keyframes/{rel}", rel

    return "", ""


def timestamp_from_frame(frame_id: int, fps: float = 25.0) -> float:
    if fps <= 0:
        fps = 25.0

    return round(frame_id / fps, 3)


def path_to_result(
    image_path: Path,
    rank: int = 0,
    caption: str = "Mock result - random keyframe",
    score: float | None = None,
) -> dict[str, Any]:
    identity = parse_image_identity(image_path)

    dataset = identity["dataset"]
    video_id = identity["video_id"]
    frame_id = identity["frame_id"]
    frame_id_text = identity["frame_id_text"]
    image_rel_path = identity["image_rel_path"]

    timestamp = timestamp_from_frame(frame_id)
    score = round(float(score if score is not None else random.uniform(0.25, 0.95)), 4)

    video_url, video_rel_path = find_video_url(dataset, video_id)
    map_url, map_rel_path = find_map_url(dataset, video_id)

    return {
        "id": f"{video_id}_{frame_id_text}",
        "video_id": video_id,
        "frame_id": frame_id,
        "frame_name": f"{frame_id_text}{image_path.suffix.lower()}",
        "path": f"{video_id}/{frame_id_text}",
        "keyframe_path": str(image_path),
        "image_url": f"/static/keyframes/{image_rel_path}",
        "image_rel_path": image_rel_path,
        "video_url": video_url,
        "video_rel_path": video_rel_path,
        "map_url": map_url,
        "map_rel_path": map_rel_path,
        "timestamp": timestamp,
        "similarity": score,
        "caption": caption,
        "rank": rank,
        "matched_sequence": [],
        "temporal": {
            "video_score": score,
            "start_time": timestamp,
            "end_time": timestamp,
            "duration_sec": 0.0,
            "avg_score": score,
        },
        "raw": {
            "dataset": dataset,
            "source_name": image_path.stem,
            "avg_score": score,
            "retrieval_score": score,
            "alignment_score": score,
            "frame_idx": frame_id,
            "fps": 25.0,
            "video_score": score,
            "temporal_start_time": timestamp,
            "temporal_end_time": timestamp,
            "temporal_duration_sec": 0.0,
            "matched_sequence": [],
            "keyframe_path": str(image_path),
            "image_rel_path": image_rel_path,
            "video_rel_path": video_rel_path,
            "map_path": "",
            "map_url": map_url,
            "map_rel_path": map_rel_path,
        },
    }


def build_temporal_sequence(selected: list[Path], query: str) -> list[dict[str, Any]]:
    sequence = []

    for idx, image_path in enumerate(selected):
        item = path_to_result(
            image_path=image_path,
            rank=idx + 1,
            caption="Mock temporal matched frame",
        )

        sequence.append(
            {
                "video_id": item["video_id"],
                "source_name": item["raw"]["source_name"],
                "keyframe_id": format_frame_id(item["frame_id"]),
                "frame_id": item["frame_id"],
                "frame_idx": item["raw"]["frame_idx"],
                "timestamp_sec": item["timestamp"],
                "fps": 25.0,
                "keyframe_path": item["keyframe_path"],
                "image_url": item["image_url"],
                "image_rel_path": item["image_rel_path"],
                "sub_query_idx": idx,
                "sub_query": query,
                "score": item["similarity"],
                "candidate_score": item["similarity"],
                "candidate_rank": idx + 1,
            }
        )

    return sequence


def sample_images(top_k: int) -> list[Path]:
    if not IMAGE_CACHE:
        return []

    return random.sample(IMAGE_CACHE, k=min(top_k, len(IMAGE_CACHE)))


# =========================
# STARTUP SCAN
# =========================


@app.on_event("startup")
def startup_scan_images():
    global IMAGE_CACHE, VIDEO_FRAME_INDEX, FRAME_LOOKUP

    start = time.perf_counter()

    logger.info(
        "[STARTUP_SCAN_START] root=%s exists=%s",
        KEYFRAMES_ROOT,
        KEYFRAMES_ROOT.exists(),
    )

    if not KEYFRAMES_ROOT.exists():
        logger.warning(
            "[STARTUP_SCAN_FAIL] keyframes root does not exist: %s", KEYFRAMES_ROOT
        )
        IMAGE_CACHE = []
        VIDEO_FRAME_INDEX = {}
        FRAME_LOOKUP = {}
        return

    images: list[Path] = []

    checked = 0
    for checked, path in enumerate(KEYFRAMES_ROOT.rglob("*"), start=1):
        if checked <= 20:
            logger.info("[STARTUP_SCAN_ITEM] %s", path)

        if checked % 10000 == 0:
            logger.info(
                "[STARTUP_SCAN_PROGRESS] checked=%d found_images=%d",
                checked,
                len(images),
            )

        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            images.append(path)

    video_index: dict[str, list[Path]] = {}
    frame_lookup: dict[tuple[str, int], Path] = {}

    for image_path in images:
        identity = parse_image_identity(image_path)
        video_id = identity["video_id"]
        frame_id = identity["frame_id"]

        video_index.setdefault(video_id, []).append(image_path)
        frame_lookup[(video_id, frame_id)] = image_path

    for video_id in video_index:
        video_index[video_id].sort(key=lambda p: safe_int(p.stem, 0))

    IMAGE_CACHE = images
    VIDEO_FRAME_INDEX = video_index
    FRAME_LOOKUP = frame_lookup

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

    logger.info(
        "[STARTUP_SCAN_DONE] checked=%d images=%d videos=%d elapsed_ms=%s",
        checked,
        len(IMAGE_CACHE),
        len(VIDEO_FRAME_INDEX),
        elapsed_ms,
    )


# =========================
# BASIC ENDPOINTS
# =========================


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "mode": "mock",
        "backend_dir": str(BACKEND_DIR),
        "keyframes_root": str(KEYFRAMES_ROOT),
        "videos_root": str(VIDEOS_ROOT),
        "map_keyframe_path": str(MAP_KEYFRAME_ROOT),
        "keyframes_root_exists": KEYFRAMES_ROOT.exists(),
        "videos_root_exists": VIDEOS_ROOT.exists(),
        "map_keyframe_root_exists": MAP_KEYFRAME_ROOT.exists(),
        "image_count": len(IMAGE_CACHE),
        "video_count": len(VIDEO_FRAME_INDEX),
        "model": {
            "name": "mock-random-frame",
            "pretrained": "none",
            "device": "none",
            "precision": "none",
            "normalize": False,
        },
    }


@app.get("/api/config")
def get_public_config():
    return {
        "search": {
            "default_top_k": DEFAULT_TOP_K,
            "max_top_k": MAX_TOP_K,
            "candidate_multiplier": 5,
            "available_modes": ["semantic", "temporal", "ocr", "asr", "auto"],
            "default_search_mode": "semantic",
            "default_duration_limit": -1,
        },
        "ui": {
            "surrounding_radius": DEFAULT_SURROUNDING_RADIUS,
            "max_surrounding_radius": MAX_SURROUNDING_RADIUS,
        },
        "translate": {
            "enabled_default": False,
            "source": "vi",
            "target": "en",
        },
        "model": {
            "name": "mock-random-frame",
            "pretrained": "none",
            "device": "none",
            "precision": "none",
            "normalize": False,
        },
    }


# =========================
# MOCK SEARCH
# =========================


@app.post("/api/search")
def search_api(payload: SearchRequest):
    request_start = time.perf_counter()

    logger.info(
        "[SEARCH_START] query=%r top_k=%s mode=%s cache_images=%d",
        payload.query,
        payload.top_k,
        payload.search_mode,
        len(IMAGE_CACHE),
    )

    original_query = payload.query.strip()

    if not original_query:
        raise HTTPException(status_code=400, detail="Query is empty")

    if not IMAGE_CACHE:
        raise HTTPException(
            status_code=404,
            detail=f"No images found in {KEYFRAMES_ROOT}",
        )

    top_k = clamp_top_k(payload.top_k)
    mode = payload.search_mode or "semantic"
    candidate_multiplier = max(1, int(payload.candidate_multiplier or 5))
    candidate_k = max(top_k * candidate_multiplier, top_k)

    sample_start = time.perf_counter()
    selected = sample_images(top_k)

    logger.info(
        "[SEARCH_AFTER_SAMPLE] selected=%d sample_ms=%s",
        len(selected),
        round((time.perf_counter() - sample_start) * 1000, 2),
    )

    serialize_start = time.perf_counter()

    results: list[dict[str, Any]] = []

    if mode == "temporal":
        for rank, image_path in enumerate(selected, start=1):
            item = path_to_result(
                image_path=image_path,
                rank=rank,
                caption="Mock temporal result - random keyframe",
            )

            sequence_paths = random.sample(
                IMAGE_CACHE,
                k=min(3, len(IMAGE_CACHE)),
            )

            matched_sequence = build_temporal_sequence(sequence_paths, original_query)
            timestamps = [safe_float(x["timestamp_sec"], 0.0) for x in matched_sequence]

            item["matched_sequence"] = matched_sequence
            item["temporal"] = {
                "video_score": item["similarity"],
                "start_time": min(timestamps) if timestamps else item["timestamp"],
                "end_time": max(timestamps) if timestamps else item["timestamp"],
                "duration_sec": (
                    max(timestamps) - min(timestamps) if len(timestamps) >= 2 else 0.0
                ),
                "avg_score": item["similarity"],
            }
            item["raw"]["matched_sequence"] = matched_sequence
            item["raw"]["temporal_start_time"] = item["temporal"]["start_time"]
            item["raw"]["temporal_end_time"] = item["temporal"]["end_time"]
            item["raw"]["temporal_duration_sec"] = item["temporal"]["duration_sec"]

            results.append(item)

    else:
        for rank, image_path in enumerate(selected, start=1):
            results.append(
                path_to_result(
                    image_path=image_path,
                    rank=rank,
                    caption=f"Mock {mode} result - random keyframe",
                )
            )

    logger.info(
        "[SEARCH_AFTER_SERIALIZE] results=%d serialize_ms=%s",
        len(results),
        round((time.perf_counter() - serialize_start) * 1000, 2),
    )

    latency_ms = round((time.perf_counter() - request_start) * 1000)

    logger.info("[SEARCH_DONE] mode=%s total_latency_ms=%s", mode, latency_ms)

    return {
        "original_query": original_query,
        "query": original_query,
        "translated_query": None,
        "use_translate": False
        if payload.use_translate is None
        else bool(payload.use_translate),
        "use_split": True if payload.use_split is None else bool(payload.use_split),
        "mode": mode,
        "search_mode": mode,
        "duration_limit": -1.0
        if payload.duration_limit is None
        else float(payload.duration_limit),
        "top_k": top_k,
        "candidate_multiplier": candidate_multiplier,
        "candidate_k": candidate_k,
        "latency_ms": latency_ms,
        "events": [original_query],
        "event_queries": [original_query],
        "sub_queries": [original_query],
        "count": len(results),
        "results": results,
    }


# =========================
# MOCK SIMILARITY SEARCH
# =========================


@app.post("/api/similarity-search")
def similarity_search_api(payload: SimilaritySearchRequest):
    request_start = time.perf_counter()

    logger.info(
        "[SIMILARITY_START] video_id=%s frame_id=%s top_k=%s cache_images=%d",
        payload.video_id,
        payload.frame_id,
        payload.top_k,
        len(IMAGE_CACHE),
    )

    if not IMAGE_CACHE:
        raise HTTPException(
            status_code=404,
            detail=f"No images found in {KEYFRAMES_ROOT}",
        )

    top_k = clamp_top_k(payload.top_k)

    selected = sample_images(top_k)

    results = []
    for rank, image_path in enumerate(selected, start=1):
        results.append(
            path_to_result(
                image_path=image_path,
                rank=rank,
                caption="Mock similarity result - random keyframe",
            )
        )

    source_path = FRAME_LOOKUP.get((str(payload.video_id), int(payload.frame_id)))

    if source_path:
        source = path_to_result(
            source_path,
            rank=0,
            caption="Mock source frame",
            score=1.0,
        )
    else:
        frame_id_text = format_frame_id(payload.frame_id)
        source = {
            "id": f"{payload.video_id}_{frame_id_text}",
            "video_id": str(payload.video_id),
            "frame_id": int(payload.frame_id),
            "frame_name": f"{frame_id_text}.jpg",
            "path": f"{payload.video_id}/{frame_id_text}",
            "keyframe_path": "",
            "image_url": "",
            "image_rel_path": "",
            "video_url": "#",
            "video_rel_path": "",
            "map_url": "",
            "map_rel_path": "",
            "timestamp": 0.0,
            "similarity": 1.0,
            "caption": "Mock source frame not found",
            "rank": 0,
            "matched_sequence": [],
            "temporal": {
                "video_score": 1.0,
                "start_time": 0.0,
                "end_time": 0.0,
                "duration_sec": 0.0,
                "avg_score": 1.0,
            },
            "raw": {},
        }

    latency_ms = round((time.perf_counter() - request_start) * 1000)

    logger.info(
        "[SIMILARITY_DONE] results=%d total_latency_ms=%s",
        len(results),
        latency_ms,
    )

    return {
        "query": f"similarity:{payload.video_id}/{int(payload.frame_id):06d}",
        "search_mode": "similarity",
        "latency_ms": latency_ms,
        "count": len(results),
        "source": source,
        "results": results,
    }


# =========================
# FRAME INFO
# =========================


@app.get("/api/frame-info")
def get_frame_info(video_id: str, keyframe_id: int):
    request_start = time.perf_counter()

    logger.info(
        "[FRAME_INFO_START] video_id=%s keyframe_id=%s",
        video_id,
        keyframe_id,
    )

    image_path = FRAME_LOOKUP.get((str(video_id), int(keyframe_id)))

    if not image_path:
        raise HTTPException(
            status_code=404,
            detail=f"Frame not found: {video_id}/{int(keyframe_id):06d}",
        )

    result = path_to_result(
        image_path,
        rank=0,
        caption="Mock frame info",
    )

    logger.info(
        "[FRAME_INFO_DONE] elapsed_ms=%s",
        round((time.perf_counter() - request_start) * 1000, 2),
    )

    return result


# =========================
# SURROUNDING FRAMES
# =========================


@app.get("/api/surrounding-frames")
def get_surrounding_frames(video_id: str, keyframe_id: int, radius: int = 10):
    request_start = time.perf_counter()

    logger.info(
        "[SURROUND_START] video_id=%s keyframe_id=%s radius=%s",
        video_id,
        keyframe_id,
        radius,
    )

    radius = max(1, min(int(radius), 50))
    video_frames = VIDEO_FRAME_INDEX.get(str(video_id), [])

    if not video_frames:
        raise HTTPException(status_code=404, detail=f"Video not found: {video_id}")

    target_frame_id = int(keyframe_id)

    center_idx = next(
        (
            idx
            for idx, path in enumerate(video_frames)
            if safe_int(path.stem, -1) == target_frame_id
        ),
        None,
    )

    if center_idx is None:
        raise HTTPException(
            status_code=404,
            detail=f"Frame not found: {video_id}/{target_frame_id:06d}",
        )

    start_idx = max(0, center_idx - radius)
    end_idx = min(len(video_frames), center_idx + radius + 1)

    selected = video_frames[start_idx:end_idx]

    frames = []

    for path in selected:
        item = path_to_result(
            path,
            rank=0,
            caption="Mock surrounding frame",
        )
        item["is_surround_center"] = item["frame_id"] == target_frame_id
        item["surround_offset"] = item["frame_id"] - target_frame_id
        frames.append(item)

    logger.info(
        "[SURROUND_DONE] frames=%d elapsed_ms=%s",
        len(frames),
        round((time.perf_counter() - request_start) * 1000, 2),
    )

    return {
        "video_id": str(video_id),
        "center_frame_id": target_frame_id,
        "radius": radius,
        "count": len(frames),
        "frames": frames,
    }


# =========================
# MOCK DRES ENDPOINTS
# =========================


@app.post("/api/dres/login")
def dres_login(payload: DresLoginRequest):
    logger.info(
        "[MOCK_DRES_LOGIN] dres_url=%s username=%s", payload.dres_url, payload.username
    )

    return {
        "status": "ok",
        "session_id": "mock-session-id",
        "evaluation_id": "mock-evaluation-id",
        "evaluations": [
            {
                "id": "mock-evaluation-id",
                "name": "Mock Evaluation",
                "status": "ACTIVE",
            }
        ],
        "user": {
            "username": payload.username,
            "mock": True,
        },
    }


@app.post("/api/dres/submit")
def dres_submit(payload: DresSubmitRequest):
    logger.info(
        "[MOCK_DRES_SUBMIT] video_id=%s frame_id=%s timestamp=%s",
        payload.video_id,
        payload.frame_id,
        payload.timestamp,
    )

    return {
        "status": random.choice(["correct", "wrong", "pending"]),
        "message": "Mock DRES verdict",
        "data": {
            "video_id": payload.video_id,
            "frame_id": payload.frame_id,
            "timestamp": payload.timestamp,
            "mock": True,
        },
    }
