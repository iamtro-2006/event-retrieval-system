"""Standalone local mock backend + Codex tool playground.

Run from backend: python -m uvicorn mock_server:app --host 127.0.0.1 --port 8001
Install only requirements-mock.txt. Real retrieval, DRES and model loading are
never imported. See docs/MOCK_SERVER.md for frontend and Codex setup.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from contextlib import aclosing, asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.mock_codex import codex_status, stream_codex
from src.mock_tools import DATASETS, MODEL, MODES, MockCatalog, MockTools
from src.mock_workers import stream_parallel
from src.api.legacy.video_keyframes import build_video_keyframes

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("MOCK_DATA_ROOT", str(ROOT / "data"))).resolve()
STATE_ROOT = DATA_ROOT / "mock-state"
Dataset = Literal["v3c", "lhe"]


@asynccontextmanager
async def lifespan(app):
    app.state.catalog = MockCatalog(DATA_ROOT)
    await asyncio.to_thread(app.state.catalog.load)
    app.state.sessions = {}
    app.state.codex_slots = asyncio.Semaphore(2)
    app.state.worker_slots = asyncio.Semaphore(3)
    yield


app = FastAPI(title="Dataset Mock + Codex Playground", lifespan=lifespan)
app.add_middleware(CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["GET", "POST", "DELETE"], allow_headers=["content-type", "ngrok-skip-browser-warning"])
for directory, mount in (("keyframes", "/keyframes"), ("videos", "/videos"), ("map_keyframes", "/mapkeyframes")):
    folder = DATA_ROOT / directory
    if folder.exists():
        app.mount(mount, StaticFiles(directory=folder), name=directory)


def catalog(request):
    return request.app.state.catalog


def public_data(data, request):
    """Make image/video URLs portable across FE origins and remove local paths."""
    if isinstance(data, list):
        return [public_data(value, request) for value in data]
    if not isinstance(data, dict):
        return data
    output = {}
    for key, value in data.items():
        if key == "keyframe_path":
            continue
        if key in {"image_url", "video_url", "map_url"} and isinstance(value, str) and value.startswith("/"):
            output[key] = str(request.base_url).rstrip("/") + value
        else:
            output[key] = public_data(value, request)
    return output


def bad_request(exc):
    raise HTTPException(422 if isinstance(exc, ValidationError) else 404, detail=str(exc)) from exc


@app.get("/health")
@app.get("/api/health")
def health(request: Request):
    return {"status": "ok", **catalog(request).capabilities(), "available_models": [MODEL],
            "ocr": {"available": True, "mock": True}, "asr": {"available": True, "mock": True}}


@app.get("/api/config")
def config(request: Request):
    return {"mock": True, "agent": {"available": True, "providers": ["demo", "codex"]},
            "datasets": {"default": "v3c", "available": catalog(request).capabilities()["datasets"]},
            "search": {"default_top_k": 20, "max_top_k": 500, "candidate_multiplier": 4,
                       "available_modes": list(MODES), "default_model_key": MODEL},
            "model": {"key": MODEL, "name": "Mock Qwen retrieval", "device": "mock"},
            "available_models": [MODEL], "semantic_model_roles": {"local": MODEL, "global": MODEL},
            "translate": {"enabled_default": False},
            "ui": {"surrounding_radius": 5, "max_surrounding_radius": 10},
            "ocr": {"available": True}, "asr": {"available": True}}


@app.get("/api/search/models")
def models(dataset: Dataset):
    return {"models": [MODEL], "mock": True, "dataset": dataset}


@app.get("/api/video-ids")
def video_ids(request: Request, dataset: Dataset):
    return {"video_ids": catalog(request).video_ids(dataset), "mock": True}


@app.get("/api/color/palette")
def palette():
    return {"mock": True, "palette": {"red": "#ef4444", "orange": "#f97316", "yellow": "#eab308",
            "green": "#22c55e", "blue": "#3b82f6", "purple": "#a855f7", "white": "#ffffff", "black": "#000000"}}


class SearchPayload(BaseModel):
    # Legacy frontend sends extra tuning fields, intentionally ignored in mock.
    dataset: Dataset
    query: str = Field(default="Mock sample", max_length=4000)
    top_k: int = Field(default=20, ge=1, le=500)
    search_mode: Literal["semantic", "temporal", "ocr", "asr", "caption", "hierarchical", "auto", "fusion", "color", "similarity"] = "semantic"
    model_key: str | None = Field(default=None, max_length=200)
    video_ids: list[str] = Field(default_factory=list, max_length=10000)
    video_id: str | None = None
    frame_id: int = Field(default=0, ge=0)


def search_response(payload, request, mode=None):
    result = catalog(request).search(payload.dataset, payload.query, mode or payload.search_mode,
                                      payload.top_k, payload.model_key or MODEL, payload.video_ids)
    result["query_plan"] = {"mode": mode or payload.search_mode, "use_split": False, "events": [[payload.query]]}
    return public_data(result, request)


@app.post("/api/search")
def search(payload: SearchPayload, request: Request):
    return search_response(payload, request)


@app.post("/api/search/multimodal")
async def multimodal(request: Request):
    form = await request.form(max_files=20, max_fields=20, max_part_size=5 * 1024 * 1024)
    try:
        payload = SearchPayload.model_validate(json.loads(str(form.get("request_json", "{}"))))
    except (ValueError, ValidationError) as exc:
        raise HTTPException(422, detail="Invalid request_json") from exc
    return search_response(payload, request)


@app.post("/api/search/{mode}")
def search_mode(mode: str, payload: SearchPayload, request: Request):
    if mode not in MODES and mode != "advanced":
        raise HTTPException(404, detail="Unknown mock search mode")
    return search_response(payload, request, "fusion" if mode == "advanced" else mode)


@app.post("/api/similarity-search")
def similarity(payload: SearchPayload, request: Request):
    if not payload.video_id:
        raise HTTPException(422, detail="video_id is required")
    try:
        source = catalog(request).find(payload.dataset, payload.video_id, payload.frame_id)
    except ValueError as exc:
        bad_request(exc)
    return {**search_response(payload, request, "similarity"), "source": public_data(source, request)}


@app.get("/api/frame-info")
def frame_info(request: Request, dataset: Dataset, video_id: str, keyframe_id: int = Query(ge=0)):
    try:
        return public_data(catalog(request).find(dataset, video_id, keyframe_id), request)
    except ValueError as exc:
        bad_request(exc)


@app.get("/api/surrounding-frames")
def surrounding(request: Request, dataset: Dataset, video_id: str,
                keyframe_id: int = Query(ge=0), radius: int = Query(default=5, ge=1, le=10),
                all_frames: bool = False):
    try:
        center = catalog(request).find(dataset, video_id, keyframe_id)
        rows = (catalog(request).by_video[(dataset, video_id)] if all_frames else
                catalog(request).surrounding(dataset, video_id, keyframe_id, radius))
        return {"frames": public_data(rows, request), "mock": True, "all_frames": all_frames,
                "count": len(rows), "center_frame_id": center["frame_id"]}
    except ValueError as exc:
        bad_request(exc)


@app.get("/api/video-preview")
def video_preview(request: Request, dataset: Dataset, video_id: str,
                  frame_id: int = Query(default=0, ge=0), timestamp_ms: int | None = Query(default=None, ge=0)):
    rows = catalog(request).by_video.get((dataset, video_id))
    if not rows:
        raise HTTPException(404, detail="Video not found in selected dataset")
    if timestamp_ms is not None:
        row = min(rows, key=lambda item: abs(item["timestamp"] * 1000 - timestamp_ms))
        frame_idx = round(timestamp_ms / 1000 * row["fps"]) if row.get("fps") else row["frame_idx"]
        result = {**row, "timestamp": timestamp_ms / 1000, "frame_idx": frame_idx, "is_preview_timestamp": True}
    else:
        row = min(rows, key=lambda item: abs(item["frame_idx"] - frame_id))
        timestamp = frame_id / row["fps"] if row.get("fps") else row["timestamp"]
        result = {**row, "timestamp": timestamp, "frame_idx": frame_id, "is_preview_frame": True}
    result["raw"] = {**row["raw"], "frame_idx": result["frame_idx"]}
    return public_data(result, request)


@app.get("/api/video-keyframes")
def video_keyframes(request: Request, dataset: Dataset, video_id: str):
    rows = catalog(request).by_video.get((dataset, video_id))
    if not rows:
        raise HTTPException(404, detail="Video not found in selected dataset")
    return public_data({"mock": True, "video_id": video_id,
                        "keyframes": build_video_keyframes(rows, lambda row: row)}, request)


class RerankPayload(BaseModel):
    query: str = "Mock rerank"
    results: list[dict] = Field(max_length=500)
    top_k: int = Field(default=20, ge=1, le=500)


@app.post("/api/rerank")
def rerank(payload: RerankPayload, request: Request):
    # Keep stable input order: mock reranking makes no relevance claims.
    return public_data({"mock": True, "query": payload.query, "results": payload.results[:payload.top_k],
                        "count": min(len(payload.results), payload.top_k), "latency_ms": 1}, request)


@app.post("/api/speech/transcribe")
def speech():
    return {"text": "[MOCK] Tìm một cảnh trong video", "language": "vi", "mock": True}


@app.post("/api/translate")
async def translate(request: Request):
    body = await request.json()
    return {"text": body.get("text", ""), "translated_text": body.get("text", ""), "provider": "mock", "mock": True}


@app.post("/api/dres/login")
def dres_login():
    return {"session_id": "mock-session", "mock": True,
            "evaluation_id": "mock-evaluation", "user": {"username": "mock-user"},
            "evaluations": [{"id": "mock-evaluation", "name": "Mock evaluation"}], "status": "ok"}


@app.post("/api/dres/submit")
def dres_submit():
    return {"status": "correct", "message": "MOCK ONLY: no answer sent to DRES.", "mock": True}


@app.get("/api/agent/tools")
def agent_tools():
    return {"tools": MockTools.specs(), "mock": True}


@app.get("/api/agent/status")
async def agent_status(request: Request):
    status = await codex_status(STATE_ROOT)
    return {"mock": True, "demo_available": True, "codex": status,
            "capabilities": catalog(request).capabilities()}


class ColorCell(BaseModel):
    model_config = ConfigDict(extra="forbid")
    row: int = Field(ge=0, le=4)
    col: int = Field(ge=0, le=4)
    color: Literal["black", "blue", "brown", "grey", "green", "orange", "pink", "purple", "red", "white", "yellow"]


class ChatPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str = Field(min_length=1, max_length=4000)
    dataset: Dataset
    provider: Literal["demo", "codex"] = "demo"
    model: str | None = Field(default=None, max_length=120, pattern=r"^[a-zA-Z0-9._:/-]+$")
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"] | None = None
    session_id: str | None = Field(default=None, max_length=100)
    top_k: int = Field(default=200, ge=1, le=500)
    top_k_limit: int = Field(default=200, ge=1, le=500)
    video_ids: list[str] = Field(default_factory=list, max_length=10000)
    max_workers: int = Field(default=3, ge=1, le=3)
    search_mode: Literal["semantic", "temporal", "ocr", "asr", "caption", "hierarchical", "auto", "fusion", "color", "similarity"] = "semantic"
    context_frame_ids: list[str] = Field(default_factory=list, max_length=100)
    color_cells: list[ColorCell] = Field(default_factory=list, max_length=25)
    images: list[Annotated[str, Field(max_length=3_000_000,
        pattern=r"^data:image/(png|jpeg|jpg|webp|gif);base64,[A-Za-z0-9+/=\r\n]+$")]] = Field(default_factory=list, max_length=6)


async def demo_stream(tools, payload):
    yield {"type": "status", "message": "Demo scripted · exercising the same tool registry"}
    for name, args in [("list_capabilities", {}), ("search", {"query": payload.message, "top_k": payload.top_k})]:
        yield {"type": "tool_start", "name": name, "arguments": args}
        data, _ = await asyncio.to_thread(tools.execute, name, args)
        yield {"type": "tool_end", "name": name, "count": data.get("count"), "success": True}
        if name == "search":
            rows = data["results"]
            yield {"type": "results", "results": rows, "query": payload.message, "tool": name}
        await asyncio.sleep(.08)
    if len(rows) > 6 and payload.max_workers > 1:
        arguments = {"frame_ids": [row["id"] for row in rows[:48]], "instruction": payload.message}
        yield {"type": "tool_start", "name": "parallel_inspect", "arguments": arguments}
        async with aclosing(stream_parallel(STATE_ROOT, tools, "parallel_inspect", arguments, demo=True)) as source:
            async for event in source:
                if event["type"] != "parallel_result":
                    yield event
        yield {"type": "tool_end", "name": "parallel_inspect", "success": True, "count": min(len(rows), 48)}
    elif rows:
        ids = [row["id"] for row in rows[:3]]
        yield {"type": "tool_start", "name": "inspect_frames", "arguments": {"frame_ids": ids}}
        await asyncio.to_thread(tools.execute, "inspect_frames", {"frame_ids": ids})
        yield {"type": "tool_end", "name": "inspect_frames", "success": True, "count": len(ids)}
    inspection_tool = "parallel_inspect" if len(rows) > 6 and payload.max_workers > 1 else "inspect_frames"
    text = (f"Đã lấy {len(rows)} ảnh ngẫu nhiên từ {payload.dataset.upper()} và chạy thử tool {inspection_tool}. "
            "Đây là demo theo kịch bản, không có model đánh giá ảnh. Chọn Codex để chat và gọi tool bằng AI thật. "
            "Thứ hạng và thời điểm đều là dữ liệu mô phỏng.")
    if not rows:
        text = f"Chưa có ảnh phù hợp bộ lọc trong {payload.dataset.upper()}. Kiểm tra thư mục dataset hoặc bỏ bộ lọc video."
    for start in range(0, len(text), 18):
        yield {"type": "delta", "text": text[start:start + 18]}
        await asyncio.sleep(.015)


@app.post("/api/agent/chat")
async def chat(payload: ChatPayload, request: Request):
    try:
        context_rows = catalog(request).selected(payload.dataset, payload.context_frame_ids)
        if payload.video_ids and any(row["video_id"] not in payload.video_ids for row in context_rows):
            raise ValueError("Frame context excluded by current video filter")
    except ValueError as exc:
        raise HTTPException(422, detail=str(exc)) from exc
    sessions = request.app.state.sessions
    now = time.monotonic()
    for identifier in list(sessions):
        if not sessions[identifier]["busy"] and now - sessions[identifier]["updated"] > 3600:
            sessions.pop(identifier)
    if payload.session_id:
        session = sessions.get(payload.session_id)
        if session is None:
            raise HTTPException(409, detail="Chat session expired. Start a new chat.")
        if session["dataset"] != payload.dataset or session["provider"] != payload.provider or session["model"] != payload.model:
            raise HTTPException(409, detail="Dataset/provider/model changed. Start a new chat.")
        identifier = payload.session_id
    else:
        if len(sessions) >= 100:
            raise HTTPException(429, detail="Too many sessions. Restart mock server or wait for expiry.")
        identifier = str(uuid.uuid4())
        session = {"dataset": payload.dataset, "provider": payload.provider, "model": payload.model,
                   "history": [], "known": [], "results": [], "busy": False, "updated": now}
        sessions[identifier] = session
    if session["busy"]:
        raise HTTPException(409, detail="This chat already has a running turn.")
    if payload.provider == "codex" and request.app.state.codex_slots.locked():
        raise HTTPException(429, detail="Two Codex turns are already running. Retry after one finishes.")
    session["busy"] = True

    async def events():
        tools = MockTools(catalog(request), payload.dataset, session["known"], payload.video_ids,
                          context_rows or session["results"], payload.max_workers,
                          getattr(request.app.state, "worker_slots", None))
        tools.known_ids.update(payload.context_frame_ids)
        tools.input_images = payload.images
        tools.reasoning_effort = payload.reasoning_effort
        tools.preferred_search_mode = payload.search_mode
        tools.color_cells = [cell.model_dump() for cell in payload.color_cells]
        answer = ""
        try:
            yield {"type": "session", "session_id": identifier, "provider": payload.provider, "dataset": payload.dataset}
            if payload.provider == "codex":
                async with request.app.state.codex_slots:
                    async with aclosing(stream_codex(STATE_ROOT, tools, payload.message, session["history"], payload.model)) as source:
                        async for event in source:
                            if event["type"] == "delta":
                                answer += event["text"]
                            if event["type"] == "results":
                                session["results"] = event["results"]
                            yield event
            else:
                async for event in demo_stream(tools, payload):
                    if event["type"] == "delta":
                        answer += event["text"]
                    if event["type"] == "results":
                        session["results"] = event["results"]
                    yield event
            session["history"] = (session["history"] + [{"role": "user", "text": payload.message},
                                                       {"role": "assistant", "text": answer}])[-16:]
            yield {"type": "done", "session_id": identifier, "tool_calls": tools.calls}
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            yield {"type": "error", "message": str(exc) or "Agent timed out. Retry with a narrower request."}
        finally:
            session["known"] = list(tools.known_ids)[-1000:]
            session["busy"] = False
            session["updated"] = time.monotonic()

    async def encoded():
        async with aclosing(events()) as source:
            async for event in source:
                yield json.dumps(public_data(event, request), ensure_ascii=False) + "\n"

    return StreamingResponse(encoded(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
