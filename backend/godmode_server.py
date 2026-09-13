"""Independent verified-result relay for collaborative DRES sessions.

Run separately from the retrieval API:
    uvicorn godmode_server:app --host 0.0.0.0 --port 8081
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import asyncio
from pathlib import Path
from typing import Any

import requests
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.api.legacy.dres_client import clean_external_url, normalize_dres_verdict

DB_PATH = Path(os.environ.get("GODMODE_DB_PATH", "data/godmode.sqlite3")).resolve()
MAX_RESULTS = max(1, int(os.environ.get("GODMODE_MAX_RESULTS", "200")))


class VerifiedSubmit(BaseModel):
    dres_url: str
    session_id: str
    evaluation_id: str
    video_id: str
    frame_id: int
    timestamp: float = Field(ge=0)
    result: dict[str, Any] = Field(default_factory=dict)


app = FastAPI(title="DRES God Mode Relay", docs_url="/docs")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[value.strip() for value in os.environ.get("GODMODE_CORS_ORIGINS", "*").split(",")],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["content-type", "ngrok-skip-browser-warning"],
)
connections: dict[WebSocket, str] = {}


def connect_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.execute("""
        CREATE TABLE IF NOT EXISTS verified_results (
            evaluation_id TEXT NOT NULL,
            video_id TEXT NOT NULL,
            frame_id INTEGER NOT NULL,
            timestamp REAL NOT NULL,
            result_json TEXT NOT NULL,
            verified_at REAL NOT NULL,
            PRIMARY KEY (evaluation_id, video_id, frame_id)
        )
    """)
    return connection


def read_results(evaluation_id: str) -> list[dict[str, Any]]:
    with connect_db() as db:
        rows = db.execute(
            "SELECT result_json FROM verified_results WHERE evaluation_id=? ORDER BY verified_at DESC LIMIT ?",
            (evaluation_id, MAX_RESULTS),
        ).fetchall()
    return [json.loads(row[0]) for row in rows]


def safe_result(payload: VerifiedSubmit) -> dict[str, Any]:
    source = payload.result
    return {
        "id": f"godmode:{payload.evaluation_id}:{payload.video_id}:{payload.frame_id}",
        "video_id": payload.video_id,
        "frame_id": payload.frame_id,
        "timestamp": payload.timestamp,
        "image_url": str(source.get("image_url") or ""),
        "video_url": str(source.get("video_url") or ""),
        "similarity": 1.0,
        "search_mode": "godmode",
        "godmode_verified": True,
        "evaluation_id": payload.evaluation_id,
        "verified_at": time.time(),
        "raw": {
            "frame_idx": payload.frame_id,
            "keyframe_path": str((source.get("raw") or {}).get("keyframe_path") or ""),
        },
    }


async def broadcast(message: dict[str, Any]) -> None:
    dead: list[WebSocket] = []
    evaluation_id = str(message.get("evaluation_id") or "")
    for socket, subscribed_evaluation in tuple(connections.items()):
        if subscribed_evaluation != evaluation_id:
            continue
        try:
            await socket.send_json(message)
        except Exception:
            dead.append(socket)
    for socket in dead:
        connections.pop(socket, None)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "connections": len(connections)}


@app.get("/api/godmode/results")
def results(evaluation_id: str) -> dict[str, Any]:
    return {"evaluation_id": evaluation_id, "results": read_results(evaluation_id)}


@app.post("/api/godmode/submit")
async def submit(payload: VerifiedSubmit) -> dict[str, Any]:
    if not payload.session_id.strip() or not payload.evaluation_id.strip():
        raise HTTPException(status_code=400, detail="Missing DRES session or evaluation ID")
    dres_url = clean_external_url(payload.dres_url)
    time_ms = int(round(payload.timestamp * 1000))
    body = {"answerSets": [{"answers": [{
        "mediaItemName": payload.video_id.strip(), "start": time_ms, "end": time_ms,
        "text": None, "mediaItemCollectionName": None,
    }]}]}
    try:
        response = await asyncio.to_thread(
            requests.post,
            f"{dres_url}/api/v2/submit/{payload.evaluation_id}",
            params={"session": payload.session_id}, json=body, timeout=15,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"DRES submit connection failed: {exc}") from exc
    verdict = normalize_dres_verdict(response)
    if verdict.get("status") != "correct":
        return verdict

    item = safe_result(payload)
    with connect_db() as db:
        db.execute(
            "INSERT OR REPLACE INTO verified_results VALUES (?, ?, ?, ?, ?, ?)",
            (payload.evaluation_id, payload.video_id, payload.frame_id, payload.timestamp,
             json.dumps(item, ensure_ascii=False), item["verified_at"]),
        )
        db.execute("""
            DELETE FROM verified_results WHERE rowid IN (
                SELECT rowid FROM verified_results WHERE evaluation_id=?
                ORDER BY verified_at DESC LIMIT -1 OFFSET ?
            )
        """, (payload.evaluation_id, MAX_RESULTS))
    await broadcast({"type": "verified_result", "evaluation_id": payload.evaluation_id, "result": item})
    return {**verdict, "verified_result": item}


@app.websocket("/ws")
async def websocket_endpoint(socket: WebSocket, evaluation_id: str):
    await socket.accept()
    connections[socket] = evaluation_id
    await socket.send_json({"type": "snapshot", "evaluation_id": evaluation_id, "results": read_results(evaluation_id)})
    try:
        while True:
            await socket.receive_text()
    except WebSocketDisconnect:
        connections.pop(socket, None)
