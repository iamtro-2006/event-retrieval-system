"""`POST /api/dres/login` + `POST /api/dres/submit` GỐC — port nguyên từ
`main.py` cũ. Không phụ thuộc `RetrievalSystem` (chỉ gọi ra DRES server qua
`requests`), nên không cần `Depends(get_legacy_system)`.
"""

from __future__ import annotations

import requests
from fastapi import APIRouter, HTTPException

from src.api.legacy.dres_client import (
    DRES_HEADERS,
    clean_external_url,
    fetch_dres_evaluations,
    normalize_dres_verdict,
    pick_active_evaluation_id,
)
from src.api.schemas.legacy import DresLoginRequest, DresSubmitRequest

router = APIRouter(tags=["legacy-dres"])


@router.post("/api/dres/login")
def dres_login(payload: DresLoginRequest):
    """Authenticate with the DRES server and retrieve session details."""
    dres_url = clean_external_url(payload.dres_url)
    if not payload.username.strip() or not payload.password:
        raise HTTPException(status_code=400, detail="Missing username or password")

    session = requests.Session()
    session.headers.update(DRES_HEADERS)

    try:
        login_res = session.post(f"{dres_url}/api/v2/login", json={"username": payload.username, "password": payload.password}, timeout=15)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"DRES login connection failed: {e}")

    if not login_res.ok:
        try:
            err_desc = login_res.json().get("description", "Login failed")
        except Exception:
            err_desc = login_res.text or "Login failed"
        raise HTTPException(status_code=login_res.status_code, detail=err_desc)

    login_data = login_res.json() if login_res.text else {}
    session_id = str(login_data.get("sessionId") or login_data.get("session_id") or "").strip()
    # Compatibility fallback for older DRES deployments.
    if not session_id:
        try:
            sess_res = session.get(f"{dres_url}/api/v2/user/session", timeout=15)
        except requests.RequestException as e:
            raise HTTPException(status_code=502, detail=f"DRES session fetch failed: {e}")
        if not sess_res.ok:
            raise HTTPException(status_code=sess_res.status_code, detail="Cannot fetch DRES session")
        session_id = sess_res.text.strip().strip('"')
    evaluations = fetch_dres_evaluations(dres_url, session_id)

    return {
        "status": "ok", "session_id": session_id,
        "evaluation_id": pick_active_evaluation_id(evaluations),
        "evaluations": evaluations,
        "user": login_data
    }


@router.post("/api/dres/submit")
def dres_submit(payload: DresSubmitRequest):
    """Submit KIS, ordered TRAKE, or QA answers to the active evaluation."""
    dres_url = clean_external_url(payload.dres_url)
    if not payload.session_id.strip():
        raise HTTPException(status_code=400, detail="Missing active session_id")

    evaluation_id = payload.evaluation_id or pick_active_evaluation_id(fetch_dres_evaluations(dres_url, payload.session_id))
    if not evaluation_id:
        raise HTTPException(status_code=400, detail="No active DRES evaluation found")

    task = payload.task.strip().lower()
    if task not in {"kis", "trake", "qa"}:
        raise HTTPException(status_code=400, detail="task must be kis, trake, or qa")

    items = list(payload.items)
    if not items and payload.video_id:
        items = [{"video_id": payload.video_id, "frame_id": payload.frame_id, "timestamp": payload.timestamp}]
    if not items:
        raise HTTPException(status_code=400, detail="Submission queue is empty")

    def item_time_ms(item: dict) -> int:
        timestamp = item.get("timestamp")
        if timestamp is not None and float(timestamp) >= 0:
            return int(round(float(timestamp) * 1000))
        return max(0, int(item.get("frame_id") or 0))

    def media_answer(item: dict) -> dict:
        video_id = str(item.get("video_id") or "").strip()
        if not video_id:
            raise HTTPException(status_code=400, detail="Every queue item needs video_id")
        time_ms = item_time_ms(item)
        return {"mediaItemName": video_id, "start": time_ms, "end": time_ms}

    if task == "kis":
        # DRES KIS accepts one located media item.
        submit_payload = {"answerSets": [{"answers": [media_answer(items[0])]}]}
    elif task == "trake":
        # TRAKE is one ordered frame sequence from one video.
        video_ids = {str(item.get("video_id") or "").strip() for item in items}
        if "" in video_ids or len(video_ids) != 1:
            raise HTTPException(status_code=400, detail="TRAKE requires all queued frames to belong to one video")
        video_id = next(iter(video_ids))
        frame_ids = ",".join(str(max(0, int(item.get("frame_id") or 0))) for item in items)
        submit_payload = {"answerSets": [{"answers": [{"text": f"TR-{video_id}-{frame_ids}"}]}]}
    else:
        answer = (payload.answer or "").strip()
        if not answer:
            raise HTTPException(status_code=400, detail="QA requires a non-empty answer")
        item = items[0]
        video_id = str(item.get("video_id") or "").strip()
        if not video_id:
            raise HTTPException(status_code=400, detail="QA queue item needs video_id")
        text = f"QA-{answer}-{video_id}-{item_time_ms(item)}"
        submit_payload = {"answerSets": [{"answers": [{"text": text}]}]}

    try:
        res = requests.post(f"{dres_url}/api/v2/submit/{evaluation_id}", params={"session": payload.session_id}, json=submit_payload, timeout=15)
        return normalize_dres_verdict(res)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"DRES submit connection failed: {e}")
