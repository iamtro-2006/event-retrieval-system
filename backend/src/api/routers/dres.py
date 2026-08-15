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
        "user": login_res.json() if login_res.text else {}
    }


@router.post("/api/dres/submit")
def dres_submit(payload: DresSubmitRequest):
    """Submit a retrieval result to the active DRES evaluation."""
    dres_url = clean_external_url(payload.dres_url)
    if not payload.session_id.strip():
        raise HTTPException(status_code=400, detail="Missing active session_id")

    evaluation_id = payload.evaluation_id or pick_active_evaluation_id(fetch_dres_evaluations(dres_url, payload.session_id))
    if not evaluation_id:
        raise HTTPException(status_code=400, detail="No active DRES evaluation found")

    time_ms = int(round(payload.timestamp * 1000)) if payload.timestamp is not None and payload.timestamp >= 0 else int(payload.frame_id)
    submit_payload = {
        "answerSets": [{
            "answers": [{
                "mediaItemName": str(payload.video_id).strip(),
                "start": time_ms, "end": time_ms,
                "text": None, "mediaItemCollectionName": None
            }]
        }]
    }

    try:
        res = requests.post(f"{dres_url}/api/v2/submit/{evaluation_id}", params={"session": payload.session_id}, json=submit_payload, timeout=15)
        return normalize_dres_verdict(res)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"DRES submit connection failed: {e}")
