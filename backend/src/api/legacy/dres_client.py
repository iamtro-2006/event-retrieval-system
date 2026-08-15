"""Helpers gọi ra DRES server (Video Retrieval evaluation server) — port
nguyên từ `main.py` gốc, dùng bởi `src/api/routers/dres.py`.
"""

from __future__ import annotations

from typing import Any

import requests
from fastapi import HTTPException

DRES_HEADERS = {"ngrok-skip-browser-warning": "true"}


def clean_external_url(url: str) -> str:
    """Validate and clean an external URL."""
    cleaned = str(url or "").strip().rstrip("/")
    if not cleaned:
        raise HTTPException(status_code=400, detail="Missing URL")
    if not (cleaned.startswith("http://") or cleaned.startswith("https://")):
        raise HTTPException(status_code=400, detail="URL must start with http:// or https://")
    return cleaned


def fetch_dres_evaluations(dres_url: str, session_id: str) -> list[dict]:
    """Fetch active evaluations from the DRES server."""
    try:
        res = requests.get(
            f"{dres_url}/api/v2/client/evaluation/list",
            params={"session": session_id},
            headers=DRES_HEADERS,
            timeout=10,
        )
        return res.json() if res.ok else []
    except Exception:
        return []


def pick_active_evaluation_id(evaluations: list) -> str | None:
    """Select the active evaluation ID from a list of evaluations."""
    if not evaluations or not isinstance(evaluations, list):
        return None
    for ev in evaluations:
        if ev.get("status") == "ACTIVE":
            return ev.get("id")
    return evaluations[0].get("id")


def normalize_dres_verdict(response: requests.Response) -> dict[str, Any]:
    """Parse and normalize the verdict response from DRES."""
    try:
        data = response.json()
    except Exception:
        data = {"raw": response.text or ""}

    raw_text = (response.text or "").lower()
    if response.status_code == 412:
        return {"status": "wrong", "message": "Wrong Answer", "data": data}
    if not response.ok:
        return {"status": "error", "message": data.get("description", f"HTTP Error {response.status_code}"), "data": data}
    if response.status_code == 202:
        return {"status": "pending", "message": "Submitted, waiting for verdict", "data": data}

    verdict = str(data.get("submission", "")).upper()
    if "CORRECT" in verdict or "CORRECT" in raw_text:
        return {"status": "correct", "message": "Correct!", "data": data}
    if "WRONG" in verdict or "WRONG" in raw_text:
        return {"status": "wrong", "message": "Wrong Answer", "data": data}
    return {"status": "pending", "message": "Submitted", "data": data}
