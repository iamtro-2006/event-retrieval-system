from __future__ import annotations

from typing import Any

import requests

DRES_HEADERS = {"ngrok-skip-browser-warning": "true"}


class DresClient:
    """Adapter around the external DRES HTTP API.

    Holds a base URL and optional session. Each method maps to one DRES
    endpoint; verdict normalisation is a pure static method so it can be
    unit-tested without a network.
    """

    def __init__(self, base_url: str) -> None:
        cleaned = str(base_url or "").strip().rstrip("/")
        if not cleaned:
            raise ValueError("Missing DRES URL")
        if not (cleaned.startswith("http://") or cleaned.startswith("https://")):
            raise ValueError("DRES URL must start with http:// or https://")
        self.base_url = cleaned

    def login(self, username: str, password: str) -> dict[str, Any]:
        session = requests.Session()
        session.headers.update(DRES_HEADERS)

        login_res = session.post(
            f"{self.base_url}/api/v2/login",
            json={"username": username, "password": password},
            timeout=15,
        )
        if not login_res.ok:
            try:
                err_desc = login_res.json().get("description", "Login failed")
            except Exception:
                err_desc = login_res.text or "Login failed"
            raise RuntimeError(err_desc)

        sess_res = session.get(f"{self.base_url}/api/v2/user/session", timeout=15)
        if not sess_res.ok:
            raise RuntimeError("Cannot fetch DRES session")

        session_id = sess_res.text.strip().strip('"')
        evaluations = self.fetch_evaluations(session_id)
        evaluation_id = self.pick_active_evaluation_id(evaluations)

        return {
            "status": "ok",
            "session_id": session_id,
            "evaluation_id": evaluation_id,
            "evaluations": evaluations,
            "user": login_res.json() if login_res.text else {},
        }

    def fetch_evaluations(self, session_id: str) -> list[dict]:
        try:
            res = requests.get(
                f"{self.base_url}/api/v2/client/evaluation/list",
                params={"session": session_id},
                headers=DRES_HEADERS,
                timeout=10,
            )
            if res.ok:
                return res.json()
            return []
        except Exception:
            return []

    @staticmethod
    def pick_active_evaluation_id(evaluations: list) -> str | None:
        if not evaluations or not isinstance(evaluations, list):
            return None
        for ev in evaluations:
            if ev.get("status") == "ACTIVE":
                return ev.get("id")
        return evaluations[0].get("id")

    def submit(
        self,
        session_id: str,
        evaluation_id: str | None,
        video_id: str,
        frame_id: int,
        timestamp: float | None = None,
    ) -> dict[str, Any]:
        if not session_id.strip():
            raise ValueError("Missing active session_id")

        if not evaluation_id:
            evals = self.fetch_evaluations(session_id)
            evaluation_id = self.pick_active_evaluation_id(evals)

        if not evaluation_id:
            raise ValueError("No active DRES evaluation found to submit")

        if timestamp is not None and timestamp >= 0:
            time_ms = int(round(timestamp * 1000))
        else:
            time_ms = int(frame_id)

        submit_payload = {
            "answerSets": [
                {
                    "answers": [
                        {
                            "mediaItemName": str(video_id).strip(),
                            "start": time_ms,
                            "end": time_ms,
                            "text": None,
                            "mediaItemCollectionName": None,
                        }
                    ]
                }
            ]
        }
        submit_url = f"{self.base_url}/api/v2/submit/{evaluation_id}"
        params = {"session": session_id}

        res = requests.post(submit_url, params=params, json=submit_payload, timeout=15)
        return self.normalize_verdict(res)

    @staticmethod
    def normalize_verdict(response: requests.Response) -> dict[str, Any]:
        try:
            data = response.json()
        except Exception:
            data = {"raw": response.text or ""}

        raw_text = (response.text or "").lower()
        if response.status_code == 412:
            return {"status": "wrong", "message": "Wrong Answer", "data": data}
        if not response.ok:
            return {
                "status": "error",
                "message": data.get(
                    "description", f"HTTP Error {response.status_code}"
                ),
                "data": data,
            }
        if response.status_code == 202:
            return {
                "status": "pending",
                "message": "Submitted, waiting for verdict",
                "data": data,
            }

        verdict = str(data.get("submission", "")).upper()
        if "CORRECT" in verdict or "CORRECT" in raw_text:
            return {"status": "correct", "message": "Correct!", "data": data}
        if "WRONG" in verdict or "WRONG" in raw_text:
            return {"status": "wrong", "message": "Wrong Answer", "data": data}
        return {"status": "pending", "message": "Submitted", "data": data}
