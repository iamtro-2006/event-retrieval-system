# ============================================================
# DRES PROXY PATCH FOR backend/main.py
# ============================================================
# 1) Add these imports near the top:
#
# import requests
# from fastapi import Request
#
# Your current file uses Request in normalize_double_slash but does not import it.
#
# 2) Add these Pydantic schemas near SearchRequest:

class DresLoginRequest(BaseModel):
    dres_url: str
    username: str
    password: str


class DresEvaluationRequest(BaseModel):
    dres_url: str
    session_id: str | None = None


class DresSubmitRequest(BaseModel):
    dres_url: str
    session_id: str
    evaluation_id: str | None = None
    video_id: str
    frame_id: int
    timestamp: float | None = None


# 3) Add these helper functions before API routes:

def clean_external_url(url: str) -> str:
    cleaned = str(url or "").strip().rstrip("/")

    if not cleaned:
        raise HTTPException(status_code=400, detail="Missing DRES URL")

    return cleaned


def parse_dres_response(response: requests.Response) -> dict[str, Any]:
    text = response.text or ""

    try:
        data = response.json()
    except Exception:
        data = {"raw": text}

    normalized = f"{text} {data}".lower()

    if response.status_code == 412:
        return {
            "status": "wrong",
            "message": "Wrong",
            "data": data,
        }

    if not response.ok:
        return {
            "status": "warning",
            "message": f"DRES request failed: HTTP {response.status_code}",
            "data": data,
        }

    if "correct" in normalized or '"correct": true' in normalized:
        return {
            "status": "correct",
            "message": "Correct",
            "data": data,
        }

    if "wrong" in normalized or "incorrect" in normalized or '"correct": false' in normalized:
        return {
            "status": "wrong",
            "message": "Wrong",
            "data": data,
        }

    return {
        "status": "pending",
        "message": "Submitted",
        "data": data,
    }


def get_active_evaluation_id(dres_url: str, session_id: str | None = None) -> str | None:
    params = {}
    if session_id:
        params["session"] = session_id

    response = requests.get(
        f"{dres_url}/api/v2/evaluation/state/list",
        params=params,
        timeout=15,
    )

    if not response.ok:
        return None

    try:
        states = response.json()
    except Exception:
        return None

    if not isinstance(states, list):
        return None

    for item in states:
        if not isinstance(item, dict):
            continue

        evaluation_status = str(item.get("evaluationStatus", "")).upper()
        task_status = str(item.get("taskStatus", "")).upper()

        if evaluation_status == "ACTIVE" and task_status == "RUNNING":
            return item.get("evaluationId")

    return None


# 4) Add these routes near the bottom of main.py:

@app.post("/api/dres/login")
def dres_login(payload: DresLoginRequest):
    dres_url = clean_external_url(payload.dres_url)

    session = requests.Session()

    login_response = session.post(
        f"{dres_url}/api/v2/login",
        json={
            "username": payload.username,
            "password": payload.password,
        },
        timeout=15,
    )

    if not login_response.ok:
        raise HTTPException(
            status_code=login_response.status_code,
            detail=login_response.text or "DRES login failed",
        )

    user_payload: Any
    try:
        user_payload = login_response.json()
    except Exception:
        user_payload = {"raw": login_response.text}

    session_response = session.get(
        f"{dres_url}/api/v2/user/session",
        timeout=15,
    )

    if not session_response.ok:
        raise HTTPException(
            status_code=session_response.status_code,
            detail=session_response.text or "Cannot get DRES session",
        )

    session_id = session_response.text.strip()

    return {
        "status": "ok",
        "session_id": session_id,
        "user": user_payload,
        "active_evaluation_id": get_active_evaluation_id(dres_url, session_id),
    }


@app.post("/api/dres/evaluations")
def dres_evaluations(payload: DresEvaluationRequest):
    dres_url = clean_external_url(payload.dres_url)

    params = {}
    if payload.session_id:
        params["session"] = payload.session_id

    response = requests.get(
        f"{dres_url}/api/v2/evaluation/state/list",
        params=params,
        timeout=15,
    )

    if not response.ok:
        raise HTTPException(
            status_code=response.status_code,
            detail=response.text or "Cannot get DRES evaluations",
        )

    return {
        "evaluations": response.json(),
    }


@app.post("/api/dres/submit")
def dres_submit(payload: DresSubmitRequest):
    dres_url = clean_external_url(payload.dres_url)

    if not payload.session_id:
        raise HTTPException(status_code=400, detail="Missing DRES session_id")

    evaluation_id = payload.evaluation_id or get_active_evaluation_id(
        dres_url=dres_url,
        session_id=payload.session_id,
    )

    if evaluation_id:
        timestamp_ms = round(float(payload.timestamp or 0) * 1000)

        submit_body = {
            "answerSets": [
                {
                    "answers": [
                        {
                            "mediaItemName": payload.video_id,
                            "start": timestamp_ms,
                            "end": timestamp_ms,
                        }
                    ]
                }
            ]
        }

        response = requests.post(
            f"{dres_url}/api/v2/submit/{evaluation_id}",
            params={
                "session": payload.session_id,
            },
            json=submit_body,
            timeout=15,
        )

        result = parse_dres_response(response)
        result["evaluation_id"] = evaluation_id
        return result

    # Fallback legacy KIS endpoint.
    response = requests.get(
        f"{dres_url}/api/v1/submit",
        params={
            "item": payload.video_id,
            "frame": payload.frame_id,
            "session": payload.session_id,
        },
        timeout=15,
    )

    result = parse_dres_response(response)
    result["evaluation_id"] = None
    return result
