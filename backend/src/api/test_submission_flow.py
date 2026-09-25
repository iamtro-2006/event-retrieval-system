from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from src.api.routers.dres import dres_submit
from src.api.schemas.legacy import DresSubmitRequest


def test_kis_submission_uses_the_first_queued_frame_timestamp(monkeypatch):
    captured = {}

    def fake_post(url, *, params, json, timeout):
        captured.update(url=url, params=params, json=json, timeout=timeout)
        return SimpleNamespace(
            ok=True,
            status_code=200,
            text='{"submission":"CORRECT"}',
            json=lambda: {"submission": "CORRECT"},
        )

    monkeypatch.setattr("src.api.routers.dres.requests.post", fake_post)
    response = dres_submit(DresSubmitRequest(
        dres_url="https://dres.example",
        session_id="session-1",
        evaluation_id="evaluation-1",
        task="kis",
        items=[
            {"video_id": "L01_V001", "frame_id": 10, "timestamp": 1.25},
            {"video_id": "L01_V002", "frame_id": 20, "timestamp": 2.5},
        ],
    ))

    assert response["status"] == "correct"
    assert captured["params"] == {"session": "session-1"}
    assert captured["json"] == {
        "answerSets": [{"answers": [{"mediaItemName": "L01_V001", "start": 1250, "end": 1250}]}]
    }


def test_trake_rejects_a_queue_containing_multiple_videos():
    with pytest.raises(HTTPException, match="one video") as error:
        dres_submit(DresSubmitRequest(
            dres_url="https://dres.example",
            session_id="session-1",
            evaluation_id="evaluation-1",
            task="trake",
            items=[
                {"video_id": "L01_V001", "frame_id": 10, "timestamp": 1.0},
                {"video_id": "L01_V002", "frame_id": 20, "timestamp": 2.0},
            ],
        ))

    assert error.value.status_code == 400
