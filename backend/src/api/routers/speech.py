"""`POST /api/speech/transcribe` GỐC — port nguyên từ `main.py` cũ (dùng
Faster-Whisper để phiên âm audio upload, KHÔNG liên quan `RetrievalSystem`).

Model Whisper lazy-load + cache thread-safe giống hệt bản gốc (biến module
`_speech_model` + lock), chỉ khác chỗ lấy `speech` config: đọc qua
`request.app.state.cfg` (đã load 1 lần lúc lifespan) thay vì biến global
`CFG` ở top-level file `main.py` cũ.
"""

from __future__ import annotations

import threading
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from fastapi import APIRouter, Depends, File, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from faster_whisper import WhisperModel

from src.api.legacy.deps import get_cfg

router = APIRouter(tags=["legacy-speech"])

_speech_model: WhisperModel | None = None
_speech_model_lock = threading.Lock()


def get_speech_model(cfg: dict[str, Any]) -> WhisperModel:
    """Lazy-load and cache the Whisper speech-to-text model thread-safely."""
    global _speech_model
    if _speech_model is None:
        with _speech_model_lock:
            if _speech_model is None:
                speech_cfg = cfg.get("speech", {})
                _speech_model = WhisperModel(
                    speech_cfg.get("model_size", "base"),
                    device=speech_cfg.get("device", "cpu"),
                    compute_type=speech_cfg.get("compute_type", "int8"),
                )
    return _speech_model


@router.post("/api/speech/transcribe")
async def transcribe_speech(
    request: Request,
    file: UploadFile = File(...),
    cfg: dict[str, Any] = Depends(get_cfg),
):
    """Transcribe uploaded audio using the Faster-Whisper model."""
    suffix = Path(file.filename or "audio.webm").suffix or ".webm"

    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        def _run_whisper():
            return get_speech_model(cfg).transcribe(tmp_path, beam_size=1, language="vi", vad_filter=True)

        segments, info = await run_in_threadpool(_run_whisper)
        text = " ".join(seg.text.strip() for seg in segments).strip()

        return {"text": text, "language": info.language, "duration": info.duration}
    finally:
        Path(tmp_path).unlink(missing_ok=True)
