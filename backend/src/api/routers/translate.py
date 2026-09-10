import os
from pathlib import Path

from fastapi import APIRouter, HTTPException

from src.api.schemas.translate import TranslateRequest, TranslateResponse
from src.translation.google_translator import GoogleCloudTranslator

router = APIRouter(prefix="/api/translate", tags=["translation"])


@router.post("", response_model=TranslateResponse)
def translate(payload: TranslateRequest) -> TranslateResponse:
    key = payload.api_key or os.getenv("GOOGLE_TRANSLATE_API_KEY", "")
    try:
        translator = GoogleCloudTranslator(key)
        translated = translator.translate(payload.text, payload.source, payload.target)
        return TranslateResponse(text=payload.text, translated_text=translated, provider="google")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Translate failed: {type(exc).__name__}: {exc}") from exc
