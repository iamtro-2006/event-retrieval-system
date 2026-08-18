import os
from pathlib import Path

from fastapi import APIRouter, HTTPException

from src.api.schemas.translate import TranslateRequest, TranslateResponse
from src.translation.envit5_translator import EnviT5Translator
from src.translation.google_translator import GoogleCloudTranslator
from src.translation.llm_translator import LLMTranslator

router = APIRouter(prefix="/api/translate", tags=["translation"])


@router.post("", response_model=TranslateResponse)
def translate(payload: TranslateRequest) -> TranslateResponse:
    provider = payload.provider.strip().lower()
    key = payload.api_key or (os.getenv("GOOGLE_TRANSLATE_API_KEY", "") if provider == "google" else os.getenv("LLM_TRANSLATE_API_KEY", os.getenv("CKEY_API_KEY", "")))
    try:
        if provider == "google":
            translator = GoogleCloudTranslator(key)
        elif provider == "llm":
            if not key:
                raise ValueError("LLM API key is missing")
            translator = LLMTranslator(api_key=key, base_url=os.getenv("LLM_TRANSLATE_BASE_URL", "https://api.xah.io/v1"), model=os.getenv("LLM_TRANSLATE_MODEL", "claude-haiku-4.5"))
        elif provider == "envit5":
            translator = EnviT5Translator.get_instance()
        else:
            raise ValueError("provider must be google, llm, or envit5")
        translated = translator.translate(payload.text, payload.source, payload.target)
        return TranslateResponse(text=payload.text, translated_text=translated, provider=provider)
    except Exception as exc:
        if provider in {"google", "llm"}:
            try:
                fallback = EnviT5Translator.get_instance().translate(payload.text, payload.source, payload.target)
                return TranslateResponse(text=payload.text, translated_text=fallback, provider="envit5", fallback=True)
            except Exception:
                pass
        raise HTTPException(status_code=502, detail=f"Translate failed: {type(exc).__name__}: {exc}") from exc
