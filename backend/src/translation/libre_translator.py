"""
Translation backend backed by a LibreTranslate server (self-hosted or the
managed https://libretranslate.com service).

API reference: https://docs.libretranslate.com/api/operations/translate/

    POST {base_url}/translate
    form-data: q, source, target, format, api_key (optional)
    -> {"translatedText": "..."}

Drop-in replacement for deep_translator.GoogleTranslator / HyMT2Translator,
so it can be swapped into main.py's `translate_query_if_needed` purely via
config (`translate_agent: libre` in configs/app.yaml).
"""

from __future__ import annotations

import threading
from typing import Optional

import requests

from .base_translator import BaseTranslator


class LibreTranslator(BaseTranslator):
    """Thin singleton client for a LibreTranslate server's /translate endpoint."""

    _instance: Optional["LibreTranslator"] = None
    _lock = threading.Lock()

    def __init__(
        self,
        base_url: str = "https://libretranslate.com",
        api_key: str | None = None,
        format: str = "text",
        timeout: float = 15.0,
    ):
        # LibreTranslate's own examples post to e.g. http://localhost:5000/translate
        self._endpoint = f"{base_url.rstrip('/')}/translate"
        self._api_key = api_key
        self._format = format
        self._timeout = timeout
        self._session = requests.Session()

    @classmethod
    def get_instance(
        cls,
        base_url: str = "https://libretranslate.com",
        api_key: str | None = None,
        format: str = "text",
        timeout: float = 15.0,
    ) -> "LibreTranslator":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(
                        base_url=base_url,
                        api_key=api_key,
                        format=format,
                        timeout=timeout,
                    )
        return cls._instance

    def translate(self, text: str, source: str = "vi", target: str = "en") -> str:
        """Translate `text` via the LibreTranslate REST API.

        Mirrors deep_translator.GoogleTranslator(source, target).translate(text).
        """
        if not text or not text.strip():
            return text

        payload = {
            "q": text,
            "source": source,
            "target": target,
            "format": self._format,
        }
        if self._api_key:
            payload["api_key"] = self._api_key

        response = self._session.post(
            self._endpoint, data=payload, timeout=self._timeout
        )
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            raise RuntimeError(f"LibreTranslate error: {data['error']}")

        return data.get("translatedText", text)
