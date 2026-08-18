from __future__ import annotations

import os
import threading
from typing import List

import requests

from .base_translator import BaseTranslator


class GoogleCloudTranslator(BaseTranslator):
    """Google Cloud Translation Basic v2 client."""

    def __init__(self, api_key: str, timeout: float = 15.0):
        self.api_key = api_key
        self.timeout = timeout
        self._session = requests.Session()
        self._lock = threading.Lock()

    def translate_batch(self, texts: List[str], source="vi", target="en") -> List[str]:
        if not self.api_key:
            raise ValueError("Google Translate API key is missing")
        indices = [i for i, value in enumerate(texts) if value and value.strip()]
        result = list(texts)
        if not indices:
            return result
        response = self._session.post(
            "https://translation.googleapis.com/language/translate/v2",
            params={"key": self.api_key},
            json={"q": [texts[i].strip() for i in indices], "source": source, "target": target, "format": "text"},
            timeout=self.timeout,
        )
        if not response.ok:
            raise RuntimeError(f"Google Translate failed ({response.status_code}): {response.text[:300]}")
        translations = response.json().get("data", {}).get("translations", [])
        if len(translations) != len(indices):
            raise RuntimeError("Google Translate returned an unexpected response")
        for i, item in zip(indices, translations):
            result[i] = item.get("translatedText", "").strip()
        return result

    def translate(self, text, source="vi", target="en"):
        if not text or not text.strip():
            return text
        return self.translate_batch([text], source, target)[0]
