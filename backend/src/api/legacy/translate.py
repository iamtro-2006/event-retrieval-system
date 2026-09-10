"""Google-only query translation for the legacy search endpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.translation import get_translator
from src.translation.google_translator import GoogleCloudTranslator


def translate_query_if_needed(query: str, use_translate: bool, cfg: dict[str, Any], backend_dir: Path, api_key: str | None = None) -> str:
    """Translate query text using Google Cloud Translation."""
    if not use_translate:
        return query
    translate_cfg = cfg.get("translate", {})
    source = translate_cfg.get("source", "vi")
    target = translate_cfg.get("target", "en")
    translator = GoogleCloudTranslator(api_key) if api_key else get_translator(cfg, backend_dir)
    return translator.translate(query, source=source, target=target)
