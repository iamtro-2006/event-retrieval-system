"""Dịch query vi->en cho endpoint `/api/search` cũ — port nguyên
`translate_query_if_needed` từ `main.py` gốc.

Dùng `src.translation.get_translator`, vốn trả về 1 singleton (xem
`EnviT5Translator.get_instance` / `LLMTranslator.get_instance`), nên gọi lại
hàm này ở mỗi request KHÔNG build lại model — an toàn để giữ nguyên style
"gọi trực tiếp" như bản gốc thay vì cache thêm 1 lớp nữa.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.translation import get_translator
from src.translation.google_translator import GoogleCloudTranslator
from src.translation.llm_translator import LLMTranslator


def translate_query_if_needed(query: str, use_translate: bool, cfg: dict[str, Any], backend_dir: Path, provider: str | None = None, api_key: str | None = None) -> str:
    """Translate query text using the configured translate agent (local EnviT5 or remote LLM API)."""
    if not use_translate:
        return query
    translate_cfg = cfg.get("translate", {})
    source = translate_cfg.get("source", "vi")
    target = translate_cfg.get("target", "en")
    translator = (
        GoogleCloudTranslator(api_key or "") if provider == "google"
        else LLMTranslator(api_key=api_key or "") if provider == "llm"
        else get_translator(cfg, backend_dir)
    )
    try:
        return translator.translate(query, source=source, target=target)
    except Exception:
        if provider in {"google", "llm"}:
            return get_translator({**cfg, "translate_agent": "envit5"}, backend_dir).translate(query, source=source, target=target)
        raise
