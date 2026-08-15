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


def translate_query_if_needed(query: str, use_translate: bool, cfg: dict[str, Any], backend_dir: Path) -> str:
    """Translate query text using the configured translate agent (local EnviT5 or remote LLM API)."""
    if not use_translate:
        return query
    translate_cfg = cfg.get("translate", {})
    source = translate_cfg.get("source", "vi")
    target = translate_cfg.get("target", "en")
    translator = get_translator(cfg, backend_dir)
    return translator.translate(query, source=source, target=target)
