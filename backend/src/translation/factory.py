"""Build the single supported translation backend: Google Cloud Translation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .base_translator import BaseTranslator
from .google_translator import GoogleCloudTranslator


def get_translator(cfg: dict[str, Any], backend_dir: Path) -> BaseTranslator:
    """Return a Google translator configured from YAML/environment."""
    del backend_dir  # Kept in the public signature for existing callers.
    translate_cfg = cfg.get("translate", {}) or {}
    google_cfg = translate_cfg.get("google", {}) or {}
    api_key = os.getenv("GOOGLE_TRANSLATE_API_KEY", str(google_cfg.get("api_key") or ""))
    timeout = float(
        os.getenv("GOOGLE_TRANSLATE_TIMEOUT", str(google_cfg.get("timeout", 15)))
    )
    return GoogleCloudTranslator(api_key=api_key, timeout=timeout)
