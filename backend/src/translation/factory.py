"""
Factory that picks a translation backend based on `configs/app.yaml`'s
`translate_agent` field, so `main.py` never has to know which concrete
translator class is behind `translate_query_if_needed`.

configs/app.yaml:

    translate_agent: hymt2   # or "libre"

    translate:
      enabled_default: false
      source: vi
      target: en

      hymt2:
        model_path: models/Hy-MT2-1.8B-2Bit.gguf   # overridable via HY_MT2_MODEL_PATH
        n_gpu_layers: 0                              # overridable via HY_MT2_N_GPU_LAYERS

      libre:
        base_url: http://localhost:5000              # overridable via LIBRE_TRANSLATE_URL
        api_key: null                                 # overridable via LIBRE_TRANSLATE_API_KEY
        format: text
        timeout: 15
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .base_translator import BaseTranslator
from .hy_mt2_translator import HyMT2Translator
from .libre_translator import LibreTranslator

_SUPPORTED_AGENTS = ("hymt2", "libre")


def get_translator(cfg: dict[str, Any], backend_dir: Path) -> BaseTranslator:
    """Return the singleton translator instance selected by `translate_agent`.

    Args:
        cfg: The full parsed `app.yaml` config (i.e. `CFG` in main.py).
        backend_dir: Backend root directory, used to resolve relative model
            paths (mirrors `resolve_backend_path` in main.py).

    Raises:
        ValueError: If `translate_agent` is set to an unsupported value.
    """
    agent = str(cfg.get("translate_agent", "hymt2")).strip().lower()
    translate_cfg = cfg.get("translate", {}) or {}

    if agent == "hymt2":
        hymt2_cfg = translate_cfg.get("hymt2", {}) or {}
        default_model_path = backend_dir / "models" / "Hy-MT2-1.8B-2Bit.gguf"
        model_path = os.getenv(
            "HY_MT2_MODEL_PATH",
            str(hymt2_cfg.get("model_path") or default_model_path),
        )
        n_gpu_layers = int(
            os.getenv(
                "HY_MT2_N_GPU_LAYERS",
                str(hymt2_cfg.get("n_gpu_layers", 0)),
            )
        )
        return HyMT2Translator.get_instance(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
        )

    if agent == "libre":
        libre_cfg = translate_cfg.get("libre", {}) or {}
        base_url = os.getenv(
            "LIBRE_TRANSLATE_URL", libre_cfg.get("base_url", "http://localhost:5000")
        )
        api_key = os.getenv("LIBRE_TRANSLATE_API_KEY", libre_cfg.get("api_key"))
        format_ = libre_cfg.get("format", "text")
        timeout = float(libre_cfg.get("timeout", 15))
        return LibreTranslator.get_instance(
            base_url=base_url,
            api_key=api_key,
            format=format_,
            timeout=timeout,
        )

    raise ValueError(
        f"Unsupported translate_agent: '{agent}'. Supported: {_SUPPORTED_AGENTS}"
    )
