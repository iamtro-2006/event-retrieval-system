"""
Factory that returns the configured translator singleton based on
`configs/app.yaml`'s `translate_agent` field, so `main.py` never has to
instantiate a concrete translator class directly.

configs/app.yaml:

    translate_agent: envit5   # "envit5" (local GPU model) or "llm" (remote API)

    translate:
      enabled_default: false
      source: vi
      target: en

      envit5:
        model_name: VietAI/envit5-translation   # overridable via ENVIT5_MODEL_NAME
        device: auto                             # overridable via ENVIT5_DEVICE ("cpu"/"cuda"/"auto")
        max_length: 512                          # overridable via ENVIT5_MAX_LENGTH
        num_beams: 1                             # overridable via ENVIT5_NUM_BEAMS
        dtype: auto                              # overridable via ENVIT5_DTYPE ("auto"/"fp16"/"bf16"/"fp32")
        max_new_tokens: 96                       # overridable via ENVIT5_MAX_NEW_TOKENS

      llm:
        api_key: null                            # overridable via LLM_TRANSLATE_API_KEY / CKEY_API_KEY
        base_url: https://api.xah.io/v1          # overridable via LLM_TRANSLATE_BASE_URL
        model: claude-haiku-4.5                  # overridable via LLM_TRANSLATE_MODEL
        timeout: 15
        max_workers: 8                           # concurrent request cap for translate_batch()
        temperature: 0.0
        max_output_tokens: 128
        persona: ""                              # free-text personalization block injected into the system prompt
        cache_size: 2048
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .base_translator import BaseTranslator
from .envit5_translator import EnviT5Translator
from .llm_translator import LLMTranslator

_SUPPORTED_AGENTS = ("envit5", "llm")


def get_translator(cfg: dict[str, Any], backend_dir: Path) -> BaseTranslator:
    """Return the singleton translator instance selected by `translate_agent`.

    Args:
        cfg: The full parsed `app.yaml` config (i.e. `CFG` in main.py).
        backend_dir: Backend root directory (kept for signature compatibility
            with callers; unused by both current backends).

    Raises:
        ValueError: If `translate_agent` is set to an unsupported value.
    """
    agent = str(cfg.get("translate_agent", "envit5")).strip().lower()
    translate_cfg = cfg.get("translate", {}) or {}

    if agent == "envit5":
        envit5_cfg = translate_cfg.get("envit5", {}) or {}
        model_name = os.getenv(
            "ENVIT5_MODEL_NAME", str(envit5_cfg.get("model_name", "VietAI/envit5-translation"))
        )
        device = os.getenv("ENVIT5_DEVICE", str(envit5_cfg.get("device", "auto")))
        max_length = int(os.getenv("ENVIT5_MAX_LENGTH", str(envit5_cfg.get("max_length", 512))))
        num_beams = int(os.getenv("ENVIT5_NUM_BEAMS", str(envit5_cfg.get("num_beams", 1))))
        dtype = os.getenv("ENVIT5_DTYPE", str(envit5_cfg.get("dtype", "auto")))
        max_new_tokens = int(
            os.getenv("ENVIT5_MAX_NEW_TOKENS", str(envit5_cfg.get("max_new_tokens", 96)))
        )

        return EnviT5Translator.get_instance(
            model_name=model_name,
            device=device,
            max_length=max_length,
            num_beams=num_beams,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
        )

    if agent == "llm":
        llm_cfg = translate_cfg.get("llm", {}) or {}
        api_key = os.getenv(
            "LLM_TRANSLATE_API_KEY", os.getenv("CKEY_API_KEY", llm_cfg.get("api_key") or "")
        )
        base_url = os.getenv(
            "LLM_TRANSLATE_BASE_URL", str(llm_cfg.get("base_url", "https://api.xah.io/v1"))
        )
        model = os.getenv("LLM_TRANSLATE_MODEL", str(llm_cfg.get("model", "claude-haiku-4.5")))
        timeout = float(os.getenv("LLM_TRANSLATE_TIMEOUT", str(llm_cfg.get("timeout", 15))))
        max_workers = int(
            os.getenv("LLM_TRANSLATE_MAX_WORKERS", str(llm_cfg.get("max_workers", 8)))
        )
        temperature = float(
            os.getenv("LLM_TRANSLATE_TEMPERATURE", str(llm_cfg.get("temperature", 0.0)))
        )
        max_output_tokens = int(
            os.getenv(
                "LLM_TRANSLATE_MAX_OUTPUT_TOKENS", str(llm_cfg.get("max_output_tokens", 128))
            )
        )
        persona = os.getenv("LLM_TRANSLATE_PERSONA", str(llm_cfg.get("persona", "")))
        cache_size = int(
            os.getenv("LLM_TRANSLATE_CACHE_SIZE", str(llm_cfg.get("cache_size", 2048)))
        )

        return LLMTranslator.get_instance(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=timeout,
            max_workers=max_workers,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            persona=persona,
            cache_size=cache_size,
        )

    raise ValueError(
        f"Unsupported translate_agent: '{agent}'. Supported: {_SUPPORTED_AGENTS}"
    )
