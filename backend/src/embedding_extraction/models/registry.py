"""Central model registry / factory.

Every backend module exposes a `load(...) -> LoadedModel` function with the
same signature shape. This file just knows *which* backend module to call
for a given `backend` string, plus a small table of ready-made presets for
the models this project cares about so configs (embeddings.yaml,
indexing.yaml) can stay short (just reference a preset key, or override any
field).

This registry is shared by BOTH:
  - offline image embedding extraction (`embedding_extraction/pipeline`), and
  - online query encoding for search (`retrieval/index/faiss_index.py`),
so a model only needs to be described here ONCE to be usable in both places.

Add a new model in 2 steps:
  1. If it needs a new loading strategy, add `backends/<new>.py` with a
     `load(...)` function returning a `LoadedModel`.
  2. Register it below (either as a MODEL_PRESETS entry, or just reference
     an existing backend directly from the YAML config).
"""

from __future__ import annotations

import logging
from typing import Any

from src.embedding_extraction.models.backends.base import LoadedModel

BACKEND_OPEN_CLIP = "open_clip"
BACKEND_BLIP2 = "blip2"
BACKEND_BEIT3 = "beit3"

_BACKEND_MODULES = {
    BACKEND_OPEN_CLIP: "src.embedding_extraction.models.backends.open_clip",
    BACKEND_BLIP2: "src.embedding_extraction.models.backends.blip2",
    BACKEND_BEIT3: "src.embedding_extraction.models.backends.beit3",
}

# Convenience presets for the models used across this project. A config
# entry can set `name: <preset key>` and omit `backend`/`pretrained` - or
# override any of these fields explicitly. Purely a shorthand, not required.
#
# `key` here is what the rest of the codebase (embeddings.yaml `models[].name`,
# indexing.yaml `models[].model_name`, the `advanced_search` model checklist)
# refers to as the human-readable model name; short aliases are included so
# configs can use whichever spelling is convenient.
MODEL_PRESETS: dict[str, dict[str, Any]] = {
    # --- SigLIP2 so400m (open_clip) ---
    "ViT-SO400M-16-SigLIP2-384": dict(backend=BACKEND_OPEN_CLIP, pretrained="webli"),
    "siglip2-so400m": dict(backend=BACKEND_OPEN_CLIP, pretrained="webli", model_name="ViT-SO400M-16-SigLIP2-384"),

    # --- DFN5B ViT-H/14-378-quickgelu (open_clip) ---
    "ViT-H-14-378-quickgelu": dict(backend=BACKEND_OPEN_CLIP, pretrained="dfn5b"),
    "vitH-378-quickgelu": dict(backend=BACKEND_OPEN_CLIP, pretrained="dfn5b", model_name="ViT-H-14-378-quickgelu"),
    # kept as an alias for the (likely typo'd) spelling "vitH-374-quickgelu"
    "vitH-374-quickgelu": dict(backend=BACKEND_OPEN_CLIP, pretrained="dfn5b", model_name="ViT-H-14-378-quickgelu"),

    # --- Long-CLIP-L (open_clip, BeichenZhang/LongCLIP-L checkpoint on HF hub) ---
    "LongCLIP-L": dict(backend=BACKEND_OPEN_CLIP, pretrained="hf-hub:BeichenZhang/LongCLIP-L"),
    "long-clipL": dict(backend=BACKEND_OPEN_CLIP, pretrained="hf-hub:BeichenZhang/LongCLIP-L", model_name="LongCLIP-L"),

    # --- BLIP2 ViT-G ITM/ITC (transformers) ---
    "BLIP2-ViT-G": dict(backend=BACKEND_BLIP2, pretrained="Salesforce/blip2-itm-vit-g"),
    "BLIP2": dict(backend=BACKEND_BLIP2, pretrained="Salesforce/blip2-itm-vit-g", model_name="BLIP2-ViT-G"),

    # --- BEiT-3 Large retrieval (external unilm repo + checkpoint, see backends/beit3.py) ---
    "BEiT3-Large-Retrieval": dict(backend=BACKEND_BEIT3, pretrained=None),
    "BEiT-3": dict(backend=BACKEND_BEIT3, pretrained=None, model_name="BEiT3-Large-Retrieval"),
}


def _resolve_backend_module(backend: str):
    import importlib

    if backend not in _BACKEND_MODULES:
        raise ValueError(
            f"Unknown backend '{backend}'. Available: {sorted(_BACKEND_MODULES)}"
        )
    return importlib.import_module(_BACKEND_MODULES[backend])


def load_model(
    model_name: str,
    backend: str | None = None,
    pretrained: str | None = None,
    precision: str = "fp32",
    device_name: str = "auto",
    logger: logging.Logger | None = None,
    **extra: Any,
) -> LoadedModel:
    """Single entry point used by both the offline extraction pipeline and
    the online search index to load ANY supported model.

    `model_name` can be a preset key (e.g. "siglip2-so400m", "BLIP2",
    "BEiT-3", "long-clipL") — in which case `backend`/`pretrained` (and,
    for presets whose real underlying model name differs from the key,
    the actual `model_name` passed to the backend) are resolved from
    `MODEL_PRESETS`. Anything explicitly passed always wins over the preset
    default. If `model_name` isn't a preset key, `backend` must be given
    explicitly (open_clip presets/checkpoints can be referenced directly by
    their real open_clip name, e.g. "ViT-SO400M-16-SigLIP2-384").
    """
    preset = MODEL_PRESETS.get(model_name, {})
    backend = backend or preset.get("backend")
    pretrained = pretrained if pretrained is not None else preset.get("pretrained")
    resolved_model_name = preset.get("model_name", model_name)

    if not backend:
        raise ValueError(
            f"No backend specified and no preset found for model '{model_name}'. "
            f"Set 'backend' explicitly in the config, or add a preset in "
            f"registry.MODEL_PRESETS. Known presets: {sorted(MODEL_PRESETS)}"
        )

    module = _resolve_backend_module(backend)
    return module.load(
        model_name=resolved_model_name,
        pretrained=pretrained,
        precision=precision,
        device_name=device_name,
        logger=logger,
        **extra,
    )
