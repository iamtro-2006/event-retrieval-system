"""Central model registry / factory.

Every backend module exposes a `load(...) -> LoadedModel` function with the
same signature shape. This file just knows *which* backend module to call
for a given `backend` string, plus a small table of ready-made presets for
the models this project cares about so configs/embeddings.yaml can stay
short (just reference a preset key, or override any field).

Add a new model in 2 steps:
  1. If it needs a new loading strategy, add `backends/<new>_backend.py`
     with a `load(...)` function returning a `LoadedModel`.
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

# Convenience presets for the models requested for this project. A config
# entry can set `name: <preset key>` and omit `backend`/`pretrained` - or
# override any of these fields explicitly. Purely a shorthand, not required.
MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "ViT-SO400M-16-SigLIP2-384": dict(backend=BACKEND_OPEN_CLIP, pretrained="webli"),
    "ViT-H-14-378-quickgelu": dict(backend=BACKEND_OPEN_CLIP, pretrained="dfn5b"),
    "BLIP2-ViT-G": dict(backend=BACKEND_BLIP2, pretrained="Salesforce/blip2-itm-vit-g"),
    "BEiT3-Large-Retrieval": dict(backend=BACKEND_BEIT3, pretrained=None),
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
    """Single entry point used by the pipeline to load ANY supported model.

    `backend`/`pretrained` can be omitted if `model_name` matches a
    MODEL_PRESETS key; anything explicitly passed always wins over the
    preset default.
    """
    preset = MODEL_PRESETS.get(model_name, {})
    backend = backend or preset.get("backend")
    pretrained = pretrained if pretrained is not None else preset.get("pretrained")

    if not backend:
        raise ValueError(
            f"No backend specified and no preset found for model '{model_name}'. "
            f"Set 'backend' explicitly in the config, or add a preset in "
            f"registry.MODEL_PRESETS. Known presets: {sorted(MODEL_PRESETS)}"
        )

    module = _resolve_backend_module(backend)
    return module.load(
        model_name=model_name,
        pretrained=pretrained,
        precision=precision,
        device_name=device_name,
        logger=logger,
        **extra,
    )
