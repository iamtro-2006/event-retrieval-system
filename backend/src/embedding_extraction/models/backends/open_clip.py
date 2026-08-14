"""open_clip backend.

Covers every model already natively supported by `open_clip`, including:
  - ViT-SO400M-16-SigLIP2-384   (pretrained="webli")
  - ViT-H-14-378-quickgelu      (pretrained="dfn5b")
  - any other open_clip / timm hf-hub checkpoint, e.g. pretrained="hf-hub:..."

This is exactly the logic that used to live directly in embedder.py -
nothing new, just isolated so it can sit next to the other backends behind
the same `load()` contract.
"""

from __future__ import annotations

import logging

import open_clip
import torch

from src.embedding_extraction.models.backends.base import LoadedModel
from src.utils.device import resolve_device


def load(
    model_name: str,
    pretrained: str,
    precision: str = "fp32",
    device_name: str = "auto",
    logger: logging.Logger | None = None,
    **_: object,
) -> LoadedModel:
    device = resolve_device(device_name)

    if device.type == "cpu" and precision in {"fp16", "amp"}:
        precision = "fp32"

    if logger:
        logger.info(
            "[open_clip] loading %s / %s on %s precision=%s",
            model_name, pretrained, device, precision,
        )

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        precision=precision,
    )
    model = model.to(device).eval()

    embedding_dim = getattr(model, "visual", None)
    embedding_dim = getattr(embedding_dim, "output_dim", None) or getattr(model, "embed_dim", None)

    return LoadedModel(
        model=model,  # open_clip model already exposes .encode_image
        preprocess=preprocess,
        device=device,
        precision=precision,
        embedding_dim=embedding_dim,
    )
