"""HuggingFace-transformers "CLIP-like" backend.

Covers models that ship a CLIPModel-compatible API on the Hub
(`get_image_features(pixel_values=...)`), most notably:

  - Long-CLIP-L: "creative-graphic-design/LongCLIP-L" (trust_remote_code=True)
    "Plug-and-Play drop-in replacement for CLIP" - handles the 248-token
    long captions, but the image side is a standard ViT-L/14, so we only
    need get_image_features() here (text side is only relevant if you also
    embed queries with the same model at search time).

Any other transformers model that exposes `get_image_features` can reuse
this backend by just pointing `pretrained` at its HF repo id.
"""

from __future__ import annotations

import logging

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from src.embedding_extraction.models.backends.base import EncodeImageWrapper, LoadedModel
from src.utils.device import resolve_device


def load(
    model_name: str,
    pretrained: str,
    precision: str = "fp32",
    device_name: str = "auto",
    trust_remote_code: bool = True,
    logger: logging.Logger | None = None,
    **_: object,
) -> LoadedModel:
    device = resolve_device(device_name)

    if device.type == "cpu" and precision in {"fp16", "amp"}:
        precision = "fp32"

    dtype = {"fp16": torch.float16, "amp": torch.float32}.get(precision, torch.float32)

    if logger:
        logger.info(
            "[hf_clip] loading %s (%s) on %s precision=%s",
            model_name, pretrained, device, precision,
        )

    hf_model = AutoModel.from_pretrained(
        pretrained,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
    ).to(device).eval()

    processor = AutoProcessor.from_pretrained(pretrained, trust_remote_code=trust_remote_code)

    def preprocess(image: Image.Image) -> torch.Tensor:
        # Return a single CHW tensor so it stacks with torch.stack() the
        # same way open_clip's preprocess() does in embedder.py.
        return processor(images=image, return_tensors="pt")["pixel_values"][0]

    @torch.inference_mode()
    def _encode_image(batch: torch.Tensor) -> torch.Tensor:
        batch = batch.to(dtype)
        if hasattr(hf_model, "get_image_features"):
            return hf_model.get_image_features(pixel_values=batch)
        # Fallback for repos that don't expose get_image_features directly.
        vision_out = hf_model.vision_model(pixel_values=batch)
        pooled = vision_out.pooler_output
        if hasattr(hf_model, "visual_projection"):
            pooled = hf_model.visual_projection(pooled)
        return pooled

    embedding_dim = getattr(getattr(hf_model, "config", None), "projection_dim", None)

    return LoadedModel(
        model=EncodeImageWrapper(_encode_image),
        preprocess=preprocess,
        device=device,
        precision=precision,
        embedding_dim=embedding_dim,
    )
