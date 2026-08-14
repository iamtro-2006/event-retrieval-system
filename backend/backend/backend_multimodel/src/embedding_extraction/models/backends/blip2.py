"""BLIP-2 backend.

Uses `Blip2VisionModelWithProjection` from transformers - this is the
official HF head trained for image-text *retrieval* (ITC), as opposed to
`Blip2ForConditionalGeneration` which is for captioning/VQA. It reuses the
same "Salesforce/blip2-itm-vit-g" checkpoint most retrieval pipelines use.

The Q-Former produces `num_query_tokens` (default 32) projected embeddings
per image, not a single vector. For a flat per-keyframe embedding we pool
them down to one vector - configurable via `pooling`:
  - "mean" (default): average over the 32 query tokens
  - "first": take the first query token only (cheaper, slightly lossier)

If you need the full multi-vector representation (best retrieval quality,
same approach as the official BLIP-2 ITC eval) keep pooling="none" and
adapt the saving step, since encode_keyframe_images assumes one vector/image.
"""

from __future__ import annotations

import logging

import torch
from PIL import Image
from transformers import AutoProcessor, Blip2VisionModelWithProjection

from src.embedding_extraction.models.backends.base import EncodeImageWrapper, LoadedModel
from src.utils.device import resolve_device


def load(
    model_name: str,
    pretrained: str = "Salesforce/blip2-itm-vit-g",
    precision: str = "fp32",
    device_name: str = "auto",
    pooling: str = "mean",
    logger: logging.Logger | None = None,
    **_: object,
) -> LoadedModel:
    if pooling not in {"mean", "first"}:
        raise ValueError(f"Unsupported BLIP-2 pooling strategy: {pooling}")

    device = resolve_device(device_name)

    if device.type == "cpu" and precision in {"fp16", "amp"}:
        precision = "fp32"

    dtype = torch.float16 if precision in {"fp16", "amp"} and device.type == "cuda" else torch.float32

    if logger:
        logger.info(
            "[blip2] loading %s on %s precision=%s pooling=%s",
            pretrained, device, precision, pooling,
        )

    vision_model = Blip2VisionModelWithProjection.from_pretrained(
        pretrained, dtype=dtype
    ).to(device).eval()

    processor = AutoProcessor.from_pretrained(pretrained)

    def preprocess(image: Image.Image) -> torch.Tensor:
        return processor(images=image, return_tensors="pt")["pixel_values"][0]

    @torch.inference_mode()
    def _encode_image(batch: torch.Tensor) -> torch.Tensor:
        batch = batch.to(dtype)
        out = vision_model(pixel_values=batch)
        # out.image_embeds: (batch, num_query_tokens, proj_dim)
        embeds = out.image_embeds
        if pooling == "mean":
            return embeds.mean(dim=1)
        return embeds[:, 0, :]

    embedding_dim = getattr(vision_model.config, "image_text_hidden_size", None)

    return LoadedModel(
        model=EncodeImageWrapper(_encode_image),
        preprocess=preprocess,
        device=device,
        precision=precision,
        embedding_dim=embedding_dim,
    )
