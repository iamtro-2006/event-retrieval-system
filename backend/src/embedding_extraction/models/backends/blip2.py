"""BLIP-2 backend.

Uses `Blip2VisionModelWithProjection` from transformers - this is the
official HF head trained for image-text *retrieval* (ITC), as opposed to
`Blip2ForConditionalGeneration` which is for captioning/VQA. It reuses the
same "Salesforce/blip2-itm-vit-g" checkpoint most retrieval pipelines use.

The Q-Former produces `num_query_tokens` (default 32) projected embeddings
per image, each already L2-normalized per-token by
`Blip2VisionModelWithProjection.forward()` - this is the exact contrastive
space BLIP-2/LAVIS was trained in.

`pooling` controls how a single flat vector is produced for storage:
  - "first" (default): take query token 0, i.e. `image_embeds[:, 0, :]`.
    This matches the convention used by LAVIS's own feature-extractor demo
    (`model.extract_features(sample, mode="image").image_embeds[0, 0, :]`,
    see the Salesforce/LAVIS "Image and text features extraction" tutorial)
    and, critically, the SAME slicing convention used on the text side
    (`text_embeds[:, 0, :]`, the BERT [CLS] token). Since retrieval is done
    later by cosine-similarity between one flat image vector and one flat
    text vector (standard vector-DB / ANN search - FAISS, Milvus, Qdrant,
    ...), both sides must live in the same single-vector convention; this
    is what makes the offline image embeddings here actually matchable
    against text embeddings extracted the same way in a later stage.
  - "none": keep the full (num_query_tokens, proj_dim) multi-vector
    representation. This is what LAVIS's official retrieval evaluation
    (`compute_sim_matrix`) uses for the *best* recall - it does
    `sim_i2t = (image_feats @ text_feat.T).max(dim=1)` (max over the 32
    tokens), not first-token indexing. Only use this if your search stage
    can do late-interaction / max-sim matching over multi-vector storage
    instead of a flat ANN index - most vector DBs can't.
  - "mean": average over the 32 query tokens into one vector. A cheaper,
    lower-recall alternative to "first"/"none" - LAVIS itself doesn't use
    this for retrieval.
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
    pooling: str = "first",
    logger: logging.Logger | None = None,
    **_: object,
) -> LoadedModel:
    if pooling not in {"none", "mean", "first"}:
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
        # out.image_embeds: (batch, num_query_tokens, proj_dim), each of the
        # num_query_tokens vectors already L2-normalized per-token by the
        # model itself (Blip2VisionModelWithProjection.forward() calls
        # F.normalize(..., dim=-1) before returning) - this is the same
        # space LAVIS uses for its max-similarity ITC scoring.
        embeds = out.image_embeds
        if pooling == "first":
            return embeds[:, 0, :]
        if pooling == "none":
            return embeds
        return embeds.mean(dim=1)

    embedding_dim = getattr(vision_model.config, "image_text_hidden_size", None)

    return LoadedModel(
        model=EncodeImageWrapper(_encode_image),
        preprocess=preprocess,
        device=device,
        precision=precision,
        embedding_dim=embedding_dim,
    )
