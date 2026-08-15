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
from transformers import AutoProcessor, Blip2TextModelWithProjection, Blip2VisionModelWithProjection

from src.embedding_extraction.models.backends.base import EncoderWrapper, LoadedModel
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

    # Text tower: same ITC checkpoint's Q-Former text side, projected into the
    # same contrastive space as the vision projection above (mirrors LAVIS'
    # `extract_features(mode="text").text_embeds[:, 0, :]` [CLS]-token
    # convention referenced in the module docstring). Needed so this backend
    # can serve `encode_text()` for search, not just offline image extraction.
    try:
        text_model = Blip2TextModelWithProjection.from_pretrained(
            pretrained, dtype=dtype
        ).to(device).eval()
    except Exception as exc:  # pragma: no cover - checkpoint without a text tower
        text_model = None
        if logger:
            logger.warning(
                "[blip2] could not load text tower from '%s' (%s: %s); "
                "encode_text() will be unavailable for this model.",
                pretrained, type(exc).__name__, exc,
            )

    processor = AutoProcessor.from_pretrained(pretrained)

    def preprocess(image: Image.Image) -> torch.Tensor:
        return processor(images=image, return_tensors="pt")["pixel_values"][0]

    def _pool(embeds: torch.Tensor) -> torch.Tensor:
        # embeds: (batch, num_query_tokens, proj_dim), each token already
        # L2-normalized by the HF projection head. See pooling docstring above.
        if pooling == "first":
            return embeds[:, 0, :]
        if pooling == "none":
            return embeds
        return embeds.mean(dim=1)

    @torch.inference_mode()
    def _encode_image(batch: torch.Tensor) -> torch.Tensor:
        batch = batch.to(dtype)
        out = vision_model(pixel_values=batch)
        return _pool(out.image_embeds)

    _encode_text_fn = None
    if text_model is not None:
        @torch.inference_mode()
        def _encode_text(texts: list[str]) -> torch.Tensor:
            inputs = processor(text=list(texts), padding=True, return_tensors="pt").to(device)
            out = text_model(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"))
            return _pool(out.text_embeds)

        _encode_text_fn = _encode_text

    embedding_dim = getattr(vision_model.config, "image_text_hidden_size", None)

    return LoadedModel(
        model=EncoderWrapper(_encode_image, _encode_text_fn, backend_label="blip2"),
        preprocess=preprocess,
        device=device,
        precision=precision,
        embedding_dim=embedding_dim,
        supports_text=_encode_text_fn is not None,
    )
