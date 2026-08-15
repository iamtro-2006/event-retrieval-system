"""open_clip backend.

Covers every model already natively supported by `open_clip`, including:
  - ViT-SO400M-16-SigLIP2-384   (pretrained="webli")
  - ViT-H-14-378-quickgelu      (pretrained="dfn5b")
  - Long-CLIP variants          (e.g. name="LongCLIP-L", pretrained="hf-hub:...")
  - any other open_clip / timm hf-hub checkpoint, e.g. pretrained="hf-hub:..."

open_clip models natively expose both `.encode_image(pixel_values)` and
`.encode_text(token_ids)`, so this backend is the reference implementation
for `encode_text` support: it wraps the model + its own `open_clip`
tokenizer behind `EncoderWrapper` so the rest of the codebase always calls
`.encode_text(list[str])` (raw strings), never handling tokenization itself.
"""

from __future__ import annotations

import logging

import open_clip
import torch

from src.embedding_extraction.models.backends.base import EncoderWrapper, LoadedModel
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
    tokenizer = open_clip.get_tokenizer(model_name)

    autocast_dtype = torch.float16 if precision in {"fp16", "amp"} else torch.bfloat16
    use_autocast = device.type == "cuda" and precision in {"fp16", "amp", "bf16"}

    @torch.inference_mode()
    def _encode_image(batch: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
            return model.encode_image(batch)

    @torch.inference_mode()
    def _encode_text(texts: list[str]) -> torch.Tensor:
        tokens = tokenizer(list(texts)).to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
            return model.encode_text(tokens)

    embedding_dim = getattr(model, "visual", None)
    embedding_dim = getattr(embedding_dim, "output_dim", None) or getattr(model, "embed_dim", None)

    return LoadedModel(
        model=EncoderWrapper(_encode_image, _encode_text, backend_label="open_clip"),
        preprocess=preprocess,
        device=device,
        precision=precision,
        embedding_dim=embedding_dim,
        supports_text=True,
    )
