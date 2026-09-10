"""Meta Perception Encoder (PE-Core) backend.

This adapter intentionally targets PE-Core CLIP checkpoints: they expose an
image tower and a text tower in one shared retrieval space. PE-Lang and
PE-Spatial are dense/token encoders and therefore are not interchangeable
with the flat text-to-image FAISS indexes used by this project.

Install the official implementation before enabling a PE model::

    pip install --no-deps \
      "perception_models @ git+https://github.com/facebookresearch/perception_models.git"

The loader accepts every PE-Core config known to the installed package, not
just the built-in presets in ``registry.py``. A local clone can alternatively
be supplied through ``perception_encoder.repo_path``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import torch

from src.embedding_extraction.models.backends.base import EncoderWrapper, LoadedModel
from src.utils.device import resolve_device


def _canonical_model_name(model_name: str) -> str:
    """Accept both ``PE-Core-*`` and Hugging Face ``facebook/PE-Core-*``."""
    return model_name.split("/", 1)[1] if model_name.startswith("facebook/") else model_name


def _import_official_package(repo_path: str | None = None):
    if repo_path:
        resolved = Path(repo_path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(
                f"Perception Models repo not found at '{resolved}'. "
                "Point perception_encoder.repo_path at the repository root."
            )
        if str(resolved) not in sys.path:
            sys.path.insert(0, str(resolved))

    try:
        import core.vision_encoder.pe as pe
        import core.vision_encoder.transforms as transforms
    except ImportError as exc:
        raise ImportError(
            "The PE-Core backend needs Meta's official 'perception_models' package. "
            "Install it from https://github.com/facebookresearch/perception_models "
            "or set perception_encoder.repo_path to a local clone."
        ) from exc
    return pe, transforms


def _resolve_pretrained(pretrained: Any, checkpoint_path: str | None) -> tuple[bool, str | None]:
    """Map project config values onto PE's ``from_config`` arguments."""
    if checkpoint_path:
        return True, str(Path(checkpoint_path).expanduser())
    if pretrained is None:
        return True, None
    if isinstance(pretrained, bool):
        return pretrained, None

    value = str(pretrained).strip()
    if value.lower() in {"", "false", "none", "random"}:
        return False, None
    if value.lower() in {"true", "hf", "huggingface", "auto", "pretrained"}:
        return True, None
    # A Hugging Face repo id/URL names the same default checkpoint that PE's
    # fetch_pe_checkpoint() resolves; it is not a local torch.load path.
    if value.startswith("facebook/PE-Core-") or value.startswith(
        "hf-hub:facebook/PE-Core-"
    ) or value.startswith("https://huggingface.co/facebook/PE-Core-"):
        return True, None
    # PE accepts a local path or its own hf://repo:file notation here.
    return True, value


def load(
    model_name: str,
    pretrained: Any = True,
    precision: str = "fp32",
    device_name: str = "auto",
    perception_encoder: dict | None = None,
    logger: logging.Logger | None = None,
    **_: object,
) -> LoadedModel:
    cfg = dict(perception_encoder or {})
    pe, transforms = _import_official_package(cfg.get("repo_path"))

    config_name = _canonical_model_name(model_name)
    available = set(pe.CLIP.available_configs())
    if config_name not in available:
        raise ValueError(
            f"'{model_name}' is not a PE-Core CLIP config in the installed "
            f"Perception Models package. Available: {sorted(available)}"
        )

    device = resolve_device(device_name)
    if device.type == "cpu" and precision in {"fp16", "amp", "bf16"}:
        precision = "fp32"

    use_pretrained, checkpoint_path = _resolve_pretrained(
        pretrained, cfg.get("checkpoint_path")
    )
    if logger:
        logger.info(
            "[perception_encoder] loading %s on %s precision=%s pretrained=%s",
            config_name,
            device,
            precision,
            use_pretrained,
        )

    model = pe.CLIP.from_config(
        config_name,
        pretrained=use_pretrained,
        checkpoint_path=checkpoint_path,
    ).to(device).eval()
    if precision == "fp16" and device.type == "cuda":
        model = model.half()
    elif precision == "bf16" and device.type == "cuda":
        model = model.bfloat16()

    preprocess = transforms.get_image_transform(
        model.image_size,
        center_crop=bool(cfg.get("center_crop", False)),
    )
    tokenizer = transforms.get_text_tokenizer(model.context_length)

    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    use_autocast = device.type == "cuda" and precision in {"fp16", "amp", "bf16"}

    @torch.inference_mode()
    def _encode_image(batch: torch.Tensor) -> torch.Tensor:
        batch = batch.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
            return model.encode_image(batch)

    @torch.inference_mode()
    def _encode_text(texts: list[str]) -> torch.Tensor:
        tokens = tokenizer(list(texts)).to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
            return model.encode_text(tokens)

    visual = getattr(model, "visual", None)
    embedding_dim = getattr(visual, "output_dim", None) or getattr(model, "output_dim", None)

    return LoadedModel(
        model=EncoderWrapper(
            _encode_image, _encode_text, backend_label="perception_encoder"
        ),
        preprocess=preprocess,
        device=device,
        precision=precision,
        embedding_dim=embedding_dim,
        supports_text=True,
    )
