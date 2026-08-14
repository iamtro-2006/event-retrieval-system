"""BEiT-3 backend.

BEiT-3 is NOT on `transformers` or `open_clip`. It only exists as raw code
+ checkpoint in the microsoft/unilm repo:
    https://github.com/microsoft/unilm/tree/master/beit3

Setup required once, outside this codebase:
  1. git clone https://github.com/microsoft/unilm.git <somewhere>
  2. pip install -r unilm/beit3/requirements.txt   (needs timm==0.4.12, torchscale)
  3. Download an *-itc retrieval checkpoint, e.g. beit3_large_patch16_384_coco_retrieval.pth
     (image-only embedding extraction does not need the beit3.spm sentencepiece
     model - that's only required for tokenizing text queries at search time)

Then point `configs/embeddings.yaml` at it, e.g.:

    - key: beit3_large_itc
      backend: beit3
      name: BEiT3-Large-Retrieval
      device: auto
      precision: fp32
      batch_size: 16
      normalize: true
      beit3:
        repo_path: third_party/unilm/beit3
        model_arch: beit3_large_patch16_384
        checkpoint_path: checkpoints/beit3_large_patch16_384_coco_retrieval.pth
        input_size: 384

This backend imports `modeling_finetune` from that repo to register the
`beit3_*` architectures with timm, builds the model, loads the checkpoint,
and wraps `BEiT3ForRetrieval.forward(image=..., only_infer=True)` so it
returns the L2-normalized vision_cls embedding - matching the same
`.encode_image(batch)` contract every other backend uses.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

from src.embedding_extraction.models.backends.base import EncodeImageWrapper, LoadedModel
from src.utils.device import resolve_device

# BEiT-3 was trained with "inception-style" normalization, not the usual
# ImageNet mean/std. See unilm/beit3/datasets.py -> build_transform().
_BEIT3_MEAN = (0.5, 0.5, 0.5)
_BEIT3_STD = (0.5, 0.5, 0.5)


def _import_beit3_modeling(repo_path: str):
    """Add the local unilm/beit3 checkout to sys.path and import it once."""
    repo_path = Path(repo_path).expanduser().resolve()
    if not repo_path.exists():
        raise FileNotFoundError(
            f"BEiT-3 repo not found at '{repo_path}'. Clone "
            "https://github.com/microsoft/unilm and point "
            "configs/embeddings.yaml -> beit3.repo_path at its 'beit3' folder."
        )

    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

    try:
        import modeling_finetune  # noqa: F401  (registers beit3_* archs with timm)
    except ImportError as e:
        raise ImportError(
            "Could not import 'modeling_finetune' from the BEiT-3 repo. "
            "Make sure beit3.repo_path points at the 'beit3' folder that "
            "directly contains modeling_finetune.py, and that its "
            "requirements (timm, torchscale, ...) are installed."
        ) from e

    return modeling_finetune


def _build_transform(input_size: int):
    return T.Compose([
        T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=_BEIT3_MEAN, std=_BEIT3_STD),
    ])


def load(
    model_name: str,
    pretrained: str | None = None,  # unused, kept for interface parity
    precision: str = "fp32",
    device_name: str = "auto",
    beit3: dict | None = None,
    logger: logging.Logger | None = None,
    **_: object,
) -> LoadedModel:
    import timm  # local import: only required when this backend is used

    beit3_cfg = beit3 or {}
    repo_path = beit3_cfg.get("repo_path")
    model_arch = beit3_cfg.get("model_arch", "beit3_large_patch16_384")
    checkpoint_path = beit3_cfg.get("checkpoint_path")
    input_size = int(beit3_cfg.get("input_size", 384))

    if not repo_path or not checkpoint_path:
        raise ValueError(
            "BEiT-3 backend requires 'beit3.repo_path' and "
            "'beit3.checkpoint_path' in the model config."
        )

    device = resolve_device(device_name)
    if device.type == "cpu" and precision in {"fp16", "amp"}:
        precision = "fp32"

    if logger:
        logger.info(
            "[beit3] loading %s from %s on %s precision=%s",
            model_arch, checkpoint_path, device, precision,
        )

    _import_beit3_modeling(repo_path)

    model = timm.models.create_model(model_arch, pretrained=False, drop_path_rate=0.0)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if logger and (missing or unexpected):
        logger.warning(
            "[beit3] state_dict mismatch - missing=%d unexpected=%d "
            "(expected for text-tower/head keys if this checkpoint isn't "
            "an *-itc retrieval checkpoint)",
            len(missing), len(unexpected),
        )

    model = model.to(device).eval()
    if precision == "fp16" and device.type == "cuda":
        model = model.half()

    transform = _build_transform(input_size)

    def preprocess(image: Image.Image) -> torch.Tensor:
        return transform(image.convert("RGB"))

    @torch.inference_mode()
    def _encode_image(batch: torch.Tensor) -> torch.Tensor:
        if precision == "fp16" and device.type == "cuda":
            batch = batch.half()
        vision_cls, _ = model(image=batch, only_infer=True)
        return vision_cls  # already L2-normalized inside BEiT3ForRetrieval

    embedding_dim = getattr(getattr(model, "vision_head", None), "out_features", None)

    return LoadedModel(
        model=EncodeImageWrapper(_encode_image),
        preprocess=preprocess,
        device=device,
        precision=precision,
        embedding_dim=embedding_dim,
    )
