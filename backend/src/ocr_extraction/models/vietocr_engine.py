from __future__ import annotations

import logging
import warnings

import numpy as np
from PIL import Image

# --- Compatibility shims -----------------------------------------------
# 1) Pillow >= 10 removed the old `Image.ANTIALIAS` alias (it was always
#    just Lanczos resampling under a misleading name). vietocr's internal
#    resize code still references it, so we restore the name here rather
#    than pinning an old, wheel-less Pillow version.
if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.Resampling.LANCZOS

# 2) Silence noisy-but-harmless UserWarnings that come from *inside*
#    vietocr's dependencies (torchvision's deprecated `pretrained=`/
#    `weights=` argument handling, and a PyTorch TransformerEncoder
#    `batch_first` perf hint). These do not affect OCR correctness, only
#    log verbosity, and we can't edit the vietocr package itself.
warnings.filterwarnings(
    "ignore",
    message=r".*'pretrained' is deprecated.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*Arguments other than a weight enum or `None`.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*enable_nested_tensor is True.*",
    category=UserWarning,
)

from vietocr.tool.config import Cfg  # noqa: E402
from vietocr.tool.predictor import Predictor  # noqa: E402


def load_vietocr_model(
    base_config: str = "vgg_transformer",
    device: str = "cpu",
    weights_path: str | None = None,
    logger: logging.Logger | None = None,
) -> Predictor:
    """Load a VietOCR Predictor.

    base_config: one of vietocr's built-in configs, e.g. "vgg_transformer"
        or "vgg_seq2seq" (both ship pretrained weights for Vietnamese).
    weights_path: optional path/URL to a fine-tuned .pth checkpoint that
        overrides the default pretrained weights of base_config.
    """
    if logger:
        logger.info(
            "Loading VietOCR model: base_config=%s device=%s weights_path=%s",
            base_config,
            device,
            weights_path or "<default pretrained>",
        )

    config = Cfg.load_config_from_name(base_config)
    config["device"] = device
    config["predictor"]["beamsearch"] = False

    if weights_path:
        config["weights"] = weights_path

    return Predictor(config)


def crop_box(image_bgr: np.ndarray, box: list[float]) -> Image.Image | None:
    """Crop an axis-aligned [x1, y1, x2, y2] box from a BGR (OpenCV) image
    and return it as an RGB PIL.Image ready for vietocr.tool.predictor.Predictor.

    Returns None if the box is degenerate / out of bounds.
    """
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))

    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    return Image.fromarray(crop[:, :, ::-1])  # BGR -> RGB


def recognize_crops(
    predictor: Predictor,
    crops: list[Image.Image],
    logger: logging.Logger | None = None,
) -> list[str]:
    """Run VietOCR recognition over a list of cropped text-region images.

    Uses vietocr's `Predictor.predict_batch`, which internally buckets crops
    by resized width and runs ONE forward pass per bucket (torch.cat), instead
    of one forward pass per crop. This is the actual throughput bottleneck
    the previous single-image loop had -- each `predict()` call pays GPU
    kernel-launch + host<->device transfer overhead on its own, which
    dominates runtime far more than any autograd/nested-tensor warning does.

    Falls back to per-crop `predict()` only if the whole-batch call fails
    (e.g. one malformed crop), so a single bad crop can't silently drop the
    whole batch's results.
    """
    if not crops:
        return []

    try:
        return list(predictor.predict_batch(crops))
    except Exception as e:
        if logger:
            logger.warning(
                "VietOCR predict_batch failed for a batch of %d crops, "
                "falling back to per-crop predict: %s",
                len(crops),
                e,
            )

    texts: list[str] = []
    for crop in crops:
        try:
            texts.append(predictor.predict(crop))
        except Exception as e:
            if logger:
                logger.warning("VietOCR failed on a crop: %s", e)
            texts.append("")
    return texts
