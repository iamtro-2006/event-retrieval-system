"""Common contract every embedding backend must satisfy.

Design goal: `embedder.py` / `extract_embeddings.py` should NOT need to know
which backend (open_clip, huggingface transformers, beit3 ...) produced the
model. Every backend loader returns a `LoadedModel`, whose `.model` object
exposes a single method: `encode_image(batch: torch.Tensor) -> torch.Tensor`.

That's the only method `encode_keyframe_images()` in embedder.py calls, so
as long as a backend wraps its underlying HF/open_clip/beit3 model behind
that one method, the rest of the pipeline (batching, saving .npy, tqdm,
overwrite handling, etc.) keeps working unmodified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import torch
from PIL import Image


class ImageEncoder(Protocol):
    """Anything with this method can be dropped into encode_keyframe_images()."""

    def encode_image(self, batch: torch.Tensor) -> torch.Tensor: ...


@dataclass
class LoadedModel:
    model: ImageEncoder
    preprocess: Callable[[Image.Image], torch.Tensor]
    device: torch.device
    precision: str
    embedding_dim: int | None = None


class EncodeImageWrapper(torch.nn.Module):
    """Wraps a callable(pixel_values) -> embedding tensor so it exposes
    the `.encode_image` name that embedder.py expects, without needing a
    subclass per backend.
    """

    def __init__(self, forward_fn: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self._forward_fn = forward_fn

    def encode_image(self, batch: torch.Tensor) -> torch.Tensor:
        return self._forward_fn(batch)
