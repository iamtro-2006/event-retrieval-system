"""Common contract every embedding backend must satisfy.

Design goal: `embedder.py` / `extract_embeddings.py` (offline image embedding)
and `retrieval/index/faiss_index.py` (online text/image query encoding for
search) should NOT need to know which backend (open_clip, transformers,
beit3 ...) produced the model. Every backend loader returns a `LoadedModel`,
whose `.model` object exposes:

  - `encode_image(batch: torch.Tensor) -> torch.Tensor`   (required)
  - `encode_text(texts: list[str]) -> torch.Tensor`        (optional)

`encode_image` is the only method `encode_keyframe_images()` in embedder.py
calls, so offline extraction keeps working unmodified. `encode_text` is what
`retrieval.index.faiss_index.FaissIndex.encode_texts()` calls for search --
a backend that doesn't implement it (e.g. a vision-only model, or one whose
text tower isn't wired up yet) should raise `NotImplementedError` with a
clear message; that model can then still be used for image-only search
(reverse image search / temporal continuation) even if text search isn't
available for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import torch
from PIL import Image


class ImageTextEncoder(Protocol):
    """Anything with these methods can be dropped into FaissIndex / embedder.py."""

    def encode_image(self, batch: torch.Tensor) -> torch.Tensor: ...

    def encode_text(self, texts: list[str]) -> torch.Tensor: ...


@dataclass
class LoadedModel:
    model: ImageTextEncoder
    preprocess: Callable[[Image.Image], torch.Tensor]
    device: torch.device
    precision: str
    embedding_dim: int | None = None
    supports_text: bool = False


def _no_text_encoder(backend_label: str) -> Callable[[list[str]], torch.Tensor]:
    def _raise(texts: list[str]) -> torch.Tensor:
        raise NotImplementedError(
            f"Backend '{backend_label}' does not implement encode_text(); "
            "this model can only be used for image-based search (reverse "
            "image / temporal continuation), not text queries."
        )

    return _raise


class EncoderWrapper(torch.nn.Module):
    """Wraps `encode_image`/`encode_text` callables so a plain function pair
    can expose the `.encode_image(...)` / `.encode_text(...)` names every
    other part of the pipeline expects, without needing a subclass per
    backend.
    """

    def __init__(
        self,
        encode_image_fn: Callable[[torch.Tensor], torch.Tensor],
        encode_text_fn: Callable[[list[str]], torch.Tensor] | None = None,
        backend_label: str = "unknown",
    ):
        super().__init__()
        self._encode_image_fn = encode_image_fn
        self._encode_text_fn = encode_text_fn or _no_text_encoder(backend_label)

    def encode_image(self, batch: torch.Tensor) -> torch.Tensor:
        return self._encode_image_fn(batch)

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        return self._encode_text_fn(texts)


# Backward-compatible alias (old name used across the codebase / older diffs).
EncodeImageWrapper = EncoderWrapper
