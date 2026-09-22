"""Shared path-native image adapter for SentenceTransformer multimodal models."""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import torch


def validate_image_batch(
    values: object,
    count: int,
    embedding_dim: int,
) -> np.ndarray:
    """Preserve model values while enforcing the per-frame artifact contract."""
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (count, embedding_dim):
        raise ValueError(
            f"Expected image embeddings ({count}, {embedding_dim}), got {result.shape}"
        )
    if not np.isfinite(result).all():
        raise ValueError("Image embeddings contain NaN/Inf")
    norms = np.linalg.norm(result, axis=1)
    if not np.allclose(norms, 1.0, rtol=0, atol=1e-3):
        raise ValueError(
            f"Image embedding L2 norms outside tolerance: {norms.min()}..{norms.max()}"
        )
    return np.ascontiguousarray(result)


class SentenceTransformerImageEncoder:
    def __init__(
        self,
        model: object,
        *,
        method: str,
        normalize_embeddings: bool,
        embedding_dim: int,
    ):
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        self.model = model
        self.method = method
        self.normalize_embeddings = normalize_embeddings
        self.embedding_dim = int(embedding_dim)

    @torch.inference_mode()
    def encode_image_paths(self, paths: list[Path]) -> np.ndarray:
        if not paths:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        documents = [{"image": str(path)} for path in paths]
        encode = getattr(self.model, self.method)
        values = encode(
            documents,
            batch_size=len(documents),
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        return validate_image_batch(values, len(paths), self.embedding_dim)

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        raise NotImplementedError(
            "Text/query encoding must be implemented by the model-specific backend"
        )

    def close(self) -> None:
        """Release a completed offline model before loading another backend."""
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
