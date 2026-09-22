"""Official image-document and text-query adapter for pinned WeMM-Embedding-2B."""

from __future__ import annotations

import logging

import numpy as np
import torch

from src.embedding_extraction.models.backends.base import LoadedModel
from src.embedding_extraction.models.backends.sentence_transformer_image import (
    SentenceTransformerImageEncoder,
)
from src.utils.device import resolve_device


MODEL_ID = "tencent/WeMM-Embedding-2B"
MODEL_REVISION = "bbd6cd4bf52cfc6716f752a2df80b2706720bd95"
EMBEDDING_DIM = 2048


class WeMMEncoder(SentenceTransformerImageEncoder):
    """Keep image documents and text queries in the same pinned WeMM space."""

    @torch.inference_mode()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        if not texts:
            return torch.empty((0, EMBEDDING_DIM), dtype=torch.float32)
        values = self.model.encode_query(
            texts,
            batch_size=len(texts),
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        vectors = np.asarray(values, dtype=np.float32)
        if vectors.shape != (len(texts), EMBEDDING_DIM):
            raise ValueError(
                f"Expected WeMM queries ({len(texts)}, {EMBEDDING_DIM}), got {vectors.shape}"
            )
        if not np.isfinite(vectors).all():
            raise ValueError("WeMM query embeddings contain NaN/Inf")
        norms = np.linalg.norm(vectors, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3, rtol=0):
            raise ValueError(
                f"WeMM query L2 norms outside tolerance: {norms.min()}..{norms.max()}"
            )
        return torch.from_numpy(np.ascontiguousarray(vectors))


def load(
    model_name: str,
    pretrained: str | None = None,
    precision: str = "fp16",
    device_name: str = "auto",
    revision: str = MODEL_REVISION,
    logger: logging.Logger | None = None,
    **_: object,
) -> LoadedModel:
    if model_name != MODEL_ID:
        raise ValueError(f"WeMM backend only supports {MODEL_ID}, got {model_name}")
    if revision != MODEL_REVISION:
        raise ValueError(f"WeMM integration requires pinned revision {MODEL_REVISION}")
    if precision not in {"fp16", "fp32"}:
        raise ValueError("WeMM backend supports fp16/fp32 only; no BF16 or AMP")

    from sentence_transformers import SentenceTransformer

    device = resolve_device(device_name)
    effective_precision = (
        "fp16" if device.type == "cuda" and precision == "fp16" else "fp32"
    )
    if logger:
        logger.info("Loading %s revision=%s on %s", MODEL_ID, revision, device)
    model = SentenceTransformer(
        MODEL_ID,
        revision=revision,
        trust_remote_code=True,
        device=str(device),
        model_kwargs={
            "torch_dtype": (
                torch.float16 if effective_precision == "fp16" else torch.float32
            ),
            "attn_implementation": "sdpa",
        },
    ).eval()
    actual_dim = model.get_sentence_embedding_dimension()
    if actual_dim is not None and actual_dim != EMBEDDING_DIM:
        raise ValueError(f"{MODEL_ID} dimension {actual_dim}, expected {EMBEDDING_DIM}")
    return LoadedModel(
        model=WeMMEncoder(
            model,
            method="encode_document",
            normalize_embeddings=False,
            embedding_dim=EMBEDDING_DIM,
        ),
        preprocess=None,
        device=device,
        precision=effective_precision,
        embedding_dim=EMBEDDING_DIM,
        supports_text=True,
    )
