"""Qwen3-VL-Embedding-2B through the shared path-native image adapter."""

from __future__ import annotations

import logging

import numpy as np
import torch

from src.embedding_extraction.models.backends.base import LoadedModel
from src.embedding_extraction.models.backends.sentence_transformer_image import (
    SentenceTransformerImageEncoder,
)
from src.embedding_extraction.models.backends.qwen_vllm import QwenVllmEncoder, resolve_engine
from src.utils.device import resolve_device


MODEL_ID = "Qwen/Qwen3-VL-Embedding-2B"
EMBEDDING_DIM = 2048


class Qwen3VLEncoder(SentenceTransformerImageEncoder):
    engine = "sentence_transformer"

    @torch.inference_mode()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        if not texts:
            return torch.empty((0, EMBEDDING_DIM), dtype=torch.float32)
        values = self.model.encode(
            texts,
            batch_size=len(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vectors = np.asarray(values, dtype=np.float32)
        if vectors.shape != (len(texts), EMBEDDING_DIM):
            raise ValueError(f"Expected Qwen3-VL queries ({len(texts)}, {EMBEDDING_DIM}), got {vectors.shape}")
        if not np.isfinite(vectors).all():
            raise ValueError("Qwen3-VL query embeddings contain NaN/Inf")
        norms = np.linalg.norm(vectors, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3, rtol=0):
            raise ValueError(f"Qwen3-VL query L2 norms outside tolerance: {norms.min()}..{norms.max()}")
        return torch.from_numpy(np.ascontiguousarray(vectors))


def load(
    model_name: str,
    pretrained: str | None = None,
    precision: str = "fp16",
    device_name: str = "auto",
    logger: logging.Logger | None = None,
    engine: str = "auto",
    vllm_url: str | None = None,
    **_: object,
) -> LoadedModel:
    if model_name != MODEL_ID:
        raise ValueError(f"Qwen3-VL embedding backend only supports {MODEL_ID}, got {model_name}")
    if precision not in {"fp16", "fp32"}:
        raise ValueError("Qwen3-VL embedding backend supports fp16/fp32 only")

    selected = resolve_engine(engine, vllm_url, logger)
    if selected == "vllm":
        if logger:
            logger.info("Using vLLM Qwen3-VL embedding server")
        return LoadedModel(
            model=QwenVllmEncoder(vllm_url),
            preprocess=None,
            device=torch.device("cpu"),
            precision=precision,
            embedding_dim=EMBEDDING_DIM,
            supports_text=True,
        )

    from sentence_transformers import SentenceTransformer

    device = resolve_device(device_name)
    effective_precision = "fp16" if device.type == "cuda" and precision == "fp16" else "fp32"
    if logger:
        logger.info("Loading %s on %s", MODEL_ID, device)
    model = SentenceTransformer(
        MODEL_ID,
        device=str(device),
        model_kwargs={"torch_dtype": torch.float16 if effective_precision == "fp16" else torch.float32},
    ).eval()
    actual_dim = model.get_sentence_embedding_dimension()
    if actual_dim is not None and actual_dim != EMBEDDING_DIM:
        raise ValueError(f"{MODEL_ID} dimension {actual_dim}, expected {EMBEDDING_DIM}")
    return LoadedModel(
        model=Qwen3VLEncoder(
            model,
            method="encode",
            normalize_embeddings=True,
            embedding_dim=EMBEDDING_DIM,
        ),
        preprocess=None,
        device=device,
        precision=effective_precision,
        embedding_dim=EMBEDDING_DIM,
        supports_text=True,
    )
