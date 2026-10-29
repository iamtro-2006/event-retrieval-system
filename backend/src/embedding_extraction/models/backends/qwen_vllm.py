"""Client for the Qwen3-VL pooling server; no vLLM package in the API process."""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
import torch

from src.embedding_extraction.models.backends.sentence_transformer_image import validate_image_batch


MODEL_ID = "Qwen/Qwen3-VL-Embedding-2B"
EMBEDDING_DIM = 2048
DEFAULT_URL = "http://127.0.0.1:8001/v1"
INSTRUCTION = "Represent the user's input."


def server_url(configured: str | None = None) -> str:
    url = (configured or os.getenv("QWEN_VLLM_URL") or DEFAULT_URL).rstrip("/")
    if not url.startswith(("http://", "https://")) or not url.endswith("/v1"):
        raise ValueError("QWEN_VLLM_URL must be an HTTP(S) URL ending in /v1")
    return url


def resolve_engine(engine: str | None = None, url: str | None = None, logger: logging.Logger | None = None) -> str:
    """Choose once at startup; never change embedding space within a request."""
    selected = (os.getenv("QWEN_EMBEDDING_ENGINE") or engine or "auto").strip().lower()
    if selected not in {"auto", "vllm", "sentence_transformer"}:
        raise ValueError(f"Invalid Qwen engine: {selected}")
    if selected == "sentence_transformer":
        return selected
    endpoint = server_url(url)
    try:
        response = httpx.get(f"{endpoint}/models", timeout=2.0)
        response.raise_for_status()
        ids = {item.get("id") for item in response.json().get("data", [])}
        if MODEL_ID not in ids:
            raise RuntimeError(f"{MODEL_ID} not served at {endpoint}")
        return "vllm"
    except (httpx.HTTPError, ValueError, KeyError, RuntimeError) as exc:
        if selected == "vllm":
            raise RuntimeError(f"Qwen vLLM unavailable at {endpoint}: {exc}") from exc
        if logger:
            logger.warning("Qwen vLLM unavailable (%s); selecting SentenceTransformer", exc)
        return "sentence_transformer"


class QwenVllmEncoder:
    embedding_dim = EMBEDDING_DIM
    engine = "vllm"

    def __init__(self, url: str | None = None, timeout: float = 120.0):
        self.url = server_url(url)
        self.client = httpx.Client(timeout=timeout)

    @staticmethod
    def _messages(content: list[dict]) -> list[dict]:
        # Follow the vLLM Qwen3-VL pooling example's chat template exactly.
        return [
            {"role": "system", "content": [{"type": "text", "text": INSTRUCTION}]},
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": ""}]},
        ]

    def _embed(self, content: list[dict]) -> np.ndarray:
        response = self.client.post(
            f"{self.url}/embeddings",
            json={
                "model": MODEL_ID,
                "messages": self._messages(content),
                "encoding_format": "float",
                "continue_final_message": True,
                "add_special_tokens": True,
            },
        )
        response.raise_for_status()
        values = response.json()["data"][0]["embedding"]
        vector = np.asarray(values, dtype=np.float32)
        if vector.shape != (EMBEDDING_DIM,) or not np.isfinite(vector).all():
            raise ValueError(f"Invalid Qwen vLLM embedding: shape={vector.shape}")
        norm = float(np.linalg.norm(vector))
        if norm <= 0:
            raise ValueError("Qwen vLLM returned a zero vector")
        return validate_image_batch([vector / norm], 1, EMBEDDING_DIM)[0]

    def encode_image_paths(self, paths: list[Path]) -> np.ndarray:
        if not paths:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

        def one(path: Path) -> np.ndarray:
            mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
            image = base64.b64encode(path.read_bytes()).decode("ascii")
            return self._embed([
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image}"}},
                {"type": "text", "text": ""},
            ])

        # One HTTP request per image. Concurrent requests allow vLLM's own
        # scheduler to batch them; caller's batch_size bounds concurrency.
        with ThreadPoolExecutor(max_workers=len(paths)) as pool:
            return np.stack(list(pool.map(one, paths))).astype(np.float32)

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        if not texts:
            return torch.empty((0, EMBEDDING_DIM), dtype=torch.float32)
        # Bound in-flight requests for corpus indexing; large ASR/caption
        # collections can contain many thousands of segments.
        with ThreadPoolExecutor(max_workers=min(8, len(texts))) as pool:
            vectors = list(pool.map(
                lambda text: self._embed([{"type": "text", "text": text}]), texts
            ))
        return torch.from_numpy(np.stack(vectors).astype(np.float32))

    def close(self) -> None:
        self.client.close()
