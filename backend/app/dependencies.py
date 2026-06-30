"""backend.app.dependencies — FastAPI dependency providers.

Provides singleton instances of the heavy objects (retrieval system, speech
model) so every route module can access them via ``Depends()`` without
re-initialising on every request.

Usage in a route::

    from fastapi import Depends
    from app.dependencies import get_retrieval_system
    from src.index.retrieval_system import FaissRetrievalSystem

    @router.post("/search")
    async def search(
        request: SearchRequest,
        rs: FaissRetrievalSystem = Depends(get_retrieval_system),
    ) -> ...
"""
from __future__ import annotations

from functools import lru_cache
from typing import Generator

from app.config import CFG, FAISS_INDEX_PATH, METADATA_PATH, VECTOR_CACHE_PATH
from src.index.retrieval_system import FaissRetrievalSystem

# ---------------------------------------------------------------------------
# Retrieval system — initialised once at import time
# ---------------------------------------------------------------------------

_retrieval_system: FaissRetrievalSystem | None = None


def _build_retrieval_system() -> FaissRetrievalSystem:
    return FaissRetrievalSystem(
        index_path=str(FAISS_INDEX_PATH),
        metadata_path=str(METADATA_PATH),
        model_name=CFG["model"]["name"],
        pretrained=CFG["model"]["pretrained"],
        device=CFG["model"].get("device", "auto"),
        precision=CFG["model"].get("precision", "fp32"),
        normalize=bool(CFG["model"].get("normalize", True)),
        ef_search=int(CFG.get("faiss", {}).get("ef_search", 64)),
        faiss_threads=CFG.get("faiss", {}).get("threads"),
        cache_index_vectors=CFG.get("faiss", {}).get("cache_index_vectors"),
        vector_cache_mode=CFG.get("faiss", {}).get("vector_cache_mode"),
        vector_cache_dtype=CFG.get("faiss", {}).get("vector_cache_dtype", "float32"),
        vector_cache_path=str(VECTOR_CACHE_PATH),
        allow_npy_fallback=bool(CFG.get("faiss", {}).get("allow_npy_fallback", False)),
        compile_model=bool(CFG.get("model", {}).get("compile", False)),
    )


def init_retrieval_system() -> None:
    """Must be called once during application startup (see ``main.py``)."""
    global _retrieval_system
    _retrieval_system = _build_retrieval_system()


def get_retrieval_system() -> FaissRetrievalSystem:
    """FastAPI dependency — returns the already-initialised retrieval system."""
    if _retrieval_system is None:  # pragma: no cover
        raise RuntimeError(
            "Retrieval system not initialised. Call init_retrieval_system() at startup."
        )
    return _retrieval_system


# ---------------------------------------------------------------------------
# Speech model — lazy-loaded on first request
# ---------------------------------------------------------------------------

_speech_model = None


@lru_cache(maxsize=1)
def _speech_cfg() -> dict:
    return CFG.get("speech", {})


def get_speech_model():
    """Lazy-load Whisper on first call; subsequent calls return the singleton."""
    global _speech_model
    if _speech_model is None:
        from faster_whisper import WhisperModel  # noqa: PLC0415 — deferred import

        cfg = _speech_cfg()
        _speech_model = WhisperModel(
            cfg.get("model_size", "base"),
            device=cfg.get("device", "cpu"),
            compute_type=cfg.get("compute_type", "int8"),
        )
    return _speech_model
