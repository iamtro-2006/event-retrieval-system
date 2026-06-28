from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from src.index.query_planning import QueryPlan, SearchMode


def create_retrieval_backend(cfg: dict[str, Any]) -> "RetrievalBackend":
    """Factory that selects the retrieval backend based on cfg['backend'].

    Supports ``backend: milvus`` and ``backend: faiss`` (legacy fallback).
    """
    backend_name = str(cfg.get("backend", "milvus")).strip().lower()

    model_cfg = cfg.get("model", {})

    if backend_name == "milvus":
        from src.index.milvus_retrieval import MilvusRetrievalSystem

        milvus_cfg = cfg.get("milvus", {})
        search_params = milvus_cfg.get("search_params", {})
        return MilvusRetrievalSystem(
            host=milvus_cfg.get("host", "localhost"),
            port=int(milvus_cfg.get("port", 19530)),
            collection_name=milvus_cfg.get("collection_name", "keyframes"),
            model_name=model_cfg.get("name", "ViT-L-16-SigLIP-256"),
            pretrained=model_cfg.get("pretrained", "webli"),
            device=model_cfg.get("device", "auto"),
            precision=model_cfg.get("precision", "fp32"),
            normalize=bool(model_cfg.get("normalize", True)),
            consistency_level=milvus_cfg.get("consistency_level", "Bounded"),
            default_ef=int(search_params.get("params", {}).get("ef", 64)),
            metric_type=search_params.get("metric_type", "IP"),
            compile_model=bool(model_cfg.get("compile", False)),
        )

    if backend_name == "faiss":
        from src.index.retrieval_system import FaissRetrievalSystem

        faiss_cfg = cfg.get("faiss", {})
        return FaissRetrievalSystem(
            index_path=faiss_cfg["index_path"],
            metadata_path=faiss_cfg["metadata_path"],
            model_name=model_cfg.get("name", "ViT-L-16-SigLIP-256"),
            pretrained=model_cfg.get("pretrained", "webli"),
            device=model_cfg.get("device", "auto"),
            precision=model_cfg.get("precision", "fp32"),
            normalize=bool(model_cfg.get("normalize", True)),
            ef_search=int(faiss_cfg.get("ef_search", 64)),
            faiss_threads=faiss_cfg.get("threads"),
            cache_index_vectors=faiss_cfg.get("cache_index_vectors", None),
            vector_cache_mode=faiss_cfg.get("vector_cache_mode", None),
            vector_cache_dtype=faiss_cfg.get("vector_cache_dtype", "float32"),
            vector_cache_path=faiss_cfg.get("vector_cache_path", None),
            compile_model=bool(model_cfg.get("compile", False)),
        )

    raise ValueError(
        f"Unknown backend: '{backend_name}'. Supported: 'milvus', 'faiss'."
    )


@runtime_checkable
class RetrievalBackend(Protocol):
    """Abstract retrieval backend interface.

    Implemented by FaissRetrievalSystem (legacy) and MilvusRetrievalSystem.
    """

    @property
    def dim(self) -> int: ...

    @property
    def num_entities(self) -> int: ...

    @property
    def cache_info(self) -> dict[str, Any]: ...

    def encode_texts(self, queries: list[str]) -> np.ndarray: ...

    def encode_image(self, image_path: str | Path) -> np.ndarray: ...

    def run_search(
        self,
        query: str,
        mode: SearchMode = "semantic",
        use_split: bool = True,
        top_k: int = 10,
        candidate_multiplier: int = 5,
        duration_limit: float = -1,
        search_ef: int | None = None,
    ) -> tuple[pd.DataFrame, QueryPlan]: ...

    def multi_query_search(
        self,
        queries: list[str],
        top_k: int = 10,
        candidate_k: int | None = None,
        query_embeddings: np.ndarray | None = None,
    ) -> pd.DataFrame: ...

    def semantic_search(
        self,
        events: list[list[str]],
        top_k: int = 10,
        candidate_k: int = 500,
        ef: int | None = None,
    ) -> pd.DataFrame: ...

    def temporal_search(
        self,
        events: list[list[str]],
        top_k: int = 10,
        candidate_k: int = 500,
        duration_limit: float = -1,
        ef: int | None = None,
    ) -> pd.DataFrame: ...

    def similarity_search_by_image(
        self,
        image_path: str | Path,
        top_k: int = 20,
    ) -> pd.DataFrame: ...

    def search(self, query: str, top_k: int = 10) -> pd.DataFrame: ...

    def get_frame(self, video_id: str, keyframe_id: int) -> dict[str, Any] | None: ...

    def get_video_frames(self, video_id: str) -> list[dict[str, Any]]: ...

    def build_query_plan(
        self,
        query: str,
        mode: SearchMode = "semantic",
        use_split: bool = True,
    ) -> QueryPlan: ...
