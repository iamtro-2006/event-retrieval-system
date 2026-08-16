"""ClipMilvusIndex — Milvus + OpenCLIP infrastructure replacing ClipFaissIndex.

Exposes the same interface as `ClipFaissIndex` so downstream retrievers
(semantic_search, temporal_search, orchestrator) work unchanged:

- `.index` — a MilvusSearchAdapter with `.search(embeddings, k) -> (scores, indices)`
- `.search_lock` — RLock for thread-safe search
- `.metadata` / `.metadata_records` — pandas DataFrame + list[dict]
- `.index_vectors` — np.ndarray/memmap vector cache (for temporal DP)
- `.encode_texts()` / `.encode_image()` — OpenCLIP encoding
- `.metadata_row_for_ocr_hit()` / `.metadata_rows_for_asr_hit()` — metadata lookup
- `.cache_info` — dict for /api/health

3-Pillar Design:
  1. ANN search  -> Milvus (fast, scalable semantic vector search)
  2. Vector cache -> np.memmap .npy (temporal DP reads a RAM-backed matrix,
     avoids network round-trips per candidate frame)
  3. Metadata     -> pandas in RAM (O(1) OCR/ASR lookups, no per-hit Milvus query)

The Milvus collection uses a positional `row_id` (INT64 PK) equal to the
metadata/memmap row index, so `.search()` returns FAISS-shaped positional
indices directly — no secondary id mapping is required.
"""

from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image

from src.retrieval.index.constants import resolve_device


class MilvusSearchAdapter:
    """Wraps a Milvus Collection to expose a FAISS-compatible `.search()` interface.

    `faiss.Index.search(embeddings, k)` returns `(scores, indices)` where:
    - scores: (n_queries, k) float32, similarity (higher = better for IP)
    - indices: (n_queries, k) int64, positional row IDs 0..N-1

    The collection's `row_id` PK is the positional index inserted in
    metadata/memmap order, so this adapter maps each hit's PK straight back to
    a positional index with no extra lookup table.
    """

    def __init__(
        self,
        collection: Any,
        dim: int,
        metric_type: str = "IP",
        search_params: dict | None = None,
    ) -> None:
        self._collection = collection
        self._dim = int(dim)
        self._metric_type = metric_type
        self._search_params = dict(search_params or {})
        self._ntotal = int(collection.num_entities)

    @property
    def ntotal(self) -> int:
        return self._ntotal

    @property
    def d(self) -> int:
        return self._dim

    def search(self, embeddings: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Search the Milvus collection and return (scores, indices) matching FAISS shape."""
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        n_queries = embeddings.shape[0]
        k = min(max(1, int(k)), self._ntotal)

        results = self._collection.search(
            data=embeddings.tolist(),
            anns_field="embedding",
            param={"metric_type": self._metric_type, "params": self._search_params},
            limit=k,
            output_fields=["row_id"],
        )

        scores = np.full((n_queries, k), -1.0, dtype=np.float32)
        indices = np.full((n_queries, k), -1, dtype=np.int64)

        for qi, hits in enumerate(results):
            for ri, hit in enumerate(hits[:k]):
                scores[qi, ri] = float(hit.distance)
                try:
                    indices[qi, ri] = int(hit.entity.get("row_id"))
                except (TypeError, ValueError):
                    indices[qi, ri] = -1

        return scores, indices

    def reconstruct_n(self, start: int, count: int, vectors: np.ndarray) -> None:
        """No-op stub — Milvus does not support vector reconstruction.

        The vector cache is maintained as a separate .npy memmap file instead
        (3-pillar design, pillar 2).
        """
        raise NotImplementedError(
            "Milvus does not support reconstruct_n. Use the .npy vector cache."
        )


class ClipMilvusIndex:
    """Thread-safe Milvus + OpenCLIP embedding store.

    Drop-in replacement for ClipFaissIndex. Loads a Milvus collection +
    metadata CSV, loads OpenCLIP, encodes text/image, manages the vector cache
    (ram/memmap), and resolves metadata rows. The `.index` attribute is a
    MilvusSearchAdapter exposing `.search()` compatible with the FAISS-shaped
    calling code in semantic_search and temporal_search.
    """

    def __init__(
        self,
        milvus_host: str,
        milvus_port: str | int,
        collection_name: str,
        metadata_path: str | Path,
        model_name: str,
        pretrained: str,
        model_key: str | None = None,
        backend: str | None = None,
        device: str = "auto",
        precision: str = "fp32",
        normalize: bool = True,
        metric_type: str = "IP",
        search_params: dict | None = None,
        vector_cache_mode: str | None = None,
        vector_cache_dtype: str = "float32",
        vector_cache_path: str | Path | None = None,
        allow_npy_fallback: bool = False,
        compile_model: bool = False,
        **_extra: Any,
    ) -> None:
        from pymilvus import Collection, connections

        self.model_key = model_key or model_name
        self.model_name = model_name
        self.pretrained = pretrained
        self.backend = backend or "milvus"
        self.supports_text = True
        # FAISS-interface parity: there is no local index file for Milvus, so
        # expose a descriptive placeholder path (`.exists()` is honestly False).
        self.index_path = Path(
            f"milvus://{milvus_host}:{milvus_port}/{collection_name}"
        )

        connections.connect("default", host=str(milvus_host), port=str(milvus_port))

        self.collection = Collection(collection_name)
        self.collection.load()
        self.metric_type = str(metric_type or "IP").upper()

        self.metadata_path = Path(metadata_path)
        self.vector_cache_path = Path(vector_cache_path) if vector_cache_path else None
        self.allow_npy_fallback = bool(allow_npy_fallback)

        self.metadata = pd.read_csv(self.metadata_path, low_memory=False)
        self.metadata.reset_index(drop=True, inplace=True)
        self.metadata["_faiss_id"] = np.arange(len(self.metadata), dtype=np.int64)
        self.metadata_records: list[dict] = self.metadata.to_dict(orient="records")
        self._build_metadata_lookup()

        ntotal = int(self.collection.num_entities)
        if ntotal != len(self.metadata):
            raise ValueError(
                f"Metadata/collection mismatch: rows={len(self.metadata)}, ntotal={ntotal}"
            )
        dim = self._infer_dimension()

        self.index = MilvusSearchAdapter(
            collection=self.collection,
            dim=dim,
            metric_type=self.metric_type,
            search_params=search_params,
        )
        self.search_lock = RLock()

        self.device = resolve_device(device)
        self.normalize = bool(normalize)
        if self.device.type == "cpu" and precision in {"fp16", "amp", "bf16"}:
            precision = "fp32"
        self.precision = precision
        self.autocast_dtype = (
            torch.float16 if precision in {"fp16", "amp"} else torch.bfloat16
        )
        self.use_autocast = self.device.type == "cuda" and precision in {
            "amp",
            "fp16",
            "bf16",
        }

        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            precision=precision,
            device=self.device,
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        if compile_model and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(
                    self.model, mode="reduce-overhead", fullgraph=False
                )
            except Exception as exc:
                print(f"[MODEL] torch.compile skipped: {type(exc).__name__}: {exc}")

        if vector_cache_mode is None:
            vector_cache_mode = "none"

        self.vector_cache_mode = str(vector_cache_mode or "none").strip().lower()
        self.vector_cache_dtype = self._normalize_cache_dtype(vector_cache_dtype)
        self._vector_cache: np.ndarray | np.memmap | None = self._init_vector_cache()

    def _infer_dimension(self) -> int:
        """Infer embedding dimension from the Milvus collection schema."""
        for field in self.collection.schema.fields:
            if getattr(field, "name", None) == "embedding":
                dim = getattr(field, "dim", None)
                if dim:
                    return int(dim)
        return 0

    @staticmethod
    def _normalize_cache_dtype(dtype_name: str) -> str:
        dtype_name = str(dtype_name or "float32").strip().lower()
        if dtype_name in {"fp32", "float32", "f32"}:
            return "float32"
        if dtype_name in {"fp16", "float16", "f16", "half"}:
            return "float16"
        raise ValueError(f"Unsupported vector_cache_dtype: {dtype_name}")

    @property
    def cache_info(self) -> dict:
        cache = self._vector_cache
        return {
            "mode": self.vector_cache_mode,
            "dtype": self.vector_cache_dtype,
            "path": str(self.vector_cache_path) if self.vector_cache_path else "",
            "available": cache is not None,
            "shape": tuple(cache.shape) if cache is not None else None,
            "memory_mb": round(float(cache.nbytes) / (1024**2), 2)
            if cache is not None
            else 0.0,
            "allow_npy_fallback": self.allow_npy_fallback,
        }

    @property
    def index_vectors(self) -> np.ndarray | np.memmap | None:
        return self._vector_cache

    def _build_metadata_lookup(self) -> None:
        self._row_by_video_frame: dict[tuple[str, int], int] = {}
        self._rows_by_video: dict[str, np.ndarray] = {}

        frame_col = (
            "keyframe_id_int" if "keyframe_id_int" in self.metadata else "keyframe_id"
        )
        if frame_col in self.metadata:
            frame_values = (
                pd.to_numeric(self.metadata[frame_col], errors="coerce")
                .fillna(-1)
                .astype(np.int64)
            )
            videos = self.metadata["video_id"].astype(str).to_numpy()
            frames = frame_values.to_numpy()
            self._row_by_video_frame = dict(
                zip(zip(videos, frames), range(len(videos)))
            )

        groups = self.metadata.groupby(
            self.metadata["video_id"].astype(str), sort=False
        ).indices
        self._rows_by_video = {
            str(video): np.asarray(rows, dtype=np.int64)
            for video, rows in groups.items()
        }

    def _init_vector_cache(self) -> np.ndarray | np.memmap | None:
        mode = self.vector_cache_mode
        if mode in {"none", "false", "off", "disable", "disabled"}:
            print(
                "[VECTOR CACHE] mode=none | temporal search will require fallback or fail fast"
            )
            return None

        if mode in {"ram", "memory", "ram_fp32", "ram_fp16"}:
            if mode.endswith("fp16"):
                self.vector_cache_dtype = "float16"
            elif mode.endswith("fp32"):
                self.vector_cache_dtype = "float32"
            cache = self._load_vector_cache_np()
            if cache is not None:
                print(
                    "[VECTOR CACHE] mode=ram | "
                    f"dtype={cache.dtype} | shape={cache.shape} | "
                    f"memory={cache.nbytes / (1024**2):.2f} MB"
                )
            return cache

        if mode in {"memmap", "mmap", "memmap_fp32", "memmap_fp16"}:
            if mode.endswith("fp16"):
                self.vector_cache_dtype = "float16"
            elif mode.endswith("fp32"):
                self.vector_cache_dtype = "float32"
            return self._load_vector_memmap()

        raise ValueError(f"Unsupported vector_cache_mode: {self.vector_cache_mode}")

    def _load_vector_cache_np(self) -> np.ndarray | None:
        if self.vector_cache_path is None:
            print("[VECTOR CACHE] ram mode requested but vector_cache_path is empty")
            return None
        if not self.vector_cache_path.exists():
            print(f"[VECTOR CACHE] cache file not found: {self.vector_cache_path}")
            return None
        cache = np.load(str(self.vector_cache_path))
        expected_dtype = (
            np.float16 if self.vector_cache_dtype == "float16" else np.float32
        )
        if cache.dtype != expected_dtype:
            cache = cache.astype(expected_dtype, copy=False)
        return np.ascontiguousarray(cache)

    def _load_vector_memmap(self) -> np.memmap | None:
        if self.vector_cache_path is None:
            print("[VECTOR CACHE] memmap requested but vector_cache_path is empty")
            return None
        if not self.vector_cache_path.exists():
            print(
                "[VECTOR CACHE] memmap file not found: "
                f"{self.vector_cache_path}. Run scripts/retrieval/run.py --task build-vector-cache first."
            )
            return None

        cache = np.load(str(self.vector_cache_path), mmap_mode="r")
        expected_shape = (
            (len(self.metadata), self.index.d) if self.index.d > 0 else None
        )
        if expected_shape is not None and tuple(cache.shape) != expected_shape:
            print(
                f"[VECTOR CACHE] warning: shape mismatch | "
                f"file={cache.shape}, expected={expected_shape}"
            )

        expected_dtype = (
            np.float16 if self.vector_cache_dtype == "float16" else np.float32
        )
        if cache.dtype != expected_dtype:
            print(
                f"[VECTOR CACHE] warning: config dtype does not match file dtype | "
                f"config={expected_dtype}, file={cache.dtype}"
            )

        print(
            "[VECTOR CACHE] mode=memmap | "
            f"dtype={cache.dtype} | shape={cache.shape} | "
            f"path={self.vector_cache_path} | file_size={cache.nbytes / (1024**2):.2f} MB"
        )
        return cache

    @torch.inference_mode()
    def encode_text(self, query: str) -> np.ndarray:
        return self.encode_texts([query])

    @torch.inference_mode()
    def encode_texts(self, queries: list[str]) -> np.ndarray:
        from src.retrieval.retriever.semantic_search.pipeline.search import (
            clean_queries,
        )

        queries = clean_queries(queries)
        if not queries:
            return np.empty((0, self.index.d), dtype=np.float32)

        tokens = self.tokenizer(queries).to(self.device, non_blocking=True)
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.autocast_dtype,
            enabled=self.use_autocast,
        ):
            emb = self.model.encode_text(tokens)
            if self.normalize:
                emb = torch.nn.functional.normalize(emb, dim=-1)

        return np.ascontiguousarray(emb.float().cpu().numpy(), dtype=np.float32)

    @torch.inference_mode()
    def encode_image(self, image_path: str | Path) -> np.ndarray:
        with Image.open(image_path) as image:
            tensor = self.preprocess(image.convert("RGB")).unsqueeze(0)
        tensor = tensor.to(self.device, non_blocking=True)
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.autocast_dtype,
            enabled=self.use_autocast,
        ):
            emb = self.model.encode_image(tensor)
            if self.normalize:
                emb = torch.nn.functional.normalize(emb, dim=-1)
        return np.ascontiguousarray(emb.float().cpu().numpy(), dtype=np.float32)

    def metadata_row_for_ocr_hit(self, video_id: str, keyframe_id: str) -> dict | None:
        """Resolve OCR hit (video_id, keyframe_id) -> full metadata row."""
        try:
            keyframe_id_int = int(str(keyframe_id))
        except ValueError:
            return None
        row_idx = self._row_by_video_frame.get((str(video_id), keyframe_id_int))
        if row_idx is None:
            return None
        return self.metadata_records[row_idx]

    def metadata_rows_for_asr_hit(
        self, video_id: str, start_time: float, end_time: float
    ) -> list[dict]:
        """Resolve ASR hit (time window) -> all keyframes of that video in [start, end]."""
        row_ids = self._rows_by_video.get(str(video_id))
        if row_ids is None or len(row_ids) == 0:
            return []

        sub = self.metadata.iloc[row_ids]

        if "timestamp_sec" in sub.columns:
            ts = pd.to_numeric(sub["timestamp_sec"], errors="coerce")
        elif "frame_idx" in sub.columns and "fps" in sub.columns:
            frame_idx = pd.to_numeric(sub["frame_idx"], errors="coerce")
            fps = pd.to_numeric(sub["fps"], errors="coerce")
            ts = frame_idx / fps.replace(0, np.nan)
        else:
            return []

        mask = ts.notna() & (ts >= float(start_time)) & (ts <= float(end_time))
        matched = sub[mask]
        if matched.empty:
            return []

        return matched.to_dict(orient="records")
