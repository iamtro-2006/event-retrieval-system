from __future__ import annotations

import os
from pathlib import Path
from threading import RLock

import faiss
import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image

from src.index.query_planning import (
    QueryPlan,
    SearchMode,
    aggregate_multi_query,
    build_query_plan,
    clean_queries,
    resolve_effective_mode,
)
from src.index.temporal import (
    Candidates,
    matches_to_dataframe,
    search as temporal_search,
)
from src.utils.device import resolve_device


class FaissRetrievalSystem:
    """Thread-safe retrieval engine preserving the existing API contract.

    Vector access modes:
    - ram + float32/float16: reconstruct FAISS vectors once at startup.
    - memmap + float32/float16: map one contiguous .npy file and read only touched pages.
    - none: no temporal vector cache; temporal search will raise at call time.
    """

    def __init__(
        self,
        index_path: str | Path,
        metadata_path: str | Path,
        model_name: str,
        pretrained: str,
        device: str = "auto",
        precision: str = "fp32",
        normalize: bool = True,
        ef_search: int = 64,
        faiss_threads: int | None = None,
        cache_index_vectors: bool | None = None,
        vector_cache_mode: str | None = None,
        vector_cache_dtype: str = "float32",
        vector_cache_path: str | Path | None = None,
        compile_model: bool = False,
    ) -> None:
        if faiss_threads is None:
            faiss_threads = max(1, min(os.cpu_count() or 1, 12))
        faiss.omp_set_num_threads(int(faiss_threads))

        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.vector_cache_path = Path(vector_cache_path) if vector_cache_path else None

        self.index = faiss.read_index(str(index_path))
        self._set_ef_search(int(ef_search))
        self._search_lock = RLock()

        self.metadata = pd.read_csv(metadata_path, low_memory=False)
        if len(self.metadata) != self.index.ntotal:
            raise ValueError(
                f"Metadata/index mismatch: rows={len(self.metadata)}, ntotal={self.index.ntotal}"
            )
        self.metadata.reset_index(drop=True, inplace=True)
        self.metadata["_faiss_id"] = np.arange(len(self.metadata), dtype=np.int64)
        self._metadata_records = self.metadata.to_dict(orient="records")
        self._build_metadata_lookup()

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

        # Backward compatibility with old config.
        if vector_cache_mode is None:
            vector_cache_mode = "ram" if bool(cache_index_vectors) else "none"

        self.vector_cache_mode = str(vector_cache_mode or "none").strip().lower()
        self.vector_cache_dtype = self._normalize_cache_dtype(vector_cache_dtype)
        self._vector_cache: np.ndarray | np.memmap | None = self._init_vector_cache()

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
        }

    def _set_ef_search(self, ef_search: int) -> None:
        target = self.index
        if hasattr(target, "hnsw"):
            target.hnsw.efSearch = ef_search
            return
        # Handle wrappers such as IndexIDMap/PreTransform.
        for attr in ("index", "base_index"):
            target = getattr(target, attr, None)
            if target is not None and hasattr(target, "hnsw"):
                target.hnsw.efSearch = ef_search
                return

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
            for idx, (video, frame) in enumerate(
                zip(self.metadata["video_id"].astype(str), frame_values)
            ):
                self._row_by_video_frame[(video, int(frame))] = idx
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

        if mode in {"ram", "memory", "reconstruct", "ram_fp32", "ram_fp16"}:
            if mode.endswith("fp16"):
                self.vector_cache_dtype = "float16"
            elif mode.endswith("fp32"):
                self.vector_cache_dtype = "float32"
            cache = self._reconstruct_index_vectors(dtype_name=self.vector_cache_dtype)
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

    def _reconstruct_index_vectors(
        self, dtype_name: str = "float32"
    ) -> np.ndarray | None:
        try:
            vectors = np.empty((self.index.ntotal, self.index.d), dtype=np.float32)
            self.index.reconstruct_n(0, self.index.ntotal, vectors)
            if self.normalize:
                faiss.normalize_L2(vectors)
            if dtype_name == "float16":
                vectors = vectors.astype(np.float16, copy=False)
            return np.ascontiguousarray(vectors)
        except Exception as exc:
            print(
                f"[VECTOR CACHE] FAISS reconstruct failed: {type(exc).__name__}: {exc}"
            )
            return None

    def _load_vector_memmap(self) -> np.memmap | None:
        if self.vector_cache_path is None:
            print("[VECTOR CACHE] memmap requested but vector_cache_path is empty")
            return None
        if not self.vector_cache_path.exists():
            print(
                "[VECTOR CACHE] memmap file not found: "
                f"{self.vector_cache_path}. Run src/pipelines/build_vector_cache.py first."
            )
            return None

        cache = np.load(str(self.vector_cache_path), mmap_mode="r")
        expected_shape = (self.index.ntotal, self.index.d)
        if tuple(cache.shape) != expected_shape:
            raise ValueError(
                f"Vector memmap shape mismatch: path={self.vector_cache_path}, "
                f"shape={cache.shape}, expected={expected_shape}"
            )

        expected_dtype = (
            np.float16 if self.vector_cache_dtype == "float16" else np.float32
        )
        if cache.dtype != expected_dtype:
            print(
                "[VECTOR CACHE] warning: config dtype does not match file dtype | "
                f"config={expected_dtype}, file={cache.dtype}"
            )

        print(
            "[VECTOR CACHE] mode=memmap | "
            f"dtype={cache.dtype} | shape={cache.shape} | "
            f"path={self.vector_cache_path} | file_size={cache.nbytes / (1024**2):.2f} MB"
        )
        return cache

    @property
    def index_vectors(self) -> np.ndarray | np.memmap | None:
        return self._vector_cache

    def build_query_plan(
        self, query: str, mode: SearchMode = "semantic", use_split: bool = True
    ) -> QueryPlan:
        return build_query_plan(query, mode, use_split)

    @torch.inference_mode()
    def encode_text(self, query: str) -> np.ndarray:
        return self.encode_texts([query])

    @torch.inference_mode()
    def encode_texts(self, queries: list[str]) -> np.ndarray:
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

    def _faiss_search(
        self, embeddings: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        k = min(max(1, int(k)), self.index.ntotal)
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        with self._search_lock:
            return self.index.search(embeddings, k)

    def _results_for_queries(
        self, embeddings: np.ndarray, queries: list[str], candidate_k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if embeddings.size == 0:
            return np.empty((0, 0), np.float32), np.empty((0, 0), np.int64)
        return self._faiss_search(embeddings, candidate_k)

    def multi_query_search(
        self,
        queries: list[str],
        top_k: int = 10,
        candidate_k: int | None = None,
        query_embeddings: np.ndarray | None = None,
    ) -> pd.DataFrame:
        queries = clean_queries(queries)
        if not queries:
            return pd.DataFrame()
        candidate_k = max(int(candidate_k or top_k), int(top_k))
        embeddings = (
            query_embeddings
            if query_embeddings is not None
            else self.encode_texts(queries)
        )
        scores, indices = self._results_for_queries(embeddings, queries, candidate_k)
        return aggregate_multi_query(
            scores,
            indices,
            queries,
            int(top_k),
            metadata_records=self._metadata_records,
        )

    def search(self, query: str, top_k: int = 10) -> pd.DataFrame:
        return self.multi_query_search([query], top_k, top_k)

    def similarity_search_by_image(
        self, image_path: str | Path, top_k: int = 20
    ) -> pd.DataFrame:
        scores, indices = self._faiss_search(self.encode_image(image_path), top_k)
        rows = []
        for rank, idx in enumerate(indices[0], 1):
            if idx < 0:
                continue
            item = dict(self._metadata_records[int(idx)])
            item.update(
                rank=rank,
                display_rank=rank,
                score=float(scores[0, rank - 1]),
                retrieval_score=float(scores[0, rank - 1]),
                query=str(image_path),
            )
            rows.append(item)
        return pd.DataFrame.from_records(rows)

    def semantic_search(
        self,
        events: list[list[str]],
        top_k: int = 10,
        candidate_k: int = 500,
        ef: int | None = None,
    ) -> pd.DataFrame:
        queries = clean_queries([query for event in events for query in event])
        embeddings = self.encode_texts(queries)
        return self.multi_query_search(queries, top_k, candidate_k, embeddings)

    def _build_temporal_candidates(
        self,
        events: list[list[str]],
        all_queries: list[str],
        all_embeddings: np.ndarray,
        candidate_k: int,
    ) -> pd.DataFrame:
        offset = 0
        frames: list[pd.DataFrame] = []
        for event_idx, event_queries in enumerate(events):
            count = len(event_queries)
            event_emb = all_embeddings[offset : offset + count]
            offset += count
            event_results = self.multi_query_search(
                event_queries, candidate_k, candidate_k, query_embeddings=event_emb
            )
            if event_results.empty:
                continue
            event_results = event_results.copy()
            event_results["candidate_score"] = event_results[
                "retrieval_score"
            ].to_numpy(np.float32)
            event_results["candidate_rank"] = np.arange(1, len(event_results) + 1)
            event_results["sub_query_idx"] = event_idx
            event_results["sub_query"] = event_queries[0]
            frames.append(event_results)
        return (
            pd.concat(frames, ignore_index=True, copy=False)
            if frames
            else pd.DataFrame()
        )

    def temporal_search(
        self,
        events: list[list[str]],
        top_k: int = 10,
        candidate_k: int = 500,
        duration_limit: float = -1,
        ef: int | None = None,
    ) -> pd.DataFrame:
        if not events:
            return pd.DataFrame()
        candidate_k = max(int(candidate_k), int(top_k))
        all_queries = [query for event in events for query in event]
        all_embeddings = self.encode_texts(all_queries)

        offsets = np.cumsum([0] + [len(event) for event in events[:-1]])
        event_embeddings = all_embeddings[offsets]
        event_queries = [event[0] for event in events]
        candidate_df = self._build_temporal_candidates(
            events, all_queries, all_embeddings, candidate_k
        )
        if candidate_df.empty:
            return pd.DataFrame()

        if self._vector_cache is None:
            raise RuntimeError(
                "Temporal search requires a vector cache. Enable "
                "vector_cache_mode in the faiss config."
            )

        candidates = self._build_temporal_candidates_object(candidate_df)
        matches = temporal_search(
            candidates=candidates,
            query_embeddings=event_embeddings,
            sub_queries=event_queries,
            duration_limit=duration_limit,
            top_k_videos=top_k,
        )
        return matches_to_dataframe(matches)

    def _build_temporal_candidates_object(
        self, candidate_df: pd.DataFrame
    ) -> Candidates:
        """Assemble a Candidates value object from the candidate DataFrame.

        Maps each candidate row to its positional index in the vector cache
        via the ``_faiss_id`` column injected at metadata load time.
        """
        faiss_ids = candidate_df["_faiss_id"].to_numpy(dtype=np.int64)
        return Candidates(
            video_id=candidate_df["video_id"].to_numpy(),
            timestamp_sec=pd.to_numeric(candidate_df["timestamp_sec"], errors="coerce")
            .fillna(0)
            .to_numpy(np.float32),
            row_index=faiss_ids,
            records=candidate_df.to_dict(orient="records"),
            embedding_matrix=self._vector_cache,
        )

    def run_search(
        self,
        query: str,
        mode: SearchMode = "semantic",
        use_split: bool = True,
        top_k: int = 10,
        candidate_multiplier: int = 5,
        duration_limit: float = -1,
        search_ef: int | None = None,
    ) -> tuple[pd.DataFrame, QueryPlan]:
        plan = self.build_query_plan(query, mode, use_split)
        if not plan.events:
            return pd.DataFrame(), plan
        candidate_k = max(int(top_k) * int(candidate_multiplier), int(top_k))
        effective_mode = resolve_effective_mode(mode, plan)
        if effective_mode == "semantic":
            return self.semantic_search(plan.events, top_k, candidate_k), plan
        if effective_mode == "temporal":
            return self.temporal_search(
                plan.events, top_k, candidate_k, duration_limit
            ), plan
        if effective_mode in {"ocr", "asr"}:
            raise NotImplementedError(
                f"Search mode '{effective_mode}' is not implemented yet"
            )
        raise ValueError(f"Unsupported search mode: {effective_mode}")

    @property
    def dim(self) -> int:
        return int(self.index.d)

    @property
    def num_entities(self) -> int:
        return int(self.index.ntotal)

    def get_frame(self, video_id: str, keyframe_id: int) -> dict | None:
        row_idx = self._row_by_video_frame.get((str(video_id), int(keyframe_id)))
        if row_idx is None:
            return None
        return dict(self._metadata_records[int(row_idx)])

    def get_video_frames(self, video_id: str) -> list[dict]:
        row_ids = self._rows_by_video.get(str(video_id))
        if row_ids is None or len(row_ids) == 0:
            return []
        frames = [dict(self._metadata_records[int(idx)]) for idx in row_ids]
        frames.sort(key=lambda r: int(r.get("keyframe_id_int", 0) or 0))
        return frames
