from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Literal, TYPE_CHECKING

import faiss
import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image

from src.logic.temporal_search import temporal_search_from_candidates

if TYPE_CHECKING:
    from src.ocr.pipelines.search import SearchPipeline as OCRSearchPipeline
    from src.asr.pipelines.search import SearchPipeline as ASRSearchPipeline

SearchMode = Literal["semantic", "temporal", "ocr", "asr", "auto"]

# Columns copied verbatim from FAISS metadata rows when enriching OCR/ASR hits,
# so that OCR/ASR results share an identical shape with semantic search results
# (and the frontend can render both with the same component).
_METADATA_DISPLAY_COLUMNS = (
    "dataset",
    "video_id",
    "keyframe_id",
    "keyframe_id_int",
    "source_name",
    "frame_idx",
    "timestamp_sec",
    "fps",
    "keyframe_path",
)


def resolve_device(device_name: str) -> torch.device:
    """Resolve the target device for PyTorch operations.

    Args:
        device_name: The requested device name ('auto', 'cuda', 'cuda:x', or 'cpu').

    Returns:
        The resolved `torch.device` object.
    """
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_name)
    return torch.device("cpu")


def _clean_queries(queries: list[str]) -> list[str]:
    """Clean, deduplicate, and normalize a list of query strings.

    Args:
        queries: A list of raw query strings.

    Returns:
        A list of cleaned, unique, and case-folded query strings.
    """
    cleaned: list[str] = []
    seen: set[str] = set()
    for query in queries:
        # Normalize whitespace and strip edges
        query = re.sub(r"\s+", " ", str(query or "").strip())
        key = query.casefold()
        if query and key not in seen:
            seen.add(key)
            cleaned.append(query)
    return cleaned


def split_temporal_events(query: str) -> list[str]:
    """Split a complex query into distinct temporal events.

    Semicolons and full-stops act as temporal separators, 
    while commas remain as semantic subqueries within an event.

    Args:
        query: The raw input query string.

    Returns:
        A list of cleaned temporal event strings.
    """
    return _clean_queries(re.split(r"[.;]+", str(query or "").replace("\n", " ")))


def split_semantic_queries(event: str) -> list[str]:
    """Split a temporal event into semantic subqueries.

    Args:
        event: A single temporal event string.

    Returns:
        A list containing the full event and its comma-separated semantic parts.
    """
    event = str(event or "").strip()
    if not event:
        return []
    return _clean_queries([event, *(part.strip() for part in event.split(","))])


@dataclass(frozen=True)
class QueryPlan:
    """Dataclass representing a structured execution plan for search queries."""
    
    query: str
    mode: SearchMode
    use_split: bool
    events: list[list[str]]

    @property
    def event_queries(self) -> list[str]:
        """Get the primary query string for each temporal event."""
        return [event[0] for event in self.events if event]

    @property
    def flat_queries(self) -> list[str]:
        """Get a flattened list of all semantic subqueries across all events."""
        return _clean_queries([query for event in self.events for query in event])


class FaissRetrievalSystem:
    """Thread-safe FAISS retrieval engine with multi-modal encoding and temporal search capabilities.

    Vector access modes:
    - ram + float32/float16: Reconstructs FAISS vectors into RAM once at startup.
    - memmap + float32/float16: Memory-maps a contiguous .npy file, reading only touched pages.
    - none: No temporal vector cache; temporal search requires `allow_npy_fallback`.
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
        allow_npy_fallback: bool = False,
        compile_model: bool = False,
        ocr_search_pipeline: "OCRSearchPipeline | None" = None,
        asr_search_pipeline: "ASRSearchPipeline | None" = None,
    ) -> None:
        """Initialize the retrieval system, load models, and configure caches.

        Args:
            index_path: Path to the FAISS index file.
            metadata_path: Path to the CSV metadata file.
            model_name: Name of the OpenCLIP model architecture.
            pretrained: Pretrained weights identifier for the model.
            device: Target device ('auto', 'cpu', 'cuda', 'cuda:x').
            precision: Model precision ('fp32', 'fp16', 'bf16', 'amp').
            normalize: Whether to L2-normalize embeddings.
            ef_search: HNSW efSearch parameter for FAISS.
            faiss_threads: Number of OpenMP threads for FAISS.
            cache_index_vectors: Legacy flag for RAM caching (overridden by vector_cache_mode).
            vector_cache_mode: Cache strategy ('ram', 'memmap', 'none').
            vector_cache_dtype: Data type for cached vectors ('float32', 'float16').
            vector_cache_path: Path to the .npy file for memmap mode.
            allow_npy_fallback: Allow fallback to disk reads if cache is missing.
            compile_model: Whether to apply `torch.compile` to the model.
            ocr_search_pipeline: Optional OCR `SearchPipeline` (Elasticsearch-backed).
                When provided, enables `mode="ocr"` in `run_search`/`search`. Injected
                as a dependency so this class stays decoupled from Elasticsearch.
            asr_search_pipeline: Optional ASR `SearchPipeline` (Elasticsearch-backed).
                When provided, enables `mode="asr"` in `run_search`/`search`. Injected
                the same way as `ocr_search_pipeline`.
        """
        if faiss_threads is None:
            faiss_threads = max(1, min(os.cpu_count() or 1, 12))
        faiss.omp_set_num_threads(int(faiss_threads))

        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.vector_cache_path = Path(vector_cache_path) if vector_cache_path else None
        self.allow_npy_fallback = bool(allow_npy_fallback)

        self.index = faiss.read_index(str(self.index_path))
        self._set_ef_search(int(ef_search))
        self._search_lock = RLock()

        self.metadata = pd.read_csv(self.metadata_path, low_memory=False)
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
        self.autocast_dtype = torch.float16 if precision in {"fp16", "amp"} else torch.bfloat16
        self.use_autocast = self.device.type == "cuda" and precision in {"amp", "fp16", "bf16"}

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
                self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=False)
            except Exception as exc:
                print(f"[MODEL] torch.compile skipped: {type(exc).__name__}: {exc}")

        # Backward compatibility with old config
        if vector_cache_mode is None:
            vector_cache_mode = "ram" if bool(cache_index_vectors) else "none"

        self.vector_cache_mode = str(vector_cache_mode or "none").strip().lower()
        self.vector_cache_dtype = self._normalize_cache_dtype(vector_cache_dtype)
        self._vector_cache: np.ndarray | np.memmap | None = self._init_vector_cache()

        # Optional OCR backend (Elasticsearch-based full-text search over on-screen text).
        self.ocr_search_pipeline: "OCRSearchPipeline | None" = ocr_search_pipeline

        # Optional ASR backend (Elasticsearch-based full-text search over speech transcripts).
        self.asr_search_pipeline: "ASRSearchPipeline | None" = asr_search_pipeline

    @staticmethod
    def _normalize_cache_dtype(dtype_name: str) -> str:
        """Normalize and validate the vector cache data type string."""
        dtype_name = str(dtype_name or "float32").strip().lower()
        if dtype_name in {"fp32", "float32", "f32"}:
            return "float32"
        if dtype_name in {"fp16", "float16", "f16", "half"}:
            return "float16"
        raise ValueError(f"Unsupported vector_cache_dtype: {dtype_name}")

    @property
    def cache_info(self) -> dict:
        """Return metadata about the current vector cache configuration."""
        cache = self._vector_cache
        return {
            "mode": self.vector_cache_mode,
            "dtype": self.vector_cache_dtype,
            "path": str(self.vector_cache_path) if self.vector_cache_path else "",
            "available": cache is not None,
            "shape": tuple(cache.shape) if cache is not None else None,
            "memory_mb": round(float(cache.nbytes) / (1024 ** 2), 2) if cache is not None else 0.0,
            "allow_npy_fallback": self.allow_npy_fallback,
        }

    def _set_ef_search(self, ef_search: int) -> None:
        """Configure the HNSW efSearch parameter for the FAISS index."""
        target = self.index
        if hasattr(target, "hnsw"):
            target.hnsw.efSearch = ef_search
            return
        # Handle wrappers such as IndexIDMap/PreTransform
        for attr in ("index", "base_index"):
            target = getattr(target, attr, None)
            if target is not None and hasattr(target, "hnsw"):
                target.hnsw.efSearch = ef_search
                return

    def _build_metadata_lookup(self) -> None:
        """Build fast lookup dictionaries for metadata rows by video and frame."""
        self._row_by_video_frame: dict[tuple[str, int], int] = {}
        self._rows_by_video: dict[str, np.ndarray] = {}
        
        frame_col = "keyframe_id_int" if "keyframe_id_int" in self.metadata else "keyframe_id"
        if frame_col in self.metadata:
            frame_values = pd.to_numeric(self.metadata[frame_col], errors="coerce").fillna(-1).astype(np.int64)
            videos = self.metadata["video_id"].astype(str).to_numpy()
            frames = frame_values.to_numpy()
            
            # Vectorized dictionary creation is significantly faster than a Python for-loop
            self._row_by_video_frame = dict(zip(zip(videos, frames), range(len(videos))))
            
        groups = self.metadata.groupby(self.metadata["video_id"].astype(str), sort=False).indices
        self._rows_by_video = {str(video): np.asarray(rows, dtype=np.int64) for video, rows in groups.items()}

    def _init_vector_cache(self) -> np.ndarray | np.memmap | None:
        """Initialize the vector cache based on the configured mode."""
        mode = self.vector_cache_mode
        if mode in {"none", "false", "off", "disable", "disabled"}:
            print("[VECTOR CACHE] mode=none | temporal search will require fallback or fail fast")
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
                    f"memory={cache.nbytes / (1024 ** 2):.2f} MB"
                )
            return cache

        if mode in {"memmap", "mmap", "memmap_fp32", "memmap_fp16"}:
            if mode.endswith("fp16"):
                self.vector_cache_dtype = "float16"
            elif mode.endswith("fp32"):
                self.vector_cache_dtype = "float32"
            return self._load_vector_memmap()

        raise ValueError(f"Unsupported vector_cache_mode: {self.vector_cache_mode}")

    def _reconstruct_index_vectors(self, dtype_name: str = "float32") -> np.ndarray | None:
        """Reconstruct all vectors from the FAISS index into a NumPy array."""
        try:
            vectors = np.empty((self.index.ntotal, self.index.d), dtype=np.float32)
            self.index.reconstruct_n(0, self.index.ntotal, vectors)
            if self.normalize:
                faiss.normalize_L2(vectors)
            if dtype_name == "float16":
                vectors = vectors.astype(np.float16, copy=False)
            return np.ascontiguousarray(vectors)
        except Exception as exc:
            print(f"[VECTOR CACHE] FAISS reconstruct failed: {type(exc).__name__}: {exc}")
            return None

    def _load_vector_memmap(self) -> np.memmap | None:
        """Load vectors from a memory-mapped .npy file."""
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

        expected_dtype = np.float16 if self.vector_cache_dtype == "float16" else np.float32
        if cache.dtype != expected_dtype:
            print(
                "[VECTOR CACHE] warning: config dtype does not match file dtype | "
                f"config={expected_dtype}, file={cache.dtype}"
            )

        print(
            "[VECTOR CACHE] mode=memmap | "
            f"dtype={cache.dtype} | shape={cache.shape} | "
            f"path={self.vector_cache_path} | file_size={cache.nbytes / (1024 ** 2):.2f} MB"
        )
        return cache

    @property
    def index_vectors(self) -> np.ndarray | np.memmap | None:
        """Return the cached index vectors."""
        return self._vector_cache

    def build_query_plan(self, query: str, mode: SearchMode = "semantic", use_split: bool = True) -> QueryPlan:
        """Parse and structure a raw query into a QueryPlan object."""
        query = str(query or "").strip()
        events = []
        for text in split_temporal_events(query):
            parts = split_semantic_queries(text) if use_split else _clean_queries([text])
            if parts:
                events.append(parts)
        return QueryPlan(query=query, mode=mode, use_split=use_split, events=events)

    @torch.inference_mode()
    def encode_text(self, query: str) -> np.ndarray:
        """Encode a single text query into an embedding vector."""
        return self.encode_texts([query])

    @torch.inference_mode()
    def encode_texts(self, queries: list[str]) -> np.ndarray:
        """Encode a batch of text queries into normalized embedding vectors.

        Args:
            queries: List of text strings to encode.

        Returns:
            A contiguous float32 NumPy array of shape (n_queries, embedding_dim).
        """
        queries = _clean_queries(queries)
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
        """Encode an image file into a normalized embedding vector."""
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

    def _faiss_search(self, embeddings: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Execute a thread-safe FAISS similarity search."""
        k = min(max(1, int(k)), self.index.ntotal)
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        with self._search_lock:
            return self.index.search(embeddings, k)

    def _results_for_queries(
        self, embeddings: np.ndarray, queries: list[str], candidate_k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Retrieve raw FAISS scores and indices for a batch of embeddings."""
        if embeddings.size == 0:
            return np.empty((0, 0), np.float32), np.empty((0, 0), np.int64)
        return self._faiss_search(embeddings, candidate_k)

    def _aggregate_multi_query(
        self,
        scores: np.ndarray,
        indices: np.ndarray,
        queries: list[str],
        top_k: int,
    ) -> pd.DataFrame:
        """Aggregate and rank multi-query search results using highly optimized NumPy vectorization.

        Args:
            scores: Matrix of similarity scores (n_queries, candidate_k).
            indices: Matrix of FAISS indices (n_queries, candidate_k).
            queries: List of original query strings.
            top_k: Number of top results to return.

        Returns:
            A DataFrame containing the aggregated, ranked metadata records.
        """
        if indices.size == 0:
            return pd.DataFrame()
            
        n_queries = len(queries)
        valid = indices >= 0
        if not np.any(valid):
            return pd.DataFrame()

        flat_ids = indices[valid].astype(np.int32, copy=False)
        flat_scores = scores[valid].astype(np.float32, copy=False)
        query_ids = np.repeat(np.arange(n_queries, dtype=np.int32), indices.shape[1])[valid.ravel()]

        # Sort by (flat_ids, query_ids) to group by ID and then by query
        order = np.lexsort((query_ids, flat_ids))
        ids_sorted = flat_ids[order]
        scores_sorted = flat_scores[order]
        q_sorted = query_ids[order]

        # Find boundaries of each unique ID
        id_change = np.ones(len(ids_sorted), dtype=np.bool_)
        id_change[1:] = ids_sorted[1:] != ids_sorted[:-1]
        starts = np.where(id_change)[0]
        
        unique_ids = ids_sorted[starts]
        counts = np.diff(np.append(starts, len(ids_sorted)))

        # Aggregate scores using reduceat (extremely fast C-level loops)
        score_sum = np.add.reduceat(scores_sorted, starts)
        max_score = np.maximum.reduceat(scores_sorted, starts)
        avg_score = score_sum / counts

        # Count unique matched queries per ID without Python loops
        q_change = np.ones(len(q_sorted), dtype=np.bool_)
        q_change[1:] = q_sorted[1:] != q_sorted[:-1]
        
        is_new_unique_q = q_change & ~id_change
        extra_unique_q = np.add.reduceat(is_new_unique_q.astype(np.int32), starts)
        matched = 1 + extra_unique_q
        
        coverage = matched.astype(np.float32) / max(1, n_queries)
        alignment = 0.8 * avg_score + 0.2 * coverage

        # Extract top_k indices
        rank_order = np.argsort(-alignment, kind="stable")[:top_k]
        
        rows: list[dict] = []
        metadata_records = self._metadata_records
        
        # Shallow copy metadata for the final top_k results
        for display_rank, pos in enumerate(rank_order, 1):
            idx = int(unique_ids[pos])
            item = metadata_records[idx].copy()
            item["avg_score"] = float(avg_score[pos])
            item["max_score"] = float(max_score[pos])
            item["matched_queries"] = int(matched[pos])
            item["coverage_score"] = float(coverage[pos])
            item["alignment_score"] = float(alignment[pos])
            item["retrieval_score"] = float(alignment[pos])
            item["display_rank"] = display_rank
            item["rank"] = display_rank
            rows.append(item)
            
        return pd.DataFrame.from_records(rows)

    def multi_query_search(
        self,
        queries: list[str],
        top_k: int = 10,
        candidate_k: int | None = None,
        query_embeddings: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Execute a multi-query search and aggregate the results."""
        queries = _clean_queries(queries)
        if not queries:
            return pd.DataFrame()
        candidate_k = max(int(candidate_k or top_k), int(top_k))
        embeddings = query_embeddings if query_embeddings is not None else self.encode_texts(queries)
        scores, indices = self._results_for_queries(embeddings, queries, candidate_k)
        return self._aggregate_multi_query(scores, indices, queries, int(top_k))

    def search(self, query: str, top_k: int = 10) -> pd.DataFrame:
        """Execute a simple single-query semantic search."""
        return self.multi_query_search([query], top_k, top_k)

    def similarity_search_by_image(self, image_path: str | Path, top_k: int = 20) -> pd.DataFrame:
        """Execute an image-to-image similarity search."""
        scores, indices = self._faiss_search(self.encode_image(image_path), top_k)
        rows = []
        for rank, idx in enumerate(indices[0], 1):
            if idx < 0:
                continue
            item = dict(self._metadata_records[int(idx)])
            item.update(rank=rank, display_rank=rank, score=float(scores[0, rank - 1]),
                        retrieval_score=float(scores[0, rank - 1]), query=str(image_path))
            rows.append(item)
        return pd.DataFrame.from_records(rows)

    def semantic_search(self, events: list[list[str]], top_k: int = 10, candidate_k: int = 500) -> pd.DataFrame:
        """Execute a semantic search across multiple events."""
        queries = _clean_queries([query for event in events for query in event])
        embeddings = self.encode_texts(queries)
        return self.multi_query_search(queries, top_k, candidate_k, embeddings)

    def _build_temporal_candidates(
        self,
        events: list[list[str]],
        all_queries: list[str],
        all_embeddings: np.ndarray,
        candidate_k: int,
    ) -> pd.DataFrame:
        """Build a unified candidate DataFrame for all temporal events."""
        offset = 0
        frames: list[pd.DataFrame] = []
        for event_idx, event_queries in enumerate(events):
            count = len(event_queries)
            event_emb = all_embeddings[offset:offset + count]
            offset += count
            
            event_results = self.multi_query_search(
                event_queries, candidate_k, candidate_k, query_embeddings=event_emb
            )
            if event_results.empty:
                continue
                
            # Use inplace operations where possible to reduce memory overhead
            event_results["candidate_score"] = event_results["retrieval_score"].to_numpy(np.float32)
            event_results["candidate_rank"] = np.arange(1, len(event_results) + 1, dtype=np.int32)
            event_results["sub_query_idx"] = np.int32(event_idx)
            event_results["sub_query"] = event_queries[0]
            frames.append(event_results)
            
        return pd.concat(frames, ignore_index=True, copy=False) if frames else pd.DataFrame()

    def temporal_search(
        self,
        events: list[list[str]],
        top_k: int = 10,
        candidate_k: int = 500,
        duration_limit: float = -1,
    ) -> pd.DataFrame:
        """Execute a temporal search across multiple sequential events."""
        if not events:
            return pd.DataFrame()
            
        candidate_k = max(int(candidate_k), int(top_k))
        all_queries = [query for event in events for query in event]
        
        # Exactly one model forward pass for all queries
        all_embeddings = self.encode_texts(all_queries)  

        # Event scoring uses each event's full description (first item)
        offsets = np.cumsum([0] + [len(event) for event in events[:-1]])
        event_embeddings = all_embeddings[offsets]
        event_queries = [event[0] for event in events]
        
        candidate_df = self._build_temporal_candidates(
            events, all_queries, all_embeddings, candidate_k
        )
        if candidate_df.empty:
            return pd.DataFrame()

        return temporal_search_from_candidates(
            query_embeddings=event_embeddings,
            sub_queries=event_queries,
            candidate_df=candidate_df,
            duration_limit=duration_limit,
            top_k_videos=top_k,
            max_sequences_per_video=3,
            overlap_threshold=0.6,
            embedding_matrix=self._vector_cache,
            allow_npy_fallback=self.allow_npy_fallback,
        )

    def _metadata_row_for_ocr_hit(self, video_id: str, keyframe_id: str) -> dict | None:
        """Resolve an OCR hit (video_id, keyframe_id) to its full FAISS metadata row.

        Reuses the same `(video_id, keyframe_id_int)` lookup table built for temporal
        search, so OCR results reference the exact same keyframe records (paths,
        timestamps, fps, ...) as semantic search.
        """
        try:
            keyframe_id_int = int(str(keyframe_id))
        except ValueError:
            return None
        row_idx = self._row_by_video_frame.get((str(video_id), keyframe_id_int))
        if row_idx is None:
            return None
        return self._metadata_records[row_idx]

    def _enrich_ocr_hits(self, hits: list[dict], top_k: int) -> pd.DataFrame:
        """Join raw OCR hits with FAISS metadata and normalize scores to `[0, 1]`.

        Hits whose (video_id, keyframe_id) cannot be resolved against the FAISS
        metadata (e.g. stale/partial OCR index) are dropped rather than surfaced
        as broken results.
        """
        if not hits:
            return pd.DataFrame()

        max_score = max(float(hit["score"]) for hit in hits) or 1.0

        rows: list[dict] = []
        for hit in hits:
            meta = self._metadata_row_for_ocr_hit(hit["video_id"], hit["keyframe_id"])
            if meta is None:
                continue

            item = {col: meta.get(col) for col in _METADATA_DISPLAY_COLUMNS}
            ocr_score = float(hit["score"])
            normalized_score = ocr_score / max_score  # bound to (0, 1] for the frontend

            item.update(
                ocr_score=ocr_score,
                score=normalized_score,
                retrieval_score=normalized_score,
                matched_texts=hit["texts"],
                search_mode="ocr",
            )
            rows.append(item)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame.from_records(rows).sort_values("retrieval_score", ascending=False)
        df = df.head(int(top_k)).reset_index(drop=True)
        df["display_rank"] = np.arange(1, len(df) + 1)
        df["rank"] = df["display_rank"]
        return df

    def ocr_search(self, query: str, top_k: int = 10, oversample_factor: int = 3) -> pd.DataFrame:
        """Execute an OCR (on-screen text) search and return results shaped like semantic search.

        Args:
            query: Free-text query matched against OCR'd on-screen text.
            top_k: Number of results to return after metadata enrichment/dedup.
            oversample_factor: Fetch `top_k * oversample_factor` raw hits from
                Elasticsearch before enrichment, to compensate for hits dropped
                during metadata resolution (e.g. stale OCR entries).

        Returns:
            A DataFrame with the same display columns as `semantic_search`, plus
            `matched_texts` (OCR strings that matched) and `ocr_score` (raw BM25 score).

        Raises:
            RuntimeError: If no OCR backend was injected at construction time.
        """
        if self.ocr_search_pipeline is None:
            raise RuntimeError(
                "OCR search is not available: FaissRetrievalSystem was built without "
                "an `ocr_search_pipeline`. Build one via "
                "src.ocr.pipelines.factory.build_ocr_search_pipeline(...) and pass it in."
            )

        query = str(query or "").strip()
        if not query:
            return pd.DataFrame()

        raw_top_k = max(int(top_k) * max(1, int(oversample_factor)), int(top_k))
        hits = self.ocr_search_pipeline.search(query=query, top_k=raw_top_k)
        return self._enrich_ocr_hits(hits, top_k)

    def _metadata_rows_for_asr_hit(
        self, video_id: str, start_time: float, end_time: float
    ) -> list[dict]:
        """Resolve an ASR hit's speech segment to every keyframe of that video
        whose timestamp falls inside `[start_time, end_time]`.

        This is the ASR analogue of `_metadata_row_for_ocr_hit`: instead of a
        single (video_id, keyframe_id) lookup, an ASR hit is a time window, so
        it expands to a *set* of frames sharing the same `video_id`.
        """
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

    def _enrich_asr_hits(
        self, hits: list[dict], top_k: int, max_frames_per_hit: int = 50
    ) -> pd.DataFrame:
        """Join raw ASR hits (speech segments) with FAISS metadata and normalize scores.

        Unlike OCR (one keyframe per hit -> displayed exactly like semantic
        search), each ASR hit is a *spoken segment* that spans many keyframes.
        To make that visually distinct in the frontend (the same way
        `temporal_search` results are), every accepted segment is collapsed
        into a SINGLE result row shaped like a temporal-search hit:
          - a representative frame at the top level (used as the card cover),
          - a `matched_sequence` list containing every keyframe inside the
            segment's `[start_time, end_time]` window.

        The frontend has no ASR-specific code path: `ResultCard`/`GroupedResults`
        already render any row carrying a non-empty `matched_sequence` as a
        horizontal "temporal sequence" strip (see `TemporalSequence.jsx`), so
        populating `matched_sequence` here is what makes ASR hits *look like*
        temporal search, exactly as requested. Each frame's `sub_query` field
        is set to the segment's transcript text so it is shown under the frame.

        `top_k` limits the number of distinct segments (hits) returned, not
        the resulting frame count.
        """
        if not hits:
            return pd.DataFrame()

        max_score = max(float(hit["score"]) for hit in hits) or 1.0

        rows: list[dict] = []
        accepted = 0
        for hit in hits:
            if accepted >= int(top_k):
                break

            frames = self._metadata_rows_for_asr_hit(
                hit["video_id"], hit["start_time"], hit["end_time"]
            )
            if not frames:
                continue

            frames = frames[: int(max_frames_per_hit)]
            accepted += 1
            asr_score = float(hit["score"])
            normalized_score = asr_score / max_score  # bound to (0, 1] for the frontend
            transcript_text = str(hit["text"])

            # Build the per-frame chain shown inside the temporal-sequence strip.
            matched_sequence: list[dict] = []
            for seq_idx, meta in enumerate(frames):
                seq_item = {col: meta.get(col) for col in _METADATA_DISPLAY_COLUMNS}
                seq_item.update(
                    score=normalized_score,
                    candidate_score=normalized_score,
                    candidate_rank=seq_idx + 1,
                    sub_query_idx=seq_idx,
                    sub_query=transcript_text,
                )
                matched_sequence.append(seq_item)

            # Representative "cover" frame for the row: the middle keyframe of
            # the segment reads better as a thumbnail than the first/last one.
            anchor_meta = frames[len(frames) // 2]
            item = {col: anchor_meta.get(col) for col in _METADATA_DISPLAY_COLUMNS}
            item.update(
                asr_score=asr_score,
                score=normalized_score,
                retrieval_score=normalized_score,
                avg_score=normalized_score,
                video_score=normalized_score,
                matched_texts=[transcript_text],
                segment_id=hit["segment_id"],
                temporal_start_time=hit["start_time"],
                temporal_end_time=hit["end_time"],
                temporal_duration_sec=max(0.0, float(hit["end_time"]) - float(hit["start_time"])),
                search_mode="asr",
                display_rank=accepted,
                matched_sequence=matched_sequence,
            )
            rows.append(item)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame.from_records(rows).sort_values("display_rank", kind="stable")
        df = df.reset_index(drop=True)
        df["rank"] = df["display_rank"]
        return df

    def asr_search(
        self,
        query: str,
        top_k: int = 10,
        oversample_factor: int = 3,
        max_frames_per_hit: int = 50,
    ) -> pd.DataFrame:
        """Execute an ASR (speech transcript) search and return results shaped like semantic search.

        Args:
            query: Free-text query matched against ASR'd speech transcripts.
            top_k: Number of distinct speech segments (hits) to return after
                metadata enrichment/dedup. Each hit may expand to several rows
                (one per matching keyframe).
            oversample_factor: Fetch `top_k * oversample_factor` raw hits from
                Elasticsearch before enrichment, to compensate for hits dropped
                during metadata resolution (e.g. a segment with no keyframes in range).
            max_frames_per_hit: Safety cap on how many frames a single segment
                can expand to.

        Returns:
            A DataFrame with one row per matched speech segment, shaped like a
            `temporal_search` hit: the same display columns as
            `semantic_search`, plus `matched_texts` (the segment's transcript),
            `asr_score` (raw BM25 score), `segment_id`, and `matched_sequence`
            (every keyframe inside the segment's time window, so the frontend
            renders it as a temporal-sequence strip). `temporal.start_time`/
            `temporal.end_time` in the serialized API response carry the
            segment's `[start_time, end_time]` window.

        Raises:
            RuntimeError: If no ASR backend was injected at construction time.
        """
        if self.asr_search_pipeline is None:
            raise RuntimeError(
                "ASR search is not available: FaissRetrievalSystem was built without "
                "an `asr_search_pipeline`. Build one via "
                "src.asr.pipelines.factory.build_asr_search_pipeline(...) and pass it in."
            )

        query = str(query or "").strip()
        if not query:
            return pd.DataFrame()

        raw_top_k = max(int(top_k) * max(1, int(oversample_factor)), int(top_k))
        hits = self.asr_search_pipeline.search(query=query, top_k=raw_top_k)
        return self._enrich_asr_hits(hits, top_k, max_frames_per_hit)

    def run_search(
        self,
        query: str,
        mode: SearchMode = "semantic",
        use_split: bool = True,
        top_k: int = 10,
        candidate_multiplier: int = 5,
        duration_limit: float = -1,
    ) -> tuple[pd.DataFrame, QueryPlan]:
        """Execute a search based on the specified mode and query plan."""
        plan = self.build_query_plan(query, mode, use_split)
        if not plan.events:
            return pd.DataFrame(), plan
            
        candidate_k = max(int(top_k) * int(candidate_multiplier), int(top_k))
        
        if mode == "auto":
            effective_mode = "temporal" if len(plan.events) > 1 else "semantic"
        else:
            effective_mode = mode
            
        if effective_mode == "semantic":
            return self.semantic_search(plan.events, top_k, candidate_k), plan
        if effective_mode == "temporal":
            return self.temporal_search(plan.events, top_k, candidate_k, duration_limit), plan
        if effective_mode == "ocr":
            # OCR search operates on the raw query text (Elasticsearch does its own
            # tokenization/fuzzy matching), not the CLIP-oriented temporal/semantic split.
            return self.ocr_search(plan.query, top_k), plan
        if effective_mode == "asr":
            # ASR search operates on the raw query text (Elasticsearch does its own
            # tokenization/fuzzy matching), not the CLIP-oriented temporal/semantic split.
            return self.asr_search(plan.query, top_k), plan

        raise ValueError(f"Unsupported search mode: {effective_mode}")