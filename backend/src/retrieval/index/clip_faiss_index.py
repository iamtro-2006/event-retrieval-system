"""ClipFaissIndex — hạ tầng FAISS + OpenCLIP dùng chung cho MỌI retriever.

Đây là phần thay thế `semantic_search/models/semantic_index.py` (class
`SemanticIndex` cũ). Khác biệt cốt lõi: class này CHỈ lo hạ tầng — load FAISS
index/metadata/model, encode text/image, cache vector, resolve metadata row.
Nó KHÔNG có method `search()` nghiệp vụ — nghiệp vụ semantic/temporal nằm ở
class `SearchPipeline` riêng của từng module (cùng hình dạng với
`SearchPipeline` của ocr_search/asr_search), nhận instance của class này qua
constructor (composition), xem:
  - retriever/semantic_search/pipeline/search.py  (class SearchPipeline)
  - retriever/temporal_search/pipeline/search.py  (class SearchPipeline)
"""
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

# Cột metadata FAISS copy nguyên văn khi enrich OCR/ASR hits (orchestrator.py),
# để OCR/ASR trả về shape giống hệt semantic search.
METADATA_DISPLAY_COLUMNS = (
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
    """Resolve target device cho các phép toán PyTorch."""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_name)
    return torch.device("cpu")


class ClipFaissIndex:
    """Thread-safe FAISS + OpenCLIP embedding store.

    Chỉ chịu trách nhiệm: load FAISS index + metadata CSV, load model
    OpenCLIP, encode text/image, quản lý vector cache (ram/memmap), và resolve
    metadata row theo (video_id, keyframe) hoặc theo time window. Không biết
    gì về "semantic"/"temporal" — các retriever gọi vào `self.index`,
    `self.search_lock`, `self.metadata_records` + `encode_texts()` để tự thực
    hiện thuật toán search của riêng chúng.
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
    ) -> None:
        if faiss_threads is None:
            faiss_threads = max(1, min(os.cpu_count() or 1, 12))
        faiss.omp_set_num_threads(int(faiss_threads))

        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.vector_cache_path = Path(vector_cache_path) if vector_cache_path else None
        self.allow_npy_fallback = bool(allow_npy_fallback)

        self.index = faiss.read_index(str(self.index_path))
        self._set_ef_search(int(ef_search))
        self.search_lock = RLock()

        self.metadata = pd.read_csv(self.metadata_path, low_memory=False)
        if len(self.metadata) != self.index.ntotal:
            raise ValueError(
                f"Metadata/index mismatch: rows={len(self.metadata)}, ntotal={self.index.ntotal}"
            )

        self.metadata.reset_index(drop=True, inplace=True)
        self.metadata["_faiss_id"] = np.arange(len(self.metadata), dtype=np.int64)
        self.metadata_records: list[dict] = self.metadata.to_dict(orient="records")
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
            "memory_mb": round(float(cache.nbytes) / (1024 ** 2), 2) if cache is not None else 0.0,
            "allow_npy_fallback": self.allow_npy_fallback,
        }

    @property
    def index_vectors(self) -> np.ndarray | np.memmap | None:
        return self._vector_cache

    def _set_ef_search(self, ef_search: int) -> None:
        target = self.index
        if hasattr(target, "hnsw"):
            target.hnsw.efSearch = ef_search
            return
        for attr in ("index", "base_index"):
            target = getattr(target, attr, None)
            if target is not None and hasattr(target, "hnsw"):
                target.hnsw.efSearch = ef_search
                return

    def _build_metadata_lookup(self) -> None:
        self._row_by_video_frame: dict[tuple[str, int], int] = {}
        self._rows_by_video: dict[str, np.ndarray] = {}

        frame_col = "keyframe_id_int" if "keyframe_id_int" in self.metadata else "keyframe_id"
        if frame_col in self.metadata:
            frame_values = pd.to_numeric(self.metadata[frame_col], errors="coerce").fillna(-1).astype(np.int64)
            videos = self.metadata["video_id"].astype(str).to_numpy()
            frames = frame_values.to_numpy()
            self._row_by_video_frame = dict(zip(zip(videos, frames), range(len(videos))))

        groups = self.metadata.groupby(self.metadata["video_id"].astype(str), sort=False).indices
        self._rows_by_video = {str(video): np.asarray(rows, dtype=np.int64) for video, rows in groups.items()}

    def _init_vector_cache(self) -> np.ndarray | np.memmap | None:
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
        if self.vector_cache_path is None:
            print("[VECTOR CACHE] memmap requested but vector_cache_path is empty")
            return None
        if not self.vector_cache_path.exists():
            print(
                "[VECTOR CACHE] memmap file not found: "
                f"{self.vector_cache_path}. Run src/retrieval/indexer/faiss/vector_cache.py first."
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

    @torch.inference_mode()
    def encode_text(self, query: str) -> np.ndarray:
        return self.encode_texts([query])

    @torch.inference_mode()
    def encode_texts(self, queries: list[str]) -> np.ndarray:
        from src.retrieval.retriever.semantic_search.pipeline.search import clean_queries

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
        """Resolve OCR hit (video_id, keyframe_id) -> full FAISS metadata row."""
        try:
            keyframe_id_int = int(str(keyframe_id))
        except ValueError:
            return None
        row_idx = self._row_by_video_frame.get((str(video_id), keyframe_id_int))
        if row_idx is None:
            return None
        return self.metadata_records[row_idx]

    def metadata_rows_for_asr_hit(self, video_id: str, start_time: float, end_time: float) -> list[dict]:
        """Resolve ASR hit (time window) -> mọi keyframe của video đó trong [start, end]."""
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
