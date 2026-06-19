"""
Memmap-backed embedding store.

Consolidates per-frame .npy embeddings into a single memory-mapped file
for O(1) random access without per-file I/O overhead.

Layout on disk:
    <base_dir>/embeddings.dat   — float32 memmap [N, dim]
    <base_dir>/embeddings.idx   — JSON mapping: embedding_path -> row index

Usage:
    # One-time build (offline / startup):
    store = EmbeddingMemmapStore.build(embedding_paths, out_dir)

    # Fast lookup at query time:
    store = EmbeddingMemmapStore.load(out_dir)
    emb = store["/path/to/frame_001.npy"]        # np.ndarray [dim]
    batch = store.get_batch(list_of_paths)        # np.ndarray [k, dim]
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from tqdm import tqdm


class EmbeddingMemmapStore:
    """Read-only memmap view over a consolidated embedding file."""

    def __init__(self, data: np.ndarray, index: dict[str, int]):
        self._data = data          # memmap or ndarray [N, dim]
        self._index = index        # path_str -> row

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    @classmethod
    def build(
        cls,
        embedding_paths: list[str | Path],
        out_dir: str | Path,
        *,
        normalize: bool = True,
        dim: int | None = None,
    ) -> "EmbeddingMemmapStore":
        """
        Read every .npy, normalize, and write a single memmap + index.
        
        Parameters
        ----------
        embedding_paths : list of paths to individual .npy embedding files
        out_dir : directory to write embeddings.dat and embeddings.idx
        normalize : L2-normalize each embedding
        dim : embedding dimension (auto-detected from first file if None)
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Deduplicate while preserving order
        seen: dict[str, int] = {}
        unique_paths: list[str] = []

        for p in embedding_paths:
            key = str(p)
            if key not in seen:
                seen[key] = len(unique_paths)
                unique_paths.append(key)

        n = len(unique_paths)
        if n == 0:
            raise ValueError("No embedding paths provided")

        # Auto-detect dimension from first file
        if dim is None:
            sample = np.load(unique_paths[0])
            if sample.ndim == 2:
                sample = sample[0]
            dim = sample.shape[0]

        # Create memmap file
        dat_path = out_dir / "embeddings.dat"
        mmap = np.memmap(dat_path, dtype=np.float32, mode="w+", shape=(n, dim))

        index: dict[str, int] = {}

        for i, path_str in enumerate(tqdm(unique_paths, desc="Building memmap")):
            emb = np.load(path_str)
            if emb.ndim == 2:
                emb = emb[0]
            emb = emb.astype(np.float32)

            if normalize:
                norm_sq = float(np.dot(emb, emb))
                if norm_sq > 1e-24:
                    emb = emb / math.sqrt(norm_sq)

            mmap[i] = emb
            index[path_str] = i

        mmap.flush()

        # Write index
        idx_path = out_dir / "embeddings.idx"
        idx_path.write_text(json.dumps(index), encoding="utf-8")

        return cls(data=mmap, index=index)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    @classmethod
    def load(cls, store_dir: str | Path) -> "EmbeddingMemmapStore":
        """Load a previously-built memmap store."""
        store_dir = Path(store_dir)

        idx_path = store_dir / "embeddings.idx"
        index: dict[str, int] = json.loads(idx_path.read_text(encoding="utf-8"))

        n = len(index)
        if n == 0:
            raise ValueError("Empty embedding index")

        dat_path = store_dir / "embeddings.dat"

        # Infer dimension from file size
        file_bytes = dat_path.stat().st_size
        dim = file_bytes // (n * 4)  # float32 = 4 bytes

        mmap = np.memmap(dat_path, dtype=np.float32, mode="r", shape=(n, dim))
        return cls(data=mmap, index=index)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------
    def __contains__(self, path: str | Path) -> bool:
        return str(path) in self._index

    def __getitem__(self, path: str | Path) -> np.ndarray:
        """Return normalized embedding for a single path. O(1)."""
        row = self._index[str(path)]
        return np.array(self._data[row], dtype=np.float32)

    def get(self, path: str | Path, default=None) -> np.ndarray | None:
        key = str(path)
        if key in self._index:
            return np.array(self._data[self._index[key]], dtype=np.float32)
        return default

    def get_batch(self, paths: list[str | Path]) -> np.ndarray:
        """Return stacked embeddings [k, dim] for a list of paths."""
        rows = np.array([self._index[str(p)] for p in paths], dtype=np.int64)
        return np.array(self._data[rows], dtype=np.float32)

    @property
    def shape(self) -> tuple[int, int]:
        return self._data.shape

    def __len__(self) -> int:
        return len(self._index)
