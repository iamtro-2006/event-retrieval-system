"""Build a contiguous vector cache .npy file from embedding files or an existing cache.

Originally part of indexer/faiss/vector_cache.py, now backend-agnostic.
Can rebuild the .npy memmap from individual embedding .npy files if the
FAISS index is no longer available.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.lib.format import open_memmap

from src.retrieval.indexer.embedding_loader import collect_embedding_files


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_path_text(path_value: str | Path) -> str:
    return str(path_value or "").replace("\\", "/")


def resolve_backend_path(backend_dir: Path, path_value: str | Path) -> Path:
    path = Path(normalize_path_text(path_value))
    return path if path.is_absolute() else backend_dir / path


def normalize_dtype(dtype_name: str) -> np.dtype:
    dtype_name = str(dtype_name or "float16").lower().strip()
    if dtype_name in {"fp16", "float16", "f16", "half"}:
        return np.dtype(np.float16)
    if dtype_name in {"fp32", "float32", "f32"}:
        return np.dtype(np.float32)
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def build_vector_cache_from_embeddings(
    embeddings_root: Path,
    output_path: Path,
    dtype: np.dtype,
    batch_size: int,
    normalize: bool,
) -> None:
    embedding_paths = collect_embedding_files(embeddings_root)
    if not embedding_paths:
        raise RuntimeError(f"No embedding files found in: {embeddings_root}")

    n = len(embedding_paths)
    sample = np.load(embedding_paths[0])
    dim = int(sample.shape[-1])

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[VECTOR CACHE BUILD] output={output_path} | shape=({n}, {dim}) | "
        f"dtype={dtype} | batch_size={batch_size} | normalize={normalize}"
    )

    vectors = open_memmap(
        str(output_path),
        mode="w+",
        dtype=dtype,
        shape=(n, dim),
    )

    for start in range(0, n, batch_size):
        count = min(batch_size, n - start)
        batch = np.empty((count, dim), dtype=np.float32)
        for i, path in enumerate(embedding_paths[start : start + count]):
            batch[i] = np.load(path).astype(np.float32).reshape(-1)

        if normalize:
            norms = np.linalg.norm(batch, axis=1, keepdims=True)
            batch /= np.clip(norms, 1e-12, None)

        vectors[start : start + count] = batch.astype(dtype, copy=False)
        vectors.flush()

        done = start + count
        print(f"[VECTOR CACHE BUILD] {done}/{n} ({done / max(1, n) * 100:.1f}%)")

    del vectors

    check = np.load(str(output_path), mmap_mode="r")
    if tuple(check.shape) != (n, dim):
        raise RuntimeError(f"Invalid output shape: {check.shape}, expected {(n, dim)}")

    print(
        f"[VECTOR CACHE BUILD] done | path={output_path} | "
        f"size={check.nbytes / (1024**2):.2f} MB"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a vector cache .npy file from embeddings."
    )
    parser.add_argument(
        "--config", default="configs/app.yaml", help="Path to backend app.yaml"
    )
    parser.add_argument("--output", default=None, help="Override output .npy path")
    parser.add_argument("--dtype", default=None, help="float16 or float32")
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument(
        "--no-normalize", action="store_true", help="Do not L2-normalize vectors"
    )
    parser.add_argument(
        "--embeddings-root", default=None, help="Override embeddings root directory"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    backend_dir = Path(__file__).resolve().parents[2]

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = backend_dir / config_path

    cfg = load_yaml(config_path)
    faiss_cfg = cfg.get("faiss", {})
    model_cfg = cfg.get("model", {})

    if args.embeddings_root:
        embeddings_root = Path(args.embeddings_root)
    else:
        indexing_cfg_path = backend_dir / "configs" / "indexing.yaml"
        if indexing_cfg_path.exists():
            indexing_cfg = load_yaml(indexing_cfg_path)
            embeddings_root = resolve_backend_path(
                backend_dir,
                indexing_cfg.get(
                    "embeddings_root",
                    "data/processed/embeddings_clip_vitl16_siglip_256",
                ),
            )
        else:
            embeddings_root = (
                backend_dir / "data/processed/embeddings_clip_vitl16_siglip_256"
            )

    output_path = (
        Path(args.output)
        if args.output
        else resolve_backend_path(
            backend_dir,
            faiss_cfg.get(
                "vector_cache_path",
                "data/database/faiss_hnsw_clip_vitl16_siglip_256/vectors_fp32.npy",
            ),
        )
    )
    if not output_path.is_absolute():
        output_path = backend_dir / output_path

    dtype = normalize_dtype(
        args.dtype or faiss_cfg.get("vector_cache_dtype", "float16")
    )
    normalize = bool(model_cfg.get("normalize", True)) and not args.no_normalize

    build_vector_cache_from_embeddings(
        embeddings_root=embeddings_root,
        output_path=output_path,
        dtype=dtype,
        batch_size=max(1, int(args.batch_size)),
        normalize=normalize,
    )


if __name__ == "__main__":
    main()
