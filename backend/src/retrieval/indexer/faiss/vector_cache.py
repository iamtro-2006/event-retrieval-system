from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import yaml
from numpy.lib.format import open_memmap


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


def build_vector_cache(
    index_path: Path,
    output_path: Path,
    dtype: np.dtype,
    batch_size: int,
    normalize: bool,
) -> None:
    print(f"[VECTOR CACHE BUILD] reading index: {index_path}")
    index = faiss.read_index(str(index_path))

    n = int(index.ntotal)
    d = int(index.d)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        "[VECTOR CACHE BUILD] output="
        f"{output_path} | shape=({n}, {d}) | dtype={dtype} | "
        f"batch_size={batch_size} | normalize={normalize}"
    )

    vectors = open_memmap(
        str(output_path),
        mode="w+",
        dtype=dtype,
        shape=(n, d),
    )

    for start in range(0, n, batch_size):
        count = min(batch_size, n - start)

        batch = np.empty((count, d), dtype=np.float32)
        index.reconstruct_n(start, count, batch)

        if normalize:
            faiss.normalize_L2(batch)

        vectors[start:start + count] = batch.astype(dtype, copy=False)
        vectors.flush()

        done = start + count
        print(f"[VECTOR CACHE BUILD] {done}/{n} ({done / max(1, n) * 100:.1f}%)")

    del vectors

    check = np.load(str(output_path), mmap_mode="r")
    if tuple(check.shape) != (n, d):
        raise RuntimeError(f"Invalid output shape: {check.shape}, expected {(n, d)}")

    print(
        "[VECTOR CACHE BUILD] done | "
        f"path={output_path} | size={check.nbytes / (1024 ** 2):.2f} MB"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a contiguous FAISS vector cache .npy file.")
    parser.add_argument("--config", default="configs/app.yaml", help="Path to backend app.yaml")
    parser.add_argument("--output", default=None, help="Override output .npy path")
    parser.add_argument("--dtype", default=None, help="float16 or float32")
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--no-normalize", action="store_true", help="Do not L2-normalize reconstructed vectors")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # File is expected at backend/src/pipelines/build_vector_cache.py
    backend_dir = Path(__file__).resolve().parents[2]

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = backend_dir / config_path

    cfg = load_yaml(config_path)
    faiss_cfg = cfg.get("faiss", {})
    model_cfg = cfg.get("model", {})

    index_path = resolve_backend_path(backend_dir, faiss_cfg["index_path"])

    output_path = (
        Path(args.output)
        if args.output
        else resolve_backend_path(
            backend_dir,
            faiss_cfg.get(
                "vector_cache_path",
                "data/database/faiss_hnsw_clip_vitl16_siglip_256/vectors_fp16.npy",
            ),
        )
    )
    if not output_path.is_absolute():
        output_path = backend_dir / output_path

    dtype = normalize_dtype(args.dtype or faiss_cfg.get("vector_cache_dtype", "float16"))
    normalize = bool(model_cfg.get("normalize", True)) and not args.no_normalize

    build_vector_cache(
        index_path=index_path,
        output_path=output_path,
        dtype=dtype,
        batch_size=max(1, int(args.batch_size)),
        normalize=normalize,
    )

if __name__ == "__main__":
    main()
