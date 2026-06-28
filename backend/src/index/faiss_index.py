from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
import pandas as pd


def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


def create_faiss_index(
    matrix: np.ndarray,
    metric: str = "cosine",
    index_type: str = "hnsw",
    hnsw_m: int = 32,
    ef_construction: int = 200,
    ef_search: int = 64,
):
    """Build Flat or HNSW index. Cosine uses normalized vectors + inner product."""
    if matrix.ndim != 2:
        raise ValueError(f"Matrix must be 2D, got shape={matrix.shape}")
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    dim = matrix.shape[1]
    metric_name = str(metric).lower()
    metric_type = (
        faiss.METRIC_INNER_PRODUCT if metric_name == "cosine" else faiss.METRIC_L2
    )
    if metric_name == "cosine":
        faiss.normalize_L2(matrix)
    elif metric_name != "l2":
        raise ValueError(f"Unsupported metric: {metric}")

    if str(index_type).lower() == "hnsw":
        index = faiss.IndexHNSWFlat(dim, int(hnsw_m), metric_type)
        index.hnsw.efConstruction = int(ef_construction)
        index.hnsw.efSearch = int(ef_search)
    elif str(index_type).lower() == "flat":
        index = (
            faiss.IndexFlatIP(dim)
            if metric_type == faiss.METRIC_INNER_PRODUCT
            else faiss.IndexFlatL2(dim)
        )
    else:
        raise ValueError(f"Unsupported index_type: {index_type}")

    index.add(matrix)
    return index


def save_faiss_index(
    index,
    metadata: pd.DataFrame,
    output_dir: str | Path,
    index_name: str,
    metadata_name: str,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index_path = output_dir / index_name
    metadata_path = output_dir / metadata_name

    faiss.write_index(index, str(index_path))
    metadata.to_csv(metadata_path, index=False, encoding="utf-8-sig")

    return index_path, metadata_path
