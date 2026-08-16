from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

from src.retrieval.indexer.embedding_loader import (
    build_matrix_and_metadata,
    normalize_matrix,
)


def _normalize_cache_dtype(dtype_name: str) -> np.dtype:
    dtype_name = str(dtype_name or "float32").strip().lower()
    if dtype_name in {"fp16", "float16", "f16", "half"}:
        return np.dtype(np.float16)
    if dtype_name in {"fp32", "float32", "f32"}:
        return np.dtype(np.float32)
    raise ValueError(f"Unsupported vector_cache_dtype: {dtype_name}")


class BuildMilvusIndexPipeline:
    """Build a Milvus collection + metadata.csv + vector cache .npy from embeddings.

    The collection holds ONLY the ANN data (`row_id` PK + `embedding` vector);
    filtering/metadata resolution stays in the pandas metadata (3-pillar design).
    `row_id` is the positional index inserted in metadata/memmap order, so the
    runtime `MilvusSearchAdapter` maps search hits straight back to
    metadata/memmap rows without a secondary id lookup.
    """

    def __init__(self, cfg: dict, logger: logging.Logger | None = None):
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)

    def run(self) -> None:
        from pymilvus import (
            Collection,
            CollectionSchema,
            DataType,
            FieldSchema,
            connections,
            utility,
        )

        matrix, metadata = build_matrix_and_metadata(
            embeddings_root=self.cfg["embeddings_root"],
            keyframes_root=self.cfg["keyframes_root"],
            map_keyframes_root=self.cfg["map_keyframes_root"],
        )
        self.logger.info("Embedding matrix shape: %s", matrix.shape)
        self.logger.info("Metadata rows: %d", len(metadata))

        milvus_cfg = self.cfg["milvus"]
        metric = str(milvus_cfg.get("metric", "cosine")).lower()
        if metric in {"cosine", "ip"}:
            matrix = normalize_matrix(matrix)
            metric_type = "IP"
        elif metric == "l2":
            metric_type = "L2"
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        n, dim = matrix.shape

        host = str(milvus_cfg.get("host", "localhost"))
        port = str(milvus_cfg.get("port", 19530))
        collection_name = str(milvus_cfg.get("collection_name", "keyframes"))

        connections.connect("default", host=host, port=port)
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)

        fields = [
            FieldSchema(
                name="row_id", dtype=DataType.INT64, is_primary=True, auto_id=False
            ),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        ]
        schema = CollectionSchema(fields, description="keyframe embeddings")
        collection = Collection(collection_name, schema=schema)

        index_params = {
            "index_type": str(milvus_cfg.get("index_type", "HNSW")),
            "metric_type": metric_type,
            "params": {
                "M": int(milvus_cfg.get("hnsw_m", 32)),
                "efConstruction": int(milvus_cfg.get("ef_construction", 200)),
            },
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        collection.load()

        batch_size = max(1, int(milvus_cfg.get("insert_batch_size", 5000)))
        row_ids = np.arange(n, dtype=np.int64)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            collection.insert([row_ids[start:end].tolist(), matrix[start:end].tolist()])
            self.logger.info("Inserted %d/%d (%.1f%%)", end, n, end / max(1, n) * 100)
        collection.flush()
        self.logger.info("Milvus collection ready: %s", collection_name)

        # Write metadata.csv + vector cache .npy in the SAME row order as insert,
        # so row_id == metadata row == memmap row.
        index_cfg = self.cfg["index"]
        output_dir = Path(index_cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        metadata_path = output_dir / index_cfg.get("metadata_name", "metadata.csv")
        metadata.to_csv(metadata_path, index=False, encoding="utf-8-sig")
        self.logger.info("Saved metadata: %s", metadata_path)

        cache_dtype = _normalize_cache_dtype(
            milvus_cfg.get("vector_cache_dtype", "float32")
        )
        vector_cache_path = output_dir / index_cfg.get(
            "vector_cache_name", "vectors_fp32.npy"
        )
        vectors = open_memmap(
            str(vector_cache_path), mode="w+", dtype=cache_dtype, shape=(n, dim)
        )
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            vectors[start:end] = matrix[start:end].astype(cache_dtype, copy=False)
        vectors.flush()
        del vectors
        self.logger.info("Saved vector cache: %s", vector_cache_path)
