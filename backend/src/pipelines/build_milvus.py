from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from pymilvus import MilvusClient

from src.index.embedding_index import build_matrix_and_metadata
from src.index.milvus_schema import (
    create_collection,
    detect_dimension,
    PK_FIELD,
    SCALAR_FIELDS,
    VECTOR_FIELD,
)


def make_pk(row: dict) -> str:
    dataset = str(row.get("dataset", "") or "")
    video_id = str(row.get("video_id", "") or "")
    keyframe_id_int = int(row.get("keyframe_id_int", 0) or 0)
    return f"{dataset}/{video_id}/{keyframe_id_int:06d}"


def row_to_record(row: dict) -> dict:
    record = {PK_FIELD: make_pk(row), VECTOR_FIELD: None}
    for name, _, _ in SCALAR_FIELDS:
        val = row.get(name)
        if name in ("keyframe_id_int", "frame_idx") and val is not None:
            try:
                if val == val:
                    record[name] = int(float(val))
                else:
                    record[name] = 0
            except (TypeError, ValueError):
                record[name] = 0
        elif name == "timestamp_sec" and val is not None:
            try:
                if val == val:
                    record[name] = float(val)
                else:
                    record[name] = 0.0
            except (TypeError, ValueError):
                record[name] = 0.0
        elif name == "fps" and val is not None:
            try:
                if val == val:
                    record[name] = float(val)
                else:
                    record[name] = 0.0
            except (TypeError, ValueError):
                record[name] = 0.0
        elif isinstance(val, float) and val != val:
            record[name] = ""
        else:
            record[name] = str(val) if val is not None else ""
    return record


class BuildMilvusPipeline:
    def __init__(
        self,
        cfg: dict,
        milvus_host: str = "localhost",
        milvus_port: int = 19530,
        logger: logging.Logger | None = None,
    ):
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)
        self.collection_name = cfg["collection_name"]
        self.client = MilvusClient(uri=f"http://{milvus_host}:{milvus_port}")

    def run(self):
        embeddings_root = Path(self.cfg["embeddings_root"])
        keyframes_root = Path(self.cfg["keyframes_root"])
        map_keyframes_root = Path(self.cfg["map_keyframes_root"])

        dim = detect_dimension(embeddings_root)
        self.logger.info("Detected embedding dimension: %d", dim)

        matrix, metadata = build_matrix_and_metadata(
            embeddings_root=embeddings_root,
            keyframes_root=keyframes_root,
            map_keyframes_root=map_keyframes_root,
        )
        self.logger.info("Embedding matrix shape: %s", matrix.shape)
        self.logger.info("Metadata rows: %d", len(metadata))

        create_collection(
            client=self.client,
            collection_name=self.collection_name,
            dim=dim,
            cfg=self.cfg,
        )
        self.logger.info("Collection '%s' created and loaded", self.collection_name)

        records = metadata.to_dict(orient="records")
        batch_size = 1000
        total_inserted = 0

        for start in range(0, len(records), batch_size):
            batch_records = records[start : start + batch_size]
            batch_vectors = matrix[start : start + batch_size].astype(
                np.float32, copy=False
            )

            data = []
            for rec, vec in zip(batch_records, batch_vectors):
                record = row_to_record(rec)
                record[VECTOR_FIELD] = vec.tolist()
                data.append(record)

            self.client.upsert(
                collection_name=self.collection_name,
                data=data,
            )
            total_inserted += len(data)
            self.logger.info(
                "Upserted %d/%d (%.1f%%)",
                total_inserted,
                len(records),
                total_inserted / max(1, len(records)) * 100,
            )

        self.client.flush(self.collection_name)

        stats = self.client.get_collection_stats(self.collection_name)
        self.logger.info("Collection stats: %s", stats)
        self.logger.info("Total upserted: %d", total_inserted)

    def close(self):
        self.client.close()
