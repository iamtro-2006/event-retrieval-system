from __future__ import annotations

import logging

from src.index.faiss_index import (
    build_matrix_and_metadata,
    create_faiss_index,
    save_faiss_index,
)


class BuildFaissIndexPipeline:
    def __init__(self, cfg: dict, logger: logging.Logger | None = None):
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)

    def run(self):
        matrix, metadata = build_matrix_and_metadata(
            embeddings_root=self.cfg["embeddings_root"],
            keyframes_root=self.cfg["keyframes_root"],
            map_keyframes_root=self.cfg["map_keyframes_root"],
        )

        self.logger.info("Embedding matrix shape: %s", matrix.shape)
        self.logger.info("Metadata rows: %d", len(metadata))

        index = create_faiss_index(
            matrix=matrix,
            metric=self.cfg["index"].get("metric", "cosine"),
        )

        index_path, metadata_path = save_faiss_index(
            index=index,
            metadata=metadata,
            output_dir=self.cfg["index"]["output_dir"],
            index_name=self.cfg["index"]["index_name"],
            metadata_name=self.cfg["index"]["metadata_name"],
        )

        self.logger.info("Saved FAISS index: %s", index_path)
        self.logger.info("Saved metadata: %s", metadata_path)