from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.config.loader import load_yaml
from src.index.embedding_index import build_matrix_and_metadata
from src.index.faiss_index import create_faiss_index, save_faiss_index


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
    )
    return logging.getLogger("build_faiss_index")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/indexing.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger()

    cfg = load_yaml(args.config)

    matrix, metadata = build_matrix_and_metadata(
        embeddings_root=cfg["embeddings_root"],
        keyframes_root=cfg["keyframes_root"],
        map_keyframes_root=cfg["map_keyframes_root"],
    )

    logger.info("Embedding matrix shape: %s", matrix.shape)
    logger.info("Metadata rows: %d", len(metadata))

    index_cfg = cfg["index"]
    index = create_faiss_index(
        matrix=matrix,
        metric=index_cfg.get("metric", "cosine"),
        index_type=index_cfg.get("type", "hnsw"),
        hnsw_m=index_cfg.get("hnsw_m", 32),
        ef_construction=index_cfg.get("ef_construction", 200),
        ef_search=index_cfg.get("ef_search", 64),
    )

    index_path, metadata_path = save_faiss_index(
        index=index,
        metadata=metadata,
        output_dir=cfg["index"]["output_dir"],
        index_name=cfg["index"]["index_name"],
        metadata_name=cfg["index"]["metadata_name"],
    )

    logger.info("Saved FAISS index: %s", index_path)
    logger.info("Saved metadata: %s", metadata_path)


if __name__ == "__main__":
    main()
