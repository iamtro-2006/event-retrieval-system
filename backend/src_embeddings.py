from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from src.pipelines.extract_embeddings import ExtractEmbeddingPipeline


def load_config(path: str | Path) -> dict:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
    )
    return logging.getLogger("extract_embeddings")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/embedding.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger()

    cfg = load_config(args.config)

    pipeline = ExtractEmbeddingPipeline(
        cfg=cfg,
        logger=logger,
    )

    pipeline.run()


if __name__ == "__main__":
    main()