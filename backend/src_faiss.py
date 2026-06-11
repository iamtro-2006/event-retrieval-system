from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from src.pipelines.build_faiss import BuildFaissIndexPipeline


def load_yaml(path: str | Path) -> dict:
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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
    pipeline = BuildFaissIndexPipeline(cfg, logger=logger)
    pipeline.run()


if __name__ == "__main__":
    main()