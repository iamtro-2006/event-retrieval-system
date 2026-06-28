from __future__ import annotations

import argparse
import logging

from src.config.loader import load_yaml
from src.pipelines.build_milvus import BuildMilvusPipeline


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
    )
    return logging.getLogger("build_milvus")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/indexing.yaml")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=19530)
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger()

    cfg = load_yaml(args.config)
    pipeline = BuildMilvusPipeline(
        cfg, milvus_host=args.host, milvus_port=args.port, logger=logger
    )
    try:
        pipeline.run()
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
