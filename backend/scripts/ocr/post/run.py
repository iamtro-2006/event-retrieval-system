from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from src.ocr_extraction.pipeline.post_process import PostOCRPipeline
from src.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the standalone post-OCR pipeline (dedupe + token/math "
                     "normalize + repetition_guard + post_correction) on already "
                     "-extracted OCR JSON. Separate from scripts.ocr.extract.run "
                     "-- xem post_process.py docstring."
    )
    parser.add_argument("--config", default="configs/ocr_extraction.yaml", help="Path to ocr config YAML.")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)

    cfg = load_config(cfg_path)
    logger = setup_logger(
        name="ocr_post_process",
        level="INFO",
        log_dir=Path(__file__).resolve().parents[2] / "logs",
        log_to_file=True,
        filename="post_process.log",
    )

    pipeline = PostOCRPipeline(cfg=cfg, logger=logger)
    pipeline.run()


if __name__ == "__main__":
    main()
