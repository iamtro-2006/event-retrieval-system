from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from src.ocr_extraction.pipeline.extract_ocr import ExtractOCRPipeline
from src.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the OCR extraction pipeline (Qwen3-VL by default, PaddleOCR+VietOCR fallback, per config)."
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
        name="ocr_extraction",
        level="INFO",
        log_dir=Path(__file__).resolve().parents[2] / "logs",
        log_to_file=True,
        filename="extract_ocr.log",
    )

    pipeline = ExtractOCRPipeline(cfg=cfg, logger=logger)
    pipeline.run()


if __name__ == "__main__":
    main()
