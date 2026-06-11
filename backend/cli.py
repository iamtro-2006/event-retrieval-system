from __future__ import annotations

import os
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "quiet"

import argparse
from pathlib import Path

from src.utils.config import load_config
from src.pipelines.extract_keyframes import KeyframeExtractionPipeline
from src.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Extract keyframes using TransNetV2 + embedding-based clustering.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--input-dir", default=None, help="Override input video directory.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    parser.add_argument("--log-level", default=None, help="Override log level, e.g. DEBUG, INFO, WARNING.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.input_dir:
        cfg.paths.input_dir = Path(args.input_dir).resolve()
    if args.output_dir:
        cfg.paths.output_dir = Path(args.output_dir).resolve()
    if args.log_level:
        cfg.logging.level = args.log_level

    logger = setup_logger(
        cfg.project.name,
        cfg.logging.level,
        cfg.logging.log_dir,
        cfg.logging.log_to_file,
        cfg.logging.filename,
    )
    KeyframeExtractionPipeline(cfg, logger).run()


if __name__ == "__main__":
    main()
