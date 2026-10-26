from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from src.asr_extraction.pipeline.extract_asr import ExtractASRPipeline


BACKEND_DIR = Path(__file__).resolve().parents[3]


def backend_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else BACKEND_DIR / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract per-video ASR JSON for V3C or LHE")
    parser.add_argument("--dataset", choices=("v3c", "lhe"), required=True)
    parser.add_argument("--video-id", action="append", dest="video_ids", help="Process one video ID; repeat for more")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing ASR JSON files")
    parser.add_argument("--input-dir", type=Path, help="Override the dataset video directory")
    parser.add_argument("--output-dir", type=Path, help="Override the dataset ASR JSON directory")
    parser.add_argument("--config", default="configs/asr_extraction.yaml")
    args = parser.parse_args()

    with backend_path(args.config).open("r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file) or {}
    with (BACKEND_DIR / "configs/app.yaml").open("r", encoding="utf-8") as file:
        app_cfg = yaml.safe_load(file) or {}

    dataset = app_cfg["datasets"]["available"][args.dataset]
    group = args.dataset.upper()
    input_dir = args.input_dir or backend_path(dataset["videos_root"]) / group
    output_dir = args.output_dir or backend_path(dataset["root"]) / "asr" / group

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger = logging.getLogger("asr_extraction")
    pipeline = ExtractASRPipeline(cfg, input_dir, output_dir, logger)
    videos = pipeline.video_files(args.video_ids)
    if not videos:
        raise FileNotFoundError(f"No videos found in {input_dir}")

    for video in videos:
        try:
            pipeline.process_video(video, overwrite=args.overwrite)
        except Exception:
            logger.exception("ASR failed for %s", video)
            raise


if __name__ == "__main__":
    main()
