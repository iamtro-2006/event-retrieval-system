from __future__ import annotations

import argparse
import faulthandler
import logging
import os
import sys
from pathlib import Path

import yaml

from src.caption_extraction.pipeline.extract_caption import ExtractCaptionPipeline
from src.utils.pipeline_progress import verbose_logging


BACKEND_DIR = Path(__file__).resolve().parents[3]


def backend_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else BACKEND_DIR / path


def main() -> None:
    faulthandler.enable(all_threads=True)
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO" if verbose_logging() else "WARNING")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0" if verbose_logging() else "1")
    parser = argparse.ArgumentParser(description="Extract shot TXT and caption JSON from videos")
    parser.add_argument("--dataset", choices=("v3c", "lhe"), required=True)
    parser.add_argument("--video-id", action="append", dest="video_ids")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--scenes-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--config", default="configs/caption_extraction.yaml")
    args = parser.parse_args()

    with backend_path(args.config).open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file) or {}
    with (BACKEND_DIR / "configs/app.yaml").open(encoding="utf-8") as file:
        app_cfg = yaml.safe_load(file) or {}
    dataset = app_cfg["datasets"]["available"][args.dataset]
    group = args.dataset.upper()
    input_dir = args.input_dir or backend_path(dataset["videos_root"]) / group
    scenes_dir = args.scenes_dir or backend_path(dataset["root"]) / "scenes" / group
    output_dir = args.output_dir or backend_path(dataset["root"]) / "captions" / group

    level = logging.INFO if verbose_logging() else logging.WARNING
    logging.basicConfig(level=level, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(message)s", force=True)
    # A private handler remains active if a native dependency reconfigures root logging.
    logger = logging.getLogger("caption_pipeline")
    logger.setLevel(level)
    logger.propagate = False
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.handlers[:] = [handler]
    pipeline = ExtractCaptionPipeline(cfg, input_dir, scenes_dir, output_dir, logger=logger)
    videos = pipeline.video_files(args.video_ids)
    if not videos:
        raise FileNotFoundError(f"No videos found in {input_dir}")
    pipeline.prepare_shots(videos, overwrite=args.overwrite)
    pipeline.caption_videos(videos, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
