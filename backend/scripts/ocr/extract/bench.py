from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml

from src.ocr_extraction.pipeline.extract_ocr import ExtractOCRPipeline
from src.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark OCR extraction speed (sec/frame, frames/sec) for ONE video's "
            "keyframes, writing output under extraction.output_root (point this at "
            "ocr_test/ via --config so it never touches the real ocr/ folder)."
        )
    )
    parser.add_argument("--config", default="configs/ocr_extraction.test.yaml", help="Path to ocr config YAML.")
    parser.add_argument("--dataset", default="L21", help="Dataset folder name, e.g. L21.")
    parser.add_argument("--video", default="L21_V001", help="Video folder name, e.g. L21_V001.")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))

    logger = setup_logger(
        name="ocr_bench",
        level="INFO",
        log_dir=Path(__file__).resolve().parents[3] / "logs",
        log_to_file=True,
        filename="extract_ocr_bench.log",
    )

    # Model load (incl. first-time GGUF download for engine="qwen") happens here,
    # BEFORE timing starts, so the benchmark below measures pure inference speed.
    pipeline = ExtractOCRPipeline(cfg=cfg, logger=logger)

    video_dir = pipeline.input_root / args.dataset / args.video
    if not video_dir.exists():
        raise FileNotFoundError(f"Video keyframes dir not found: {video_dir}")

    n_input_frames = sum(1 for p in video_dir.iterdir() if p.is_file())
    logger.info(
        "Benchmarking %s (%d keyframe files) with engine=%s", video_dir, n_input_frames, pipeline.engine
    )

    t0 = time.perf_counter()
    n_saved = pipeline.process_video_dir(video_dir)
    elapsed = time.perf_counter() - t0

    sec_per_frame = elapsed / n_saved if n_saved else float("nan")
    fps = n_saved / elapsed if elapsed > 0 else float("nan")

    print("")
    print(f"engine        : {pipeline.engine}")
    print(f"video         : {args.dataset}/{args.video}")
    print(f"frames        : {n_saved} / {n_input_frames} found")
    print(f"total time    : {elapsed:.2f}s")
    print(f"sec / frame   : {sec_per_frame:.3f}s")
    print(f"frames / sec  : {fps:.3f}")
    print(f"output file   : {pipeline.output_json_for(video_dir)}")

    if n_saved == 0:
        print(
            "\nNote: 0 frames processed -- either the folder was empty, or "
            "output_file already existed AND skip_existing=true in your config "
            "(the .test.yaml sets skip_existing=false by default)."
        )


if __name__ == "__main__":
    main()
