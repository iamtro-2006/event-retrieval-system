from __future__ import annotations

import argparse
from pathlib import Path
import sys

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the keyframe extraction pipeline.")
    parser.add_argument(
        "--method",
        choices=("distribution", "kfeavi", "vlm"),
        default="distribution",
        help="Keyframe method: two-distribution anomaly detector, KFEAVI, or PE-Core embeddings.",
    )
    parser.add_argument(
        "--video",
        type=Path,
        help="Input video for distribution/kfeavi; for vlm use --input-dir or the video's parent directory.",
    )
    parser.add_argument("--input-dir", type=Path, help="Input directory for the PE-Core pipeline.")
    parser.add_argument("--output-dir", type=Path, default=BACKEND_ROOT.parent / "data")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--config", default="configs/kf_extraction.yaml", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.video is None and (args.method != 'vlm' or args.input_dir is None):
        raise SystemExit("--video is required, or use --method vlm --input-dir")

    video = args.video.resolve() if args.video else None
    output_dir = args.output_dir.resolve()

    if args.method == "distribution":
        # Keep the statistically strongest pipeline as the single distribution entrypoint.
        from scripts.keyframe_extraction import run_two_distribution_hsv_motion_v2

        forwarded = [
            "--video", str(video), "--output-dir", str(output_dir),
        ]
        if args.overwrite:
            forwarded.append("--overwrite")
        old_argv = sys.argv
        try:
            sys.argv = [old_argv[0], *forwarded]
            run_two_distribution_hsv_motion_v2.main()
        finally:
            sys.argv = old_argv
        return

    if args.method == "kfeavi":
        from src.keyframe_extraction.baselines import extract_kfeavi
        image_dir = output_dir / "keyframes" / video.parent.name / video.stem
        map_path = output_dir / "map_keyframes" / video.parent.name / f"{video.stem}.csv"
        if map_path.exists() and not args.overwrite:
            raise SystemExit(f"Output exists; use --overwrite: {map_path}")
        if args.overwrite and image_dir.exists():
            import shutil
            shutil.rmtree(image_dir)
        result = extract_kfeavi(video, image_dir, map_path)
        print(result)
        return

    # Historical CLI name kept for compatibility: this is the project's
    # embedding-based keyframe pipeline (PE-Core), not a captioning VLM.
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = BACKEND_ROOT / config_path
    cfg = load_config(config_path)
    if args.input_dir is not None:
        cfg.paths.input_dir = args.input_dir.resolve()
    elif args.video is not None:
        cfg.paths.input_dir = video.parent
    cfg.paths.output_dir = output_dir
    if args.overwrite:
        cfg.omni.skip_existing = False
        cfg.embedding.skip_existing = False
        cfg.keyframe.skip_existing = False

    from src.keyframe_extraction.pipeline.extract_keyframes import KeyframeExtractionPipeline

    KeyframeExtractionPipeline(cfg).run(videos=[video] if video else None)


if __name__ == "__main__":
    main()
