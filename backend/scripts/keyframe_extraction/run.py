from __future__ import annotations

import argparse
from pathlib import Path
import sys

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from src.utils.config import load_config
from src.keyframe_extraction.pipeline.extract_keyframes import KeyframeExtractionPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the keyframe extraction pipeline.")
    parser.add_argument("--config", default="configs/kf_extraction.yaml", help="Path to keyframe extraction config YAML.")
    parser.add_argument(
        "--video-ids",
        nargs="+",
        metavar="VIDEO_ID",
        help="Process only the requested video stems (for example L21_V001 L21_V002).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = Path(__file__).resolve().parents[2] / cfg_path

    cfg = load_config(cfg_path)
    pipeline = KeyframeExtractionPipeline(cfg)
    pipeline.run(video_ids=args.video_ids)


if __name__ == "__main__":
    main()
