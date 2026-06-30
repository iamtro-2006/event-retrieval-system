from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.config import load_config
from src.keyframes.pipelines.extract_keyframes import KeyframeExtractionPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the keyframe extraction pipeline.")
    parser.add_argument("--config", default="configs/kf_extraction.yaml", help="Path to keyframe extraction config YAML.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = Path(__file__).resolve().parents[2] / cfg_path

    cfg = load_config(cfg_path)
    pipeline = KeyframeExtractionPipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
