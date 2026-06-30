from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from src.embeddings.pipelines.extract_embeddings import ExtractEmbeddingPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the embeddings extraction pipeline.")
    parser.add_argument("--config", default="configs/embeddings.yaml", help="Path to embeddings config YAML.")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = Path(__file__).resolve().parents[2] / cfg_path

    cfg = load_config(cfg_path)
    pipeline = ExtractEmbeddingPipeline(cfg=cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
