from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.retrieval.models.retrieval_system import FaissRetrievalSystem


def load_yaml(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an ad-hoc text query against the FAISS retrieval index.")
    parser.add_argument("--config", default="configs/indexing.yaml", help="Path to retrieval config YAML.")
    parser.add_argument("--query", required=True, help="Text query to search for.")
    parser.add_argument("--top_k", type=int, default=10, help="Number of results to return.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = Path(__file__).resolve().parents[2] / cfg_path

    cfg = load_yaml(cfg_path)
    index_dir = Path(cfg["index"]["output_dir"])

    system = FaissRetrievalSystem(
        index_path=index_dir / cfg["index"]["index_name"],
        metadata_path=index_dir / cfg["index"]["metadata_name"],
        model_name=cfg["model"]["name"],
        pretrained=cfg["model"]["pretrained"],
        device=cfg["model"].get("device", "auto"),
        precision=cfg["model"].get("precision", "fp32"),
        normalize=cfg["model"].get("normalize", True),
    )

    results = system.search(args.query, top_k=args.top_k)

    print(results[[
        "rank",
        "score",
        "dataset",
        "video_id",
        "keyframe_id",
        "keyframe_path",
    ]])


if __name__ == "__main__":
    main()
