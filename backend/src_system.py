from __future__ import annotations

import argparse

from src.config.loader import load_yaml
from src.index.retrieval_system import FaissRetrievalSystem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/indexing.yaml")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    cfg = load_yaml(args.config)

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

    print(
        results[
            [
                "rank",
                "score",
                "dataset",
                "video_id",
                "keyframe_id",
                "keyframe_path",
            ]
        ]
    )


if __name__ == "__main__":
    main()
