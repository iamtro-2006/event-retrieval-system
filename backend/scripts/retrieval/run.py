from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from src.retrieval.pipelines.build_faiss import BuildFaissIndexPipeline
from src.retrieval.pipelines.build_vector_cache import build_vector_cache


def load_yaml(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval-related pipelines.")
    parser.add_argument("--config", default="configs/indexing.yaml", help="Path to retrieval config YAML.")
    parser.add_argument("--task", choices=["build-index", "build-vector-cache"], default="build-index")
    parser.add_argument("--output", default=None, help="Optional override output path for vector cache.")
    parser.add_argument("--dtype", default=None, help="Optional override dtype for vector cache (float16 or float32).")
    parser.add_argument("--batch-size", type=int, default=50_000, help="Batch size for vector cache creation.")
    parser.add_argument("--no-normalize", action="store_true", help="Do not normalize reconstructed vectors when building cache.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = Path(__file__).resolve().parents[2] / cfg_path

    cfg = load_yaml(cfg_path)

    if args.task == "build-index":
        pipeline = BuildFaissIndexPipeline(cfg)
        pipeline.run()
        return

    faiss_cfg = cfg.get("faiss", {})
    model_cfg = cfg.get("model", {})
    backend_dir = Path(__file__).resolve().parents[2]
    index_path = backend_dir / Path(faiss_cfg["index_path"]).resolve() if not Path(faiss_cfg["index_path"]).is_absolute() else Path(faiss_cfg["index_path"])
    output_path = (
        Path(args.output)
        if args.output
        else backend_dir / Path(faiss_cfg.get("vector_cache_path", "data/database/faiss_hnsw_clip_vitl16_siglip_256/vectors_fp16.npy"))
    )
    dtype_name = args.dtype or faiss_cfg.get("vector_cache_dtype", "float16")
    dtype = build_vector_cache.__globals__["normalize_dtype"](dtype_name)  # reuse helper from imported module
    normalize = bool(model_cfg.get("normalize", True)) and not args.no_normalize

    build_vector_cache(
        index_path=index_path,
        output_path=output_path,
        dtype=dtype,
        batch_size=max(1, int(args.batch_size)),
        normalize=normalize,
    )


if __name__ == "__main__":
    main()
