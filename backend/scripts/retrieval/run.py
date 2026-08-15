from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from src.retrieval.indexer.faiss.build_pipeline import BuildFaissIndexPipeline
from src.retrieval.indexer.faiss.vector_cache import build_vector_cache, normalize_dtype


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

    backend_dir = Path(__file__).resolve().parents[2]

    # `build-index` uses configs/indexing.yaml. `build-vector-cache` may use
    # either that legacy build config or configs/app.yaml's `semantic.models`
    # runtime registry.
    if "semantic" in cfg and isinstance(cfg["semantic"], dict):
        models = [m for m in cfg["semantic"].get("models", []) if m.get("enabled", True)]
        if not models:
            raise ValueError("No enabled semantic models found in app config.")
        selected = models[0]
        faiss_cfg = {
            "index_path": selected["index_path"],
            "vector_cache_path": selected.get("vector_cache_path"),
            "vector_cache_dtype": selected.get("vector_cache_dtype", "float32"),
        }
        model_cfg = {"normalize": selected.get("normalize", True)}
    else:
        faiss_cfg = cfg.get("faiss", {})
        model_cfg = cfg.get("model", {})

    raw_index_path = Path(str(faiss_cfg["index_path"]).replace("\\", "/"))
    index_path = raw_index_path if raw_index_path.is_absolute() else backend_dir / raw_index_path
    output_path = (
        Path(args.output)
        if args.output
        else backend_dir / Path(faiss_cfg.get("vector_cache_path") or "data/database/faiss_hnsw_clip_vitl16_siglip_256/vectors_fp16.npy")
    )
    dtype_name = args.dtype or faiss_cfg.get("vector_cache_dtype", "float16")
    dtype = normalize_dtype(dtype_name)
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
