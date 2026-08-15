from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.retrieval.index.factory import build_faiss_index, build_index_manager_from_config
from src.retrieval.retriever.semantic_search.pipeline.search import SearchPipeline

PATH_KEYS = {"index_path", "metadata_path", "vector_cache_path", "repo_path", "checkpoint_path", "spm_path"}


def load_yaml(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_backend_paths(value, backend_dir: Path):
    if isinstance(value, list):
        return [resolve_backend_paths(item, backend_dir) for item in value]
    if not isinstance(value, dict):
        return value

    out = {}
    for key, item in value.items():
        if isinstance(item, (dict, list)):
            out[key] = resolve_backend_paths(item, backend_dir)
        elif key in PATH_KEYS and item not in (None, ""):
            path = Path(str(item).replace("\\", "/"))
            out[key] = str(path if path.is_absolute() else backend_dir / path)
        else:
            out[key] = item
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an ad-hoc text query against the FAISS retrieval index.")
    parser.add_argument("--config", default="configs/app.yaml", help="Path to app/search config YAML.")
    parser.add_argument("--query", required=True, help="Text query to search for.")
    parser.add_argument("--top_k", type=int, default=10, help="Number of results to return.")
    parser.add_argument(
        "--model-key",
        default=None,
        help="Which model to query when configs/indexing.yaml declares several models "
        "under `models:` (see MODEL_PRESETS / README). Defaults to the config's "
        "`default_model_key`, or the only model if there's just one.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print the model keys available in --config and exit (no search performed).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = Path(__file__).resolve().parents[2] / cfg_path

    backend_dir = Path(__file__).resolve().parents[2]
    cfg = load_yaml(cfg_path)

    semantic_cfg = cfg.get("semantic", cfg)

    # Supported shapes:
    #   - app.yaml runtime shape: top-level `semantic: {models: [...]}`
    #   - direct multi-model shape: top-level `models: [...]`
    #   - legacy indexing shape: top-level `index:`/`model:` keys
    if isinstance(semantic_cfg, dict) and "models" in semantic_cfg:
        index = build_index_manager_from_config(resolve_backend_paths(semantic_cfg, backend_dir))
        if args.list_models:
            print("Available model keys:", ", ".join(index.keys()))
            return
        faiss_index = index.get(args.model_key)
    else:
        index_dir = Path(cfg["index"]["output_dir"])
        if not index_dir.is_absolute():
            index_dir = backend_dir / index_dir
        faiss_index = build_faiss_index(
            dict(
                index_path=index_dir / cfg["index"]["index_name"],
                metadata_path=index_dir / cfg["index"]["metadata_name"],
                model_name=cfg["model"]["name"],
                pretrained=cfg["model"].get("pretrained"),
                device=cfg["model"].get("device", "auto"),
                precision=cfg["model"].get("precision", "fp32"),
                normalize=cfg["model"].get("normalize", True),
            )
        )
        if args.list_models:
            print("Available model keys:", faiss_index.model_key)
            return

    pipeline = SearchPipeline(faiss_index)
    results = pipeline.multi_query_search([args.query], top_k=args.top_k)

    print(f"[model={faiss_index.model_key}]")
    print(results[[
        "rank",
        "retrieval_score",
        "dataset",
        "video_id",
        "keyframe_id",
        "keyframe_path",
    ]])


if __name__ == "__main__":
    main()
