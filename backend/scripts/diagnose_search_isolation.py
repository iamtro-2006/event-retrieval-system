"""Compare deterministic search output for one or more loaded model keys.

Run this script in separate processes for single-model and multi-model
configurations.  The final stdout line is machine-readable JSON; startup logs
before it may be ignored.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from src.retrieval.system import build_system


QUERIES = (
    "a red car on the road",
    "a person standing on a stage",
    "a football match in a stadium",
)
TEMPORAL_QUERIES = (
    "a person walks into a room; the person sits down",
    "a football player runs; the player kicks the ball",
)


def _identity_rows(frame) -> list[str]:
    if frame is None or frame.empty:
        return []
    rows = []
    for row in frame.to_dict(orient="records"):
        rows.append(
            "/".join(
                (
                    str(row.get("dataset", "")),
                    str(row.get("video_id", "")),
                    str(row.get("keyframe_id_int", row.get("keyframe_id", ""))),
                )
            )
        )
    return rows


def _embedding_digest(index, query: str) -> str:
    embedding = np.ascontiguousarray(index.encode_texts([query]), dtype=np.float32)
    return hashlib.sha256(embedding.tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/app.yaml")
    parser.add_argument("--models", required=True, help="Comma-separated model_key list to load")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--search-mode", choices=("semantic", "temporal", "auto"), default="semantic")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    requested = [key.strip() for key in args.models.split(",") if key.strip()]
    available = {entry["model_key"]: entry for entry in cfg["semantic"]["models"]}
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise KeyError(f"Unknown model keys: {unknown}")

    cfg["semantic"] = {
        "default_model_key": requested[0],
        "models": [{**available[key], "enabled": True} for key in requested],
    }
    # These subsystems are irrelevant to model isolation and can introduce
    # external service variability during the diagnostic.
    cfg.pop("ocr", None)
    cfg.pop("asr", None)
    cfg["translate"] = {"enabled_default": False}
    cfg["query_enrichment"] = {"enabled": False}

    system = build_system(cfg)
    observations: dict[str, list[dict]] = {key: [] for key in requested}
    queries = QUERIES if args.search_mode == "semantic" else TEMPORAL_QUERIES
    work = [
        (iteration, key, query)
        for iteration in range(max(1, args.iterations))
        for key in requested
        for query in queries
    ]

    def _run(item):
        iteration, key, query = item
        index = system.orchestrator.semantic_search._resolve(key)
        search = getattr(system, f"search_{args.search_mode}")
        result, _ = search(
            query,
            top_k=20,
            candidate_multiplier=4,
            model_key=key,
            translate=False,
        )
        return key, {
            "iteration": iteration,
            "query": query,
            "embedding_sha256": _embedding_digest(index, query),
            "top_ids": _identity_rows(result),
        }

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            completed = list(executor.map(_run, work))
    else:
        completed = [_run(item) for item in work]
    for key, observation in completed:
        observations[key].append(observation)

    print(json.dumps({
        "loaded_models": requested,
        "search_mode": args.search_mode,
        "observations": observations,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
