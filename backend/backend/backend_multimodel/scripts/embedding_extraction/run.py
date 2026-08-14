from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from src.embedding_extraction.models.registry import MODEL_PRESETS
from src.embedding_extraction.pipeline.extract_embeddings import run_multi_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the embeddings extraction pipeline.")
    parser.add_argument("--config", default="configs/embeddings.yaml", help="Path to embeddings config YAML.")
    parser.add_argument(
        "--model", "-m",
        action="append",
        default=None,
        help=(
            "Model 'key' (or 'name') from the config to run, e.g. "
            "--model siglip2_so400m_384. Repeat the flag for several models "
            "(--model siglip2_so400m_384 --model blip2_itm_vitg), or pass a "
            "comma-separated list (--model siglip2_so400m_384,blip2_itm_vitg). "
            "If omitted, runs every model in the config with enabled: true."
        ),
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print every model 'key' available in --config (and the built-in presets) and exit.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _expand_model_arg(values: list[str] | None) -> list[str] | None:
    """--model can be repeated and/or comma-separated - normalize to a flat list."""
    if not values:
        return None
    flat: list[str] = []
    for v in values:
        flat.extend(part.strip() for part in v.split(",") if part.strip())
    return flat


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = Path(__file__).resolve().parents[2] / cfg_path

    cfg = load_config(cfg_path)

    if args.list_models:
        entries = cfg.get("models") or ([cfg["model"]] if "model" in cfg else [])
        print("Models in config:")
        for m in entries:
            key = m.get("key") or m.get("name")
            status = "enabled" if m.get("enabled", True) else "disabled"
            print(f"  - {key}  ({m['name']}, {status})")
        print("\nBuilt-in presets (usable via 'name:' even if not listed above):")
        for name in MODEL_PRESETS:
            print(f"  - {name}")
        return

    models = _expand_model_arg(args.model)
    run_multi_model(cfg, models=models)


if __name__ == "__main__":
    main()
