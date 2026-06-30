"""backend.app.config — load app.yaml and resolve all filesystem paths.

All path-resolution and config-loading logic is centralised here so that
``main.py`` (and every route module) can simply do::

    from app.config import CFG, FAISS_INDEX_PATH, KEYFRAMES_ROOT, …

Nothing in this module imports FastAPI — it is safe to import at startup
before the application object is created.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

load_dotenv()

HF_TOKEN: str | None = os.getenv("HF_TOKEN")
if HF_TOKEN:
    print("[ENV] HF_TOKEN loaded")

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------

BACKEND_DIR: Path = Path(__file__).resolve().parent.parent
CONFIG_PATH: Path = BACKEND_DIR / "configs" / "app.yaml"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file and return its contents as a dict."""
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _normalize_path_text(value: str | Path) -> str:
    """Convert backslashes to forward slashes."""
    return str(value or "").replace("\\", "/")


def resolve_backend_path(value: str | Path) -> Path:
    """Resolve *value* relative to BACKEND_DIR if it is not absolute."""
    path = Path(_normalize_path_text(value))
    return path if path.is_absolute() else BACKEND_DIR / path


# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------

CFG: dict[str, Any] = load_yaml(CONFIG_PATH)

# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------

SHOULD_PROFILE: bool = bool(CFG.get("debug", {}).get("profile", False))

# ---------------------------------------------------------------------------
# FAISS / vector cache
# ---------------------------------------------------------------------------

FAISS_INDEX_PATH: Path = resolve_backend_path(CFG["faiss"]["index_path"])
METADATA_PATH: Path = resolve_backend_path(CFG["faiss"]["metadata_path"])
VECTOR_CACHE_PATH: Path = resolve_backend_path(
    CFG.get("faiss", {}).get(
        "vector_cache_path",
        "data/database/faiss_hnsw_clip_vitl16_siglip_256/vectors_fp16.npy",
    )
)

# ---------------------------------------------------------------------------
# Filesystem roots for static files
# ---------------------------------------------------------------------------

KEYFRAMES_ROOT: Path = resolve_backend_path(CFG["paths"]["keyframes_root"])
VIDEOS_ROOT: Path = resolve_backend_path(CFG["paths"]["videos_root"])
MAP_KEYFRAME_ROOT: Path = resolve_backend_path(CFG["paths"]["map_keyframe_path"])
