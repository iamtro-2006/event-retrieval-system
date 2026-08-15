"""Config-loading và path-resolution helpers — port nguyên bản từ `main.py`
cũ (không đổi hành vi). Dùng chung cho toàn bộ package `legacy/` và các
router legacy (`health`, `dres`, `speech`, `legacy_search`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    """Load and parse a YAML configuration file.

    Args:
        path: The path to the YAML file.

    Returns:
        A dictionary containing the parsed configuration.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_path_text(path_value: str | Path) -> str:
    """Normalize path separators to forward slashes."""
    return str(path_value or "").replace("\\", "/")


def resolve_backend_path(backend_dir: Path, path_value: str | Path) -> Path:
    """Resolve a relative or absolute path against the backend directory.

    Args:
        backend_dir: The backend root directory (`main.py`'s parent).
        path_value: The raw path string or Path object from the config.

    Returns:
        The absolute Path object.
    """
    path = Path(normalize_path_text(path_value))
    return path if path.is_absolute() else backend_dir / path


class LegacyPaths:
    """Bundle các path tĩnh mà nhiều router legacy cùng cần (keyframes/videos
    /map-keyframes root, dùng để mount static file + build `image_url` v.v.
    trong `serializers.py`). Được tính 1 lần lúc lifespan, lưu vào
    `app.state.legacy_paths` — KHÔNG tính lại mỗi request.
    """

    def __init__(self, backend_dir: Path, config_path: Path, cfg: dict[str, Any]) -> None:
        self.backend_dir = backend_dir
        self.config_path = config_path
        self.keyframes_root = resolve_backend_path(backend_dir, cfg["paths"]["keyframes_root"])
        self.videos_root = resolve_backend_path(backend_dir, cfg["paths"]["videos_root"])
        self.map_keyframe_root = resolve_backend_path(backend_dir, cfg["paths"]["map_keyframe_path"])
