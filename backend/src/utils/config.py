from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml


@dataclass
class ProjectConfig:
    name: str = "keyframe_pipeline"
    seed: int = 42


@dataclass
class PathConfig:
    input_dir: Path
    output_dir: Path


@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_dir: Path = Path("logs")
    log_to_file: bool = True
    filename: str = "extract_keyframes.log"


@dataclass
class TransNetConfig:
    repo_dir: Path
    weights_path: Path
    threshold: float = 0.5
    batch_size: int = 100
    device: str = "auto"
    skip_existing: bool = True


@dataclass
class FrameLoaderConfig:
    backend: str = "decord"
    image_size: int | None = None
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = False


@dataclass
class EmbeddingConfig:
    model_name: str = "MobileCLIP2-S4"
    pretrained: str = "dfndr2b"
    batch_size: int = 256
    device: str = "auto"
    precision: str = "fp16"
    skip_existing: bool = True


@dataclass
class KeyframeConfig:
    min_scene_frames: int = 3
    max_scene_gap_frames: int = 5000
    hist_threshold: float = 0.90
    min_hist_bins: int = 10
    image_quality: int = 95
    save_images: bool = True
    skip_existing: bool = True


@dataclass
class AppConfig:
    project: ProjectConfig
    paths: PathConfig
    logging: LoggingConfig
    transnet: TransNetConfig
    frame_loader: FrameLoaderConfig
    embedding: EmbeddingConfig
    keyframe: KeyframeConfig


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def load_config(config_path: str | Path) -> AppConfig:
    config_path = Path(config_path).resolve()
    root = config_path.parent.parent
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    project = raw.get("project", {})
    paths = raw.get("paths", {})
    logging = raw.get("logging", {})
    transnet = raw.get("transnet", {})
    frame_loader = raw.get("frame_loader", {})
    embedding = raw.get("embedding", {})
    keyframe = raw.get("keyframe", {})

    return AppConfig(
        project=ProjectConfig(
            name=str(project.get("name", "keyframe_pipeline")),
            seed=int(project.get("seed", 42)),
        ),
        paths=PathConfig(
            input_dir=_resolve(root, paths["input_dir"]),
            output_dir=_resolve(root, paths["output_dir"]),
        ),
        logging=LoggingConfig(
            level=str(logging.get("level", "INFO")),
            log_dir=_resolve(root, logging.get("log_dir", "logs")),
            log_to_file=_as_bool(logging.get("log_to_file", True), True),
            filename=str(logging.get("filename", "extract_keyframes.log")),
        ),
        transnet=TransNetConfig(
            repo_dir=_resolve(root, transnet["repo_dir"]),
            weights_path=_resolve(root, transnet["weights_path"]),
            threshold=float(transnet.get("threshold", 0.5)),
            batch_size=int(transnet.get("batch_size", 100)),
            device=str(transnet.get("device", "auto")),
            skip_existing=_as_bool(transnet.get("skip_existing", True), True),
        ),
        frame_loader=FrameLoaderConfig(
            backend=str(frame_loader.get("backend", "decord")),
            image_size=frame_loader.get("image_size", None),
            num_workers=int(frame_loader.get("num_workers", 0)),
            pin_memory=_as_bool(frame_loader.get("pin_memory", True), True),
            persistent_workers=_as_bool(frame_loader.get("persistent_workers", False), False),
        ),
        embedding=EmbeddingConfig(
            model_name=str(embedding.get("model_name", "MobileCLIP2-S4")),
            pretrained=str(embedding.get("pretrained", "dfndr2b")),
            batch_size=int(embedding.get("batch_size", 256)),
            device=str(embedding.get("device", "auto")),
            precision=str(embedding.get("precision", "fp16")),
            skip_existing=_as_bool(embedding.get("skip_existing", True), True),
        ),
        keyframe=KeyframeConfig(
            min_scene_frames=int(keyframe.get("min_scene_frames", 3)),
            max_scene_gap_frames=int(keyframe.get("max_scene_gap_frames", 5000)),
            hist_threshold=float(keyframe.get("hist_threshold", 0.90)),
            min_hist_bins=int(keyframe.get("min_hist_bins", 10)),
            image_quality=int(keyframe.get("image_quality", 95)),
            save_images=_as_bool(keyframe.get("save_images", True), True),
            skip_existing=_as_bool(keyframe.get("skip_existing", True), True),
        ),
    )
