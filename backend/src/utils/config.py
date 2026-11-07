from __future__ import annotations

from dataclasses import dataclass, field
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
class FrameLoaderConfig:
    backend: str = "nvdec"
    image_size: int | None = 336
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = False


@dataclass
class EmbeddingConfig:
    model_name: str = "PE-Core-L14-336"
    backend: str | None = None
    pretrained: str = "true"
    batch_size: int = 16
    device: str = "auto"
    precision: str = "fp16"
    skip_existing: bool = True
    sample_fps: float | None = 3.0


@dataclass
class KeyframeConfig:
    max_scene_gap_frames: int = 5000
    max_scene_seconds: float | None = 30.0
    final_similarity_threshold: float = 0.96
    min_hist_bins: int = 5
    image_quality: int = 95
    save_images: bool = True
    skip_existing: bool = True


@dataclass
class OmniConfig:
    weights_path: Path | None = None
    cache_dir: Path = Path("weights/omni")
    overlap: int = 10
    decode_batch_size: int = 8
    skip_existing: bool = True


@dataclass
class AppConfig:
    project: ProjectConfig
    paths: PathConfig
    logging: LoggingConfig
    frame_loader: FrameLoaderConfig
    embedding: EmbeddingConfig
    keyframe: KeyframeConfig
    omni: OmniConfig = field(default_factory=OmniConfig)


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
    frame_loader = raw.get("frame_loader", {})
    embedding = raw.get("embedding", {})
    keyframe = raw.get("keyframe", {})
    omni = raw.get("omni", {})

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
        frame_loader=FrameLoaderConfig(
            backend=str(frame_loader.get("backend", "nvdec")),
            image_size=frame_loader.get("image_size", 336),
            num_workers=int(frame_loader.get("num_workers", 0)),
            pin_memory=_as_bool(frame_loader.get("pin_memory", True), True),
            persistent_workers=_as_bool(frame_loader.get("persistent_workers", False), False),
        ),
        embedding=EmbeddingConfig(
            model_name=str(embedding.get("model_name", "PE-Core-L14-336")),
            backend=embedding.get("backend"),
            pretrained=str(embedding.get("pretrained", "true")),
            batch_size=int(embedding.get("batch_size", 16)),
            device=str(embedding.get("device", "auto")),
            precision=str(embedding.get("precision", "fp16")),
            skip_existing=_as_bool(embedding.get("skip_existing", True), True),
            sample_fps=float(embedding.get('sample_fps', 3.0)) if embedding.get('sample_fps', 3.0) is not None else None,
        ),
        keyframe=KeyframeConfig(
            max_scene_gap_frames=int(keyframe.get("max_scene_gap_frames", 5000)),
            max_scene_seconds=(
                None
                if keyframe.get("max_scene_seconds", 30.0) is None
                else float(keyframe.get("max_scene_seconds", 30.0))
            ),
            final_similarity_threshold=float(keyframe.get("final_similarity_threshold", 0.96)),
            min_hist_bins=int(keyframe.get("min_hist_bins", 5)),
            image_quality=int(keyframe.get("image_quality", 95)),
            save_images=_as_bool(keyframe.get("save_images", True), True),
            skip_existing=_as_bool(keyframe.get("skip_existing", True), True),
        ),
        omni=OmniConfig(
            weights_path=_resolve(root, omni["weights_path"]) if omni.get("weights_path") else None,
            cache_dir=_resolve(root, omni.get("cache_dir", "weights/omni")),
            overlap=int(omni.get("overlap", 10)),
            decode_batch_size=int(omni.get("decode_batch_size", 8)),
            skip_existing=_as_bool(omni.get("skip_existing", True), True),
        ),
    )
