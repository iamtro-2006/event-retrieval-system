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
class SceneConfig:
    max_duration_sec: float = 20.0


@dataclass
class CandidateConfig:
    min_gap_sec: float = 0.25
    max_gap_sec: float = 1.0
    phash_min_distance: int = 6
    pixel_change_threshold: float = 0.035
    boundary_margin_sec: float = 0.30


@dataclass
class QualityConfig:
    blur_min: float = 20.0
    brightness_min: float = 10.0
    brightness_max: float = 245.0
    entropy_min: float = 1.2
    edge_density_min: float = 0.002
    clipped_fraction_max: float = 0.60


@dataclass
class ClusteringConfig:
    max_clusters_per_shot: int = 5
    min_cluster_size: int = 2
    min_spread: float = 0.025
    aucc_complexity_penalty: float = 0.035
    kmeans_n_init: int = 10
    kmedoids_max_iter: int = 50


@dataclass
class SelectorConfig:
    common_lambda: float = 0.70
    local_neighbor_radius: int = 2
    unique_alpha: float = 0.50
    min_unique_quality: float = 0.35
    max_common_cosine: float = 0.90
    min_shot_spread: float = 0.025
    min_volatility_range: float = 0.02
    min_unique_volatility: float = 0.60
    min_unique_atypicality: float = 0.50
    unique_boundary_margin_sec: float = 0.30
    mmr_lambda: float = 0.60
    min_temporal_gap_sec: float = 1.0
    global_novelty_cosine: float = 0.90
    max_extras_per_shot: int = 2


@dataclass
class DedupConfig:
    phash_hamming_threshold: int = 5
    dense_cosine_threshold: float = 0.97
    temporal_window_sec: float = 4.0


@dataclass
class KeyframeConfig:
    strategy: str = "legacy_lmske"
    min_scene_frames: int = 3
    max_scene_gap_frames: int = 5000
    hist_threshold: float = 0.90
    min_hist_bins: int = 10
    image_quality: int = 95
    save_images: bool = True
    skip_existing: bool = True
    selection_embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    candidate: CandidateConfig = field(default_factory=CandidateConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    selector: SelectorConfig = field(default_factory=SelectorConfig)
    dedup: DedupConfig = field(default_factory=DedupConfig)


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
    selection_embedding = keyframe.get("selection_embedding", {})
    scene = keyframe.get("scene", {})
    candidate = keyframe.get("candidate", {})
    quality = keyframe.get("quality", {})
    clustering = keyframe.get("clustering", {})
    selector = keyframe.get("selector", {})
    dedup = keyframe.get("dedup", {})

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
            strategy=str(keyframe.get("strategy", "legacy_lmske")),
            min_scene_frames=int(keyframe.get("min_scene_frames", 3)),
            max_scene_gap_frames=int(keyframe.get("max_scene_gap_frames", 5000)),
            hist_threshold=float(keyframe.get("hist_threshold", 0.90)),
            min_hist_bins=int(keyframe.get("min_hist_bins", 10)),
            image_quality=int(keyframe.get("image_quality", 95)),
            save_images=_as_bool(keyframe.get("save_images", True), True),
            skip_existing=_as_bool(keyframe.get("skip_existing", True), True),
            selection_embedding=EmbeddingConfig(
                model_name=str(selection_embedding.get("model_name", "MobileCLIP2-S4")),
                pretrained=str(selection_embedding.get("pretrained", "dfndr2b")),
                batch_size=int(selection_embedding.get("batch_size", 16)),
                device=str(selection_embedding.get("device", "auto")),
                precision=str(selection_embedding.get("precision", "fp16")),
                skip_existing=_as_bool(selection_embedding.get("skip_existing", True), True),
            ),
            scene=SceneConfig(max_duration_sec=float(scene.get("max_duration_sec", 20.0))),
            candidate=CandidateConfig(
                min_gap_sec=float(candidate.get("min_gap_sec", 0.25)),
                max_gap_sec=float(candidate.get("max_gap_sec", 1.0)),
                phash_min_distance=int(candidate.get("phash_min_distance", 6)),
                pixel_change_threshold=float(candidate.get("pixel_change_threshold", 0.035)),
                boundary_margin_sec=float(candidate.get("boundary_margin_sec", 0.30)),
            ),
            quality=QualityConfig(
                blur_min=float(quality.get("blur_min", 20.0)),
                brightness_min=float(quality.get("brightness_min", 10.0)),
                brightness_max=float(quality.get("brightness_max", 245.0)),
                entropy_min=float(quality.get("entropy_min", 1.2)),
                edge_density_min=float(quality.get("edge_density_min", 0.002)),
                clipped_fraction_max=float(quality.get("clipped_fraction_max", 0.60)),
            ),
            clustering=ClusteringConfig(
                max_clusters_per_shot=int(clustering.get("max_clusters_per_shot", 5)),
                min_cluster_size=int(clustering.get("min_cluster_size", 2)),
                min_spread=float(clustering.get("min_spread", 0.025)),
                aucc_complexity_penalty=float(clustering.get("aucc_complexity_penalty", 0.035)),
                kmeans_n_init=int(clustering.get("kmeans_n_init", 10)),
                kmedoids_max_iter=int(clustering.get("kmedoids_max_iter", 50)),
            ),
            selector=SelectorConfig(
                common_lambda=float(selector.get("common_lambda", 0.70)),
                local_neighbor_radius=int(selector.get("local_neighbor_radius", 2)),
                unique_alpha=float(selector.get("unique_alpha", 0.50)),
                min_unique_quality=float(selector.get("min_unique_quality", 0.35)),
                max_common_cosine=float(selector.get("max_common_cosine", 0.90)),
                min_shot_spread=float(selector.get("min_shot_spread", 0.025)),
                min_volatility_range=float(selector.get("min_volatility_range", 0.02)),
                min_unique_volatility=float(selector.get("min_unique_volatility", 0.60)),
                min_unique_atypicality=float(selector.get("min_unique_atypicality", 0.50)),
                unique_boundary_margin_sec=float(selector.get("unique_boundary_margin_sec", 0.30)),
                mmr_lambda=float(selector.get("mmr_lambda", 0.60)),
                min_temporal_gap_sec=float(selector.get("min_temporal_gap_sec", 1.0)),
                global_novelty_cosine=float(selector.get("global_novelty_cosine", 0.90)),
                max_extras_per_shot=int(selector.get("max_extras_per_shot", 2)),
            ),
            dedup=DedupConfig(
                phash_hamming_threshold=int(dedup.get("phash_hamming_threshold", 5)),
                dense_cosine_threshold=float(dedup.get("dense_cosine_threshold", 0.97)),
                temporal_window_sec=float(dedup.get("temporal_window_sec", 4.0)),
            ),
        ),
    )
