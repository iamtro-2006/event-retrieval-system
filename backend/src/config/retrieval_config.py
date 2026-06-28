from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config.loader import load_yaml


@dataclass(frozen=True)
class MilvusSearchParams:
    metric_type: str = "IP"
    ef: int = 64


@dataclass(frozen=True)
class MilvusConfig:
    host: str = "localhost"
    port: int = 19530
    collection_name: str = "keyframes"
    consistency_level: str = "Bounded"
    search_params: MilvusSearchParams = field(default_factory=MilvusSearchParams)


@dataclass(frozen=True)
class FaissConfig:
    index_path: str = ""
    metadata_path: str = ""
    ef_search: int = 64
    threads: int | None = None
    vector_cache_mode: str | None = None
    vector_cache_dtype: str = "float32"
    vector_cache_path: str | None = None


@dataclass(frozen=True)
class ModelConfig:
    name: str = "ViT-L-16-SigLIP-256"
    pretrained: str = "webli"
    device: str = "auto"
    precision: str = "fp32"
    normalize: bool = True
    compile: bool = False


@dataclass(frozen=True)
class PathsConfig:
    keyframes_root: str = "data/processed/keyframes"
    videos_root: str = "data/raw/videos"
    map_keyframe_path: str = "data/processed/map_keyframes"


@dataclass(frozen=True)
class SearchConfig:
    candidate_multiplier: int = 5
    default_top_k: int = 20
    max_top_k: int = 200


@dataclass(frozen=True)
class UiConfig:
    surrounding_radius: int = 5
    max_surrounding_radius: int = 10


@dataclass(frozen=True)
class TranslateConfig:
    enabled_default: bool = False
    source: str = "vi"
    target: str = "en"


@dataclass(frozen=True)
class SpeechConfig:
    model_size: str = "base"
    device: str = "cpu"
    compute_type: str = "int8"
    lazy_load: bool = True


@dataclass(frozen=True)
class DebugConfig:
    profile: bool = False


@dataclass(frozen=True)
class RetrievalConfig:
    backend: str = "milvus"
    milvus: MilvusConfig = field(default_factory=MilvusConfig)
    faiss: FaissConfig = field(default_factory=FaissConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    ui: UiConfig = field(default_factory=UiConfig)
    translate: TranslateConfig = field(default_factory=TranslateConfig)
    speech: SpeechConfig = field(default_factory=SpeechConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)


def load_retrieval_config(path: str | Path) -> RetrievalConfig:
    raw = load_yaml(path)

    milvus_raw = raw.get("milvus", {})
    search_params_raw = milvus_raw.get("search_params", {})
    params_raw = search_params_raw.get("params", {})

    faiss_raw = raw.get("faiss", {})
    model_raw = raw.get("model", {})
    paths_raw = raw.get("paths", {})
    search_raw = raw.get("search", {})
    ui_raw = raw.get("ui", {})
    translate_raw = raw.get("translate", {})
    speech_raw = raw.get("speech", {})
    debug_raw = raw.get("debug", {})

    return RetrievalConfig(
        backend=str(raw.get("backend", "milvus")).strip().lower(),
        milvus=MilvusConfig(
            host=str(milvus_raw.get("host", "localhost")),
            port=int(milvus_raw.get("port", 19530)),
            collection_name=str(milvus_raw.get("collection_name", "keyframes")),
            consistency_level=str(milvus_raw.get("consistency_level", "Bounded")),
            search_params=MilvusSearchParams(
                metric_type=str(search_params_raw.get("metric_type", "IP")),
                ef=int(params_raw.get("ef", 64)),
            ),
        ),
        faiss=FaissConfig(
            index_path=str(faiss_raw.get("index_path", "")),
            metadata_path=str(faiss_raw.get("metadata_path", "")),
            ef_search=int(faiss_raw.get("ef_search", 64)),
            threads=faiss_raw.get("threads"),
            vector_cache_mode=faiss_raw.get("vector_cache_mode"),
            vector_cache_dtype=str(faiss_raw.get("vector_cache_dtype", "float32")),
            vector_cache_path=faiss_raw.get("vector_cache_path"),
        ),
        model=ModelConfig(
            name=str(model_raw.get("name", "ViT-L-16-SigLIP-256")),
            pretrained=str(model_raw.get("pretrained", "webli")),
            device=str(model_raw.get("device", "auto")),
            precision=str(model_raw.get("precision", "fp32")),
            normalize=bool(model_raw.get("normalize", True)),
            compile=bool(model_raw.get("compile", False)),
        ),
        paths=PathsConfig(
            keyframes_root=str(
                paths_raw.get("keyframes_root", "data/processed/keyframes")
            ),
            videos_root=str(paths_raw.get("videos_root", "data/raw/videos")),
            map_keyframe_path=str(
                paths_raw.get("map_keyframe_path", "data/processed/map_keyframes")
            ),
        ),
        search=SearchConfig(
            candidate_multiplier=int(search_raw.get("candidate_multiplier", 5)),
            default_top_k=int(search_raw.get("default_top_k", 20)),
            max_top_k=int(search_raw.get("max_top_k", 200)),
        ),
        ui=UiConfig(
            surrounding_radius=int(ui_raw.get("surrounding_radius", 5)),
            max_surrounding_radius=int(ui_raw.get("max_surrounding_radius", 10)),
        ),
        translate=TranslateConfig(
            enabled_default=bool(translate_raw.get("enabled_default", False)),
            source=str(translate_raw.get("source", "vi")),
            target=str(translate_raw.get("target", "en")),
        ),
        speech=SpeechConfig(
            model_size=str(speech_raw.get("model_size", "base")),
            device=str(speech_raw.get("device", "cpu")),
            compute_type=str(speech_raw.get("compute_type", "int8")),
            lazy_load=bool(speech_raw.get("lazy_load", True)),
        ),
        debug=DebugConfig(
            profile=bool(debug_raw.get("profile", False)),
        ),
    )


def to_dict(cfg: RetrievalConfig) -> dict[str, Any]:
    """Convert a RetrievalConfig back into the raw dict shape for the
    create_retrieval_backend factory and code paths still using dicts.

    This is a bridge: existing callers (retrieval_backend.create_retrieval_backend,
    main.py route handlers) expect the raw dict shape. As those callers are
    migrated to typed access, they can stop calling this.
    """
    from dataclasses import asdict

    raw = asdict(cfg)
    raw["milvus"]["search_params"]["params"] = {
        "ef": raw["milvus"]["search_params"].pop("ef")
    }
    return raw
