"""`GET /api/health` + `GET /api/config` GỐC — port nguyên shape response từ
`main.py` cũ (frontend cũ đọc trực tiếp các field này để dựng UI: available
search modes, surrounding radius mặc định, trạng thái OCR/ASR, v.v.).

1 điểm bắt buộc phải đổi so với bản gốc: bản gốc đọc thẳng `CFG["model"]`
(1 model duy nhất, thiết kế app.yaml CŨ). `configs/app.yaml` giờ hỗ trợ
NHIỀU model (`semantic.models`, xem `src/retrieval/index/index_manager.py`),
không còn khoá `CFG["model"]` phẳng nữa — nên field `model` ở đây được suy
ra từ model MẶC ĐỊNH đang chạy (`orchestrator.index`, luôn resolve về 1
`FaissIndex` cụ thể) thay vì đọc thẳng config. Response *shape* (tên field)
giữ nguyên 100%, chỉ khác nguồn dữ liệu.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from src.api.legacy.deps import (
    get_cfg,
    get_legacy_index,
    get_legacy_paths,
    get_legacy_system,
)
from src.api.legacy.paths import LegacyPaths
from src.retrieval.index.faiss_index import FaissIndex
from src.retrieval.system import RetrievalSystem

router = APIRouter(tags=["legacy-health"])


@router.get("/api/health")
def health(
    system: RetrievalSystem = Depends(get_legacy_system),
    clip_index: FaissIndex = Depends(get_legacy_index),
    cfg: dict[str, Any] = Depends(get_cfg),
    paths: LegacyPaths = Depends(get_legacy_paths),
):
    """Health check endpoint returning system and configuration status."""
    orchestrator = system.orchestrator
    return {
        "status": "ok",
        "backend_dir": str(paths.backend_dir),
        "config_path": str(paths.config_path),
        "faiss_index_path": str(clip_index.index_path),
        "metadata_path": str(clip_index.metadata_path),
        "vector_cache_path": str(clip_index.vector_cache_path or ""),
        "vector_cache_exists": bool(
            clip_index.vector_cache_path and clip_index.vector_cache_path.exists()
        ),
        "vector_cache": clip_index.cache_info,
        "keyframes_root": str(paths.keyframes_root),
        "videos_root": str(paths.videos_root),
        "map_keyframe_path": str(paths.map_keyframe_root),
        "keyframes_root_exists": paths.keyframes_root.exists(),
        "videos_root_exists": paths.videos_root.exists(),
        "map_keyframe_root_exists": paths.map_keyframe_root.exists(),
        "faiss_index_exists": clip_index.index_path.exists(),
        "metadata_exists": clip_index.metadata_path.exists(),
        "model": {
            "name": clip_index.model_name,
            "pretrained": clip_index.pretrained,
            "device": str(clip_index.device),
            "precision": clip_index.precision,
            "normalize": bool(clip_index.normalize),
        },
        "available_models": orchestrator.available_semantic_models(),
        "ocr": {
            "config_path": str(cfg.get("ocr", {}).get("config_path", ""))
            if isinstance(cfg.get("ocr"), dict)
            else "",
            "available": orchestrator.ocr_search_pipeline is not None,
        },
        "asr": {
            "config_path": str(cfg.get("asr", {}).get("config_path", ""))
            if isinstance(cfg.get("asr"), dict)
            else "",
            "available": orchestrator.asr_search_pipeline is not None,
        },
    }


@router.get("/api/config")
def get_public_config(
    system: RetrievalSystem = Depends(get_legacy_system),
    clip_index: FaissIndex = Depends(get_legacy_index),
    cfg: dict[str, Any] = Depends(get_cfg),
):
    """Return the public-facing configuration parameters for the frontend."""
    orchestrator = system.orchestrator
    return {
        "search": {
            "default_top_k": int(cfg["search"].get("default_top_k", 20)),
            "max_top_k": int(cfg["search"].get("max_top_k", 200)),
            "candidate_multiplier": int(cfg["search"].get("candidate_multiplier", 1)),
            "available_modes": [
                "semantic",
                "temporal",
                "ocr",
                "asr",
                "text",
                "fusion",
                "auto",
            ],
            "default_search_mode": "semantic",
            "default_duration_limit": -1,
        },
        "ui": {
            "surrounding_radius": int(cfg["ui"].get("surrounding_radius", 5)),
            "max_surrounding_radius": int(cfg["ui"].get("max_surrounding_radius", 10)),
        },
        "translate": {
            "enabled_default": bool(
                cfg.get("translate", {}).get("enabled_default", False)
            ),
            "source": cfg.get("translate", {}).get("source", "vi"),
            "target": cfg.get("translate", {}).get("target", "en"),
            "agent": str(cfg.get("translate_agent", "envit5")),
        },
        "model": {
            "name": clip_index.model_name,
            "pretrained": clip_index.pretrained,
            "device": str(clip_index.device),
            "precision": clip_index.precision,
            "normalize": bool(clip_index.normalize),
        },
        "available_models": orchestrator.available_semantic_models(),
        # `configs/app.yaml` giờ khai báo ef_search/threads/vector_cache_* theo
        # TỪNG model (`semantic.models[i]`, xem `index_manager.py`), không còn
        # 1 khoá `faiss` phẳng dùng chung ở root config như bản cũ nữa -> lấy
        # trực tiếp từ model mặc định đang chạy (`clip_index.cache_info`)
        # thay vì `cfg.get("faiss")` (giờ luôn rỗng).
        "faiss": {
            "vector_cache_mode": clip_index.cache_info.get("mode"),
            "vector_cache_dtype": clip_index.cache_info.get("dtype"),
            "vector_cache_path": str(clip_index.vector_cache_path or ""),
            "vector_cache_available": clip_index.cache_info.get("available", False),
            "allow_npy_fallback": bool(clip_index.allow_npy_fallback),
        },
        "ocr": {
            "available": orchestrator.ocr_search_pipeline is not None,
        },
        "asr": {
            "available": orchestrator.asr_search_pipeline is not None,
        },
    }
