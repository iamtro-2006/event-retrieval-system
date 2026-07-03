"""retrieval/system.py — FACADE DUY NHẤT giữa `api/` và `retrieval/retriever/*`.

Xem REFACTOR_PLAN.md mục 1b / Prompt 6b.

`api/` chỉ được phép import từ module này (`from src.retrieval.system import
build_system, RetrievalSystem`) — không tự gọi rải rác các `factory.py` của
từng retriever con. Bên trong `system.py` chỉ điều phối (gọi các factory 1
lần lúc khởi động, rồi expose các hàm `search_*` phẳng); toàn bộ logic search
thật sự vẫn nằm ở `common/orchestrator.py` và các retriever con.

Reranker KHÔNG được wire vào đây (xem retriever/reranker/pipeline/rerank.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.retrieval.index.factory import build_clip_faiss_index
from src.retrieval.retriever.asr_search.factory import build_asr_search_pipeline
from src.retrieval.retriever.common.orchestrator import Orchestrator, QueryPlan
from src.retrieval.retriever.ocr_search.factory import build_ocr_search_pipeline
from src.retrieval.retriever.semantic_search.factory import build_semantic_search_pipeline
from src.retrieval.retriever.temporal_search.factory import build_temporal_search_pipeline


class RetrievalSystem:
    """Thin wrapper around `Orchestrator`, expose 1 hàm phẳng / endpoint.

    Router chỉ gọi đúng 1 trong các hàm `search_*` này — không tự chọn
    retriever, không biết gì về `Orchestrator`/`SemanticIndex`/factory.
    """

    def __init__(self, orchestrator: Orchestrator) -> None:
        self._orchestrator = orchestrator

    @property
    def orchestrator(self) -> Orchestrator:
        """Escape hatch cho các trường hợp cần truy cập trực tiếp (vd. test)."""
        return self._orchestrator

    def search_semantic(
        self, query: str, top_k: int = 10, use_split: bool = True, candidate_multiplier: int = 5
    ) -> tuple[pd.DataFrame, QueryPlan]:
        return self._orchestrator.run_search(
            query, mode="semantic", use_split=use_split, top_k=top_k, candidate_multiplier=candidate_multiplier
        )

    def search_temporal(
        self,
        query: str,
        top_k: int = 10,
        use_split: bool = True,
        candidate_multiplier: int = 5,
        duration_limit: float = -1,
    ) -> tuple[pd.DataFrame, QueryPlan]:
        return self._orchestrator.run_search(
            query,
            mode="temporal",
            use_split=use_split,
            top_k=top_k,
            candidate_multiplier=candidate_multiplier,
            duration_limit=duration_limit,
        )

    def search_ocr(self, query: str, top_k: int = 10) -> tuple[pd.DataFrame, QueryPlan]:
        return self._orchestrator.run_search(query, mode="ocr", top_k=top_k)

    def search_asr(self, query: str, top_k: int = 10) -> tuple[pd.DataFrame, QueryPlan]:
        return self._orchestrator.run_search(query, mode="asr", top_k=top_k)

    def search_auto(
        self, query: str, top_k: int = 10, use_split: bool = True, candidate_multiplier: int = 5
    ) -> tuple[pd.DataFrame, QueryPlan]:
        return self._orchestrator.run_search(
            query, mode="auto", use_split=use_split, top_k=top_k, candidate_multiplier=candidate_multiplier
        )


def _build_ocr_pipeline_or_none(cfg: dict[str, Any]):
    """OCR là subsystem optional — lỗi ở đây không được làm sập cả hệ thống."""
    ocr_cfg = cfg.get("ocr")
    if not ocr_cfg:
        return None
    try:
        config_path = ocr_cfg.get("config_path", "configs/ocr.yaml") if isinstance(ocr_cfg, dict) else ocr_cfg
        inline_cfg = ocr_cfg.get("cfg") if isinstance(ocr_cfg, dict) else None
        return build_ocr_search_pipeline(cfg=inline_cfg, config_path=config_path)
    except Exception as exc:  # pragma: no cover - lỗi hạ tầng (ES down, v.v.)
        print(f"[system] OCR pipeline init failed, disabling OCR search: {type(exc).__name__}: {exc}")
        return None


def _build_asr_pipeline_or_none(cfg: dict[str, Any]):
    """ASR là subsystem optional — tương tự OCR."""
    asr_cfg = cfg.get("asr")
    if not asr_cfg:
        return None
    try:
        config_path = asr_cfg.get("config_path", "configs/asr.yaml") if isinstance(asr_cfg, dict) else asr_cfg
        inline_cfg = asr_cfg.get("cfg") if isinstance(asr_cfg, dict) else None
        return build_asr_search_pipeline(cfg=inline_cfg, config_path=config_path)
    except Exception as exc:  # pragma: no cover
        print(f"[system] ASR pipeline init failed, disabling ASR search: {type(exc).__name__}: {exc}")
        return None


def build_system(config: dict[str, Any]) -> RetrievalSystem:
    """Dựng toàn bộ `RetrievalSystem` từ 1 config dict, load model 1 LẦN.

    Args:
        config: dict với 3 khoá con:
            - "semantic": kwargs truyền thẳng vào `ClipFaissIndex.__init__`
              (index_path, metadata_path, model_name, pretrained, device,
              precision, normalize, ef_search, faiss_threads,
              vector_cache_mode, vector_cache_dtype, vector_cache_path,
              allow_npy_fallback, compile_model). Bắt buộc.
            - "ocr": optional. Có thể là str/Path (config_path) hoặc
              {"config_path": ...} hoặc {"cfg": {...đã parse sẵn...}}.
              Bỏ qua hoặc None -> OCR search bị vô hiệu hoá (raise khi gọi).
            - "asr": tương tự "ocr".

    Returns:
        `RetrievalSystem` singleton, dùng chung cho toàn bộ lifetime của app
        (dựng 1 lần trong `api/main.py` lifespan, lưu vào `app.state`).
    """
    index = build_clip_faiss_index(config["semantic"])
    # 4 backend deu la SearchPipeline duoc build qua cung 1 dang factory
    # `build_*_search_pipeline(...)` roi expose thong nhat qua `.search(...)`
    # (khong con class nghiep vu rieng SemanticSearch/TemporalSearch).
    semantic_search = build_semantic_search_pipeline(index)
    temporal_search = build_temporal_search_pipeline(index)
    ocr_pipeline = _build_ocr_pipeline_or_none(config)
    asr_pipeline = _build_asr_pipeline_or_none(config)

    orchestrator = Orchestrator(
        index=index,
        semantic_search=semantic_search,
        temporal_search=temporal_search,
        ocr_search_pipeline=ocr_pipeline,
        asr_search_pipeline=asr_pipeline,
    )
    return RetrievalSystem(orchestrator)
