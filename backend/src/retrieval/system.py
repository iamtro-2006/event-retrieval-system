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

from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

from src.retrieval.index.factory import (
    build_faiss_index,
    build_index_manager_from_config,
)
from src.retrieval.retriever.asr_search.factory import build_asr_search_pipeline
from src.retrieval.retriever.common.orchestrator import Orchestrator, QueryPlan
from src.retrieval.retriever.ocr_search.factory import build_ocr_search_pipeline
from src.retrieval.retriever.semantic_search.factory import (
    build_semantic_search_pipeline,
)
from src.retrieval.retriever.temporal_search.factory import (
    build_temporal_search_pipeline,
)
from src.retrieval.retriever.text_search.factory import build_text_search_pipeline
from src.translation.base_translator import BaseTranslator
from src.translation.factory import get_translator

BACKEND_DIR = Path(__file__).resolve().parents[2]

_PATH_KEYS = {
    "index_path",
    "metadata_path",
    "vector_cache_path",
    "repo_path",
    "checkpoint_path",
    "spm_path",
    "config_path",
    "weights_path",
    "local_dir",
}


def _looks_like_url(value: str) -> bool:
    return value.startswith(("http://", "https://", "hf://"))


def _resolve_backend_path(value: str | Path) -> str:
    path_text = str(value).replace("\\", "/")
    if _looks_like_url(path_text):
        return path_text
    path = Path(path_text)
    return str(path if path.is_absolute() else BACKEND_DIR / path)


def _resolve_known_paths(obj: Any) -> Any:
    """Return a copy where config path fields are absolute backend paths."""
    if isinstance(obj, list):
        return [_resolve_known_paths(item) for item in obj]
    if not isinstance(obj, dict):
        return obj

    resolved: dict[str, Any] = {}
    for key, value in obj.items():
        if isinstance(value, (dict, list)):
            resolved[key] = _resolve_known_paths(value)
        elif key in _PATH_KEYS and value not in (None, ""):
            resolved[key] = _resolve_backend_path(value)
        else:
            resolved[key] = value
    return resolved


def _prepare_runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    return _resolve_known_paths(deepcopy(config))


class RetrievalSystem:
    """Thin wrapper around `Orchestrator`, expose 1 hàm phẳng / endpoint.

    Router chỉ gọi đúng 1 trong các hàm `search_*` này — không tự chọn
    retriever, không biết gì về `Orchestrator`/`SemanticIndex`/factory.
    """

    def __init__(
        self,
        orchestrator: Orchestrator,
        translator: BaseTranslator | None = None,
        translate_cfg: dict[str, Any] | None = None,
        fusion_cfg: dict[str, Any] | None = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._translator = translator
        self._translate_cfg = translate_cfg or {}
        self._fusion_cfg = fusion_cfg or {}

    @property
    def orchestrator(self) -> Orchestrator:
        """Escape hatch cho các trường hợp cần truy cập trực tiếp (vd. test)."""
        return self._orchestrator

    def _translate_query(self, query: str, translate: bool | None) -> str:
        """Bước tiền xử lý dịch vi->en TRƯỚC khi vào Orchestrator (chỉ dùng
        cho semantic/temporal/auto — xem docstring class về lý do KHÔNG áp
        cho ocr/asr/advanced).

        `translate=None` (mặc định) -> dùng `translate.enabled_default`
        trong config. `translate=True/False` -> ép buộc, ghi đè config.

        Nếu `translate` hiệu lực là True nhưng subsystem dịch chưa được cấu
        hình/khởi tạo thất bại lúc `build_system()`:
        - `translate=True` (client CHỦ ĐỘNG yêu cầu) -> raise `RuntimeError`
          (router map thành 503, cùng pattern với OCR/ASR).
        - `translate=None` (chỉ đang dùng default) -> bỏ qua dịch, search
          bằng query gốc, KHÔNG raise (tránh biến toàn bộ search mặc định
          thành 503 hàng loạt chỉ vì subsystem dịch bị lỗi hạ tầng — search
          vẫn chạy được, chỉ là kém chính xác hơn với model chỉ hiểu tiếng
          Anh tốt).
        """
        cfg = self._translate_cfg
        effective = (
            bool(cfg.get("enabled_default", False))
            if translate is None
            else bool(translate)
        )
        if not effective:
            return query
        if self._translator is None:
            if translate is True:
                raise RuntimeError(
                    "Translation chưa được cấu hình hoặc khởi tạo thất bại (xem 'translate_agent' trong config)."
                )
            return query
        return self._translator.translate(
            query,
            source=str(cfg.get("source", "vi")),
            target=str(cfg.get("target", "en")),
        )

    def search_semantic(
        self,
        query: str,
        top_k: int = 10,
        use_split: bool = True,
        candidate_multiplier: int = 5,
        model_key: str | None = None,
        translate: bool | None = None,
    ) -> tuple[pd.DataFrame, QueryPlan]:
        query = self._translate_query(query, translate)
        return self._orchestrator.run_search(
            query,
            mode="semantic",
            use_split=use_split,
            top_k=top_k,
            candidate_multiplier=candidate_multiplier,
            model_key=model_key,
        )

    def search_temporal(
        self,
        query: str,
        top_k: int = 10,
        use_split: bool = True,
        candidate_multiplier: int = 5,
        duration_limit: float = -1,
        model_key: str | None = None,
        translate: bool | None = None,
    ) -> tuple[pd.DataFrame, QueryPlan]:
        query = self._translate_query(query, translate)
        return self._orchestrator.run_search(
            query,
            mode="temporal",
            use_split=use_split,
            top_k=top_k,
            candidate_multiplier=candidate_multiplier,
            duration_limit=duration_limit,
            model_key=model_key,
        )

    def search_ocr(self, query: str, top_k: int = 10) -> tuple[pd.DataFrame, QueryPlan]:
        # Cố ý KHÔNG dịch — OCR search trên text tiếng Việt thô trong khung
        # hình (xem orchestrator.run_search: "OCR search operates on the
        # raw query text").
        return self._orchestrator.run_search(query, mode="ocr", top_k=top_k)

    def search_asr(self, query: str, top_k: int = 10) -> tuple[pd.DataFrame, QueryPlan]:
        # Cố ý KHÔNG dịch — tương tự search_ocr (ASR transcript tiếng Việt thô).
        return self._orchestrator.run_search(query, mode="asr", top_k=top_k)

    def search_text(
        self, query: str, top_k: int = 10
    ) -> tuple[pd.DataFrame, QueryPlan]:
        # Cố ý KHÔNG dịch — text search match OCR/ASR text thô.
        return self._orchestrator.run_search(query, mode="text", top_k=top_k)

    def search_fusion(
        self,
        query: str,
        top_k: int = 10,
        use_split: bool = True,
        candidate_multiplier: int = 5,
        translate: bool | None = None,
    ) -> tuple[pd.DataFrame, QueryPlan]:
        """Fusion search (semantic + text): semantic arm dùng query đã dịch,
        text arm dùng query gốc (raw)."""
        translated_query = self._translate_query(query, translate)
        return self._orchestrator.run_search(
            query,
            mode="fusion",
            use_split=use_split,
            top_k=top_k,
            candidate_multiplier=candidate_multiplier,
            translated_query=translated_query,
            fusion_config=self._fusion_cfg,
        )

    def search_auto(
        self,
        query: str,
        top_k: int = 10,
        use_split: bool = True,
        candidate_multiplier: int = 5,
        translate: bool | None = None,
    ) -> tuple[pd.DataFrame, QueryPlan]:
        # "auto" mode luôn chọn semantic hoặc temporal (run_search: effective_mode
        # in {"semantic","temporal"} khi mode="auto") — chưa bao giờ chọn ocr/asr,
        # nên dịch ở đây an toàn giống search_semantic/search_temporal.
        query = self._translate_query(query, translate)
        return self._orchestrator.run_search(
            query,
            mode="auto",
            use_split=use_split,
            top_k=top_k,
            candidate_multiplier=candidate_multiplier,
        )

    def available_models(self) -> list[str]:
        """Model keys usable to build the `advanced_search` checklist UI
        (semantic/temporal sections)."""
        return self._orchestrator.available_semantic_models()

    def search_advanced(
        self,
        query: str,
        semantic_models: list[str] | None = None,
        temporal: bool = False,
        use_ocr: bool = False,
        use_asr: bool = False,
        top_k: int = 10,
        use_split: bool = True,
        candidate_multiplier: int = 5,
        duration_limit: float = -1,
        weights: dict[str, float] | None = None,
    ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        """Advanced search: combine several ticked semantic models + an
        on/off `temporal` toggle (temporal search runs on that same ticked
        model list, combined via RRF before DP alignment — no separate
        temporal model checklist) + ocr/asr, via Reciprocal Rank Fusion. See
        `Orchestrator.advanced_search` for the full contract (this is a thin
        passthrough, matching the shape of `search_semantic`/`search_temporal`/
        etc. above).

        KHÔNG dịch `query` ở đây (khác `search_semantic`/`search_temporal`/
        `search_auto`): `Orchestrator.advanced_search` dùng CHUNG 1 chuỗi
        `query` cho cả semantic/temporal LẪN ocr/asr (không có tham số
        "semantic query" riêng), nên dịch trước khi gọi xuống đây sẽ dịch
        luôn cả phần OCR/ASR (vốn cần giữ nguyên tiếng Việt) — cần sửa
        `Orchestrator.advanced_search` để tách 2 đường trước khi wire dịch
        vào đây an toàn được; để lại cho lần sau (xem handoff note)."""
        return self._orchestrator.advanced_search(
            query,
            semantic_models=semantic_models,
            temporal=temporal,
            use_ocr=use_ocr,
            use_asr=use_asr,
            top_k=top_k,
            use_split=use_split,
            candidate_multiplier=candidate_multiplier,
            duration_limit=duration_limit,
            weights=weights,
        )


def _build_ocr_pipeline_or_none(cfg: dict[str, Any]):
    """OCR là subsystem optional — lỗi ở đây không được làm sập cả hệ thống."""
    ocr_cfg = cfg.get("ocr")
    if not ocr_cfg:
        return None
    try:
        config_path = (
            ocr_cfg.get("config_path", "configs/ocr_extraction.yaml")
            if isinstance(ocr_cfg, dict)
            else ocr_cfg
        )
        inline_cfg = ocr_cfg.get("cfg") if isinstance(ocr_cfg, dict) else None
        return build_ocr_search_pipeline(cfg=inline_cfg, config_path=config_path)
    except Exception as exc:  # pragma: no cover - lỗi hạ tầng (ES down, v.v.)
        print(
            f"[system] OCR pipeline init failed, disabling OCR search: {type(exc).__name__}: {exc}"
        )
        return None


def _build_asr_pipeline_or_none(cfg: dict[str, Any]):
    """ASR là subsystem optional — tương tự OCR."""
    asr_cfg = cfg.get("asr")
    if not asr_cfg:
        return None
    try:
        config_path = (
            asr_cfg.get("config_path", "configs/asr_extraction.yaml")
            if isinstance(asr_cfg, dict)
            else asr_cfg
        )
        inline_cfg = asr_cfg.get("cfg") if isinstance(asr_cfg, dict) else None
        return build_asr_search_pipeline(cfg=inline_cfg, config_path=config_path)
    except Exception as exc:  # pragma: no cover
        print(
            f"[system] ASR pipeline init failed, disabling ASR search: {type(exc).__name__}: {exc}"
        )
        return None


def _build_text_pipeline_or_none(cfg: dict[str, Any]):
    """Unified text (OCR+ASR) là subsystem optional — tương tự OCR/ASR."""
    text_cfg = cfg.get("text")
    if not text_cfg:
        return None
    try:
        config_path = (
            text_cfg.get("config_path", "configs/text.yaml")
            if isinstance(text_cfg, dict)
            else text_cfg
        )
        inline_cfg = text_cfg.get("cfg") if isinstance(text_cfg, dict) else None
        return build_text_search_pipeline(cfg=inline_cfg, config_path=config_path)
    except Exception as exc:  # pragma: no cover
        print(
            f"[system] Text pipeline init failed, disabling text/fusion search: {type(exc).__name__}: {exc}"
        )
        return None


def _build_translator_or_none(config: dict[str, Any]) -> BaseTranslator | None:
    """Translation là subsystem optional — tương tự OCR/ASR, lỗi ở đây
    (thiếu network để tải HF model, thiếu API key cho backend `llm`, sai
    `translate_agent`, v.v.) không được làm sập cả hệ thống search; chỉ vô
    hiệu hoá bước dịch (search vẫn chạy bằng query gốc, xem
    `RetrievalSystem._translate_query`)."""
    if not config.get("translate") and not config.get("translate_agent"):
        return None
    try:
        backend_dir = Path(__file__).resolve().parents[2]
        return get_translator(config, backend_dir)
    except (
        Exception
    ) as exc:  # pragma: no cover - lỗi hạ tầng (network, API key, model load...)
        print(
            f"[system] Translator init failed, disabling translation: {type(exc).__name__}: {exc}"
        )
        return None


def build_system(config: dict[str, Any]) -> RetrievalSystem:
    """Dựng toàn bộ `RetrievalSystem` từ 1 config dict, load model 1 LẦN.

    Args:
        config: dict với các khoá con:
            - "semantic"/"indexes"/"models": danh sách (hoặc dict có key
              "models") các model để load — mỗi entry là kwargs cho
              `FaissIndex.__init__` (index_path, metadata_path, model_name,
              backend, pretrained, model_key, device, precision, normalize,
              ef_search, faiss_threads, vector_cache_mode, vector_cache_dtype,
              vector_cache_path, allow_npy_fallback, compile_model,
              model_extra). Chấp nhận CẢ 2 dạng để tương thích ngược:
                - dict đơn (1 model, dạng cũ)         -> build 1 `FaissIndex`
                - list[dict] hoặc {"models": [...]}   -> build `IndexManager`
                  nhiều model, dùng chung cho semantic VÀ temporal (chọn
                  model nào ở từng request qua `model_key`, xem
                  `RetrievalSystem.search_semantic/search_temporal/search_advanced`).
              Bắt buộc.
            - "ocr": optional. Có thể là str/Path (config_path) hoặc
              {"config_path": ...} hoặc {"cfg": {...đã parse sẵn...}}.
              Bỏ qua hoặc None -> OCR search bị vô hiệu hoá (raise khi gọi).
            - "asr": tương tự "ocr".
            - "translate_agent"/"translate": optional (xem
              `src/translation/factory.py::get_translator` cho shape đầy
              đủ). Bỏ qua hoặc khởi tạo lỗi -> dịch bị vô hiệu hoá, các hàm
              `search_semantic`/`search_temporal`/`search_auto` chạy bằng
              query gốc (KHÔNG raise, xem
              `RetrievalSystem._translate_query`) trừ khi client CHỦ ĐỘNG
              truyền `translate=True` (khi đó raise `RuntimeError` -> 503).

    Returns:
        `RetrievalSystem` singleton, dùng chung cho toàn bộ lifetime của app
        (dựng 1 lần trong `api/main.py` lifespan, lưu vào `app.state`).
    """
    config = _prepare_runtime_config(config)
    semantic_cfg = config.get("semantic", config.get("indexes"))
    is_multi_model = isinstance(semantic_cfg, list) or (
        isinstance(semantic_cfg, dict) and "models" in semantic_cfg
    )
    index = (
        build_index_manager_from_config(semantic_cfg, milvus_cfg=config.get("milvus"))
        if is_multi_model
        else build_faiss_index(semantic_cfg)
    )

    # 4 backend deu la SearchPipeline duoc build qua cung 1 dang factory
    # `build_*_search_pipeline(...)` roi expose thong nhat qua `.search(...)`
    # (khong con class nghiep vu rieng SemanticSearch/TemporalSearch).
    # `index` o day co the la 1 FaissIndex (1 model) hoac 1 IndexManager
    # (nhieu model) - ca 2 pipeline deu chap nhan (xem `_resolve()` cua tung
    # pipeline), semantic va temporal DUNG CHUNG index_manager (khong load
    # model 2 lan).
    semantic_search = build_semantic_search_pipeline(index)
    temporal_search = build_temporal_search_pipeline(index)
    ocr_pipeline = _build_ocr_pipeline_or_none(config)
    asr_pipeline = _build_asr_pipeline_or_none(config)
    text_pipeline = _build_text_pipeline_or_none(config)
    translator = _build_translator_or_none(config)

    orchestrator = Orchestrator(
        index=index,
        semantic_search=semantic_search,
        temporal_search=temporal_search,
        ocr_search_pipeline=ocr_pipeline,
        asr_search_pipeline=asr_pipeline,
        text_search_pipeline=text_pipeline,
    )
    return RetrievalSystem(
        orchestrator,
        translator=translator,
        translate_cfg=config.get("translate"),
        fusion_cfg=config.get("fusion"),
    )
