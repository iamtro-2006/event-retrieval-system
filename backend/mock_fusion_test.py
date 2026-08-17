"""Mock test cho ADVANCED SEARCH (fusion RRF: semantic + temporal + ocr + asr).

Dung khi CHUA co Elasticsearch/OCR-index/ASR-index that (moi tick "OCR"/"ASR"
o UI se raise vi `ocr_search_pipeline`/`asr_search_pipeline` la None). Script
nay:

  - Dung index FAISS + semantic/temporal pipeline THAT (tu configs/app.yaml,
    build_system() y het luc server chay) - phan nay KHONG mock.
  - Chi mock tang duoi cung: `OCRRepository`/`ASRRepository`.search() (tuc la
    "gia lap Elasticsearch tra ve gi"). Toan bo logic phia tren (chuan hoa
    OCRHit/ASRHit trong `ocr_search/pipeline/search.py` +
    `asr_search/pipeline/search.py`, roi enrich/join voi FAISS metadata,
    normalize score, dung sang temporal-sequence card cho ASR trong
    `orchestrator._enrich_asr_hits`) la CODE THAT, khong mock - de con test
    dung phan fusion/enrich, chi khong can ES that.
  - Lay vai dong metadata THAT (video_id/keyframe_id/timestamp_sec that dang
    co trong index cua ban) de dung lam "hit gia" - dam bao mock hit resolve
    duoc qua metadata_row_for_ocr_hit/metadata_rows_for_asr_hit (khong bi
    drop het vi khong khop video_id/keyframe_id that).
  - Goi `Orchestrator.advanced_search(...)` that (ham fusion RRF that), in
    log chi tiet tung nguon truoc khi fuse + ket qua fuse cuoi cung.

Chay o thu muc backend/:

    python mock_fusion_test.py
    python mock_fusion_test.py --query "nguoi dan ong mac ao do di vao xe hoi" --top-k 10
    python mock_fusion_test.py --no-asr   # chi test semantic+temporal+ocr
    python mock_fusion_test.py --semantic-model siglip2_so400m

LUU Y: neu metadata.csv cua ban van dang bi bug timestamp_sec=0/NaN (xem
debug_temporal.py), ASR mock se tra ve it/khong ket qua vi
`metadata_rows_for_asr_hit()` loc theo [start_time, end_time] tren
timestamp_sec that - day la HANH VI DUNG (dang phan anh dung bug that), khong
phai loi cua mock. Sua/rebuild index truoc de test ASR day du.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))


def line(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# Mock Elasticsearch-backed repositories.
#
# Real `OCRRepository.search(text, top_k) -> list[dict]` and
# `ASRRepository.search(text, top_k) -> list[dict]` return raw ES hits shaped
# like `{"_score": float, "_source": {...}}` (see
# `ocr_search/pipeline/search.py::SearchPipeline.search` /
# `asr_search/pipeline/search.py::SearchPipeline.search`, which read exactly
# these two keys). We reproduce that shape here so the REAL SearchPipeline
# classes on top of it don't need any change to run.
# ---------------------------------------------------------------------------


class MockOCRRepository:
    """Fake Elasticsearch OCR index: naive substring match over a small canned corpus."""

    def __init__(self, corpus: list[dict[str, Any]]) -> None:
        # corpus item: {"dataset","video_id","keyframe_id","texts": [str,...]}
        self.corpus = corpus

    def search(self, text: str, top_k: int = 10) -> list[dict[str, Any]]:
        terms = [t for t in text.lower().split() if t]
        hits: list[dict[str, Any]] = []
        for item in self.corpus:
            haystack = " ".join(item["texts"]).lower()
            score = sum(haystack.count(t) for t in terms)
            if score <= 0:
                continue
            hits.append({"_score": float(score), "_source": item})
        hits.sort(key=lambda h: h["_score"], reverse=True)
        return hits[:top_k]


class MockASRRepository:
    """Fake Elasticsearch ASR index: naive substring match over canned transcript segments."""

    def __init__(self, corpus: list[dict[str, Any]]) -> None:
        # corpus item: {"dataset","video_id","start_time","end_time","text"}
        self.corpus = corpus

    def search(self, text: str, top_k: int = 10) -> list[dict[str, Any]]:
        terms = [t for t in text.lower().split() if t]
        hits: list[dict[str, Any]] = []
        for item in self.corpus:
            haystack = item["text"].lower()
            score = sum(haystack.count(t) for t in terms)
            if score <= 0:
                continue
            hits.append({"_score": float(score), "_source": item})
        hits.sort(key=lambda h: h["_score"], reverse=True)
        return hits[:top_k]


def build_mock_ocr_pipeline(metadata: pd.DataFrame, query_terms: list[str]):
    from src.retrieval.retriever.ocr_search.pipeline.search import SearchPipeline as OCRSearchPipeline

    sample = metadata.dropna(subset=["video_id", "keyframe_id"]).head(20)
    corpus = []
    for i, (_, row) in enumerate(sample.iterrows()):
        # Gan 1-2 tu khoa that vao vai dong dau de dam bao co it nhat vai hit
        # khop query cua ban (chu khong toan bo corpus deu random/vo nghia).
        planted = query_terms[i % len(query_terms)] if query_terms and i < 6 else "bien so xe mau xanh"
        corpus.append(
            {
                "dataset": row.get("dataset"),
                "video_id": row.get("video_id"),
                "keyframe_id": row.get("keyframe_id"),
                "texts": [planted, "chu chay tren man hinh"],
            }
        )
    return OCRSearchPipeline(MockOCRRepository(corpus))


def build_mock_asr_pipeline(metadata: pd.DataFrame, query_terms: list[str]):
    from src.retrieval.retriever.asr_search.pipeline.search import SearchPipeline as ASRSearchPipeline

    video_ids = metadata["video_id"].dropna().unique().tolist()[:5]
    corpus = []
    for i, vid in enumerate(video_ids):
        planted = query_terms[i % len(query_terms)] if query_terms else "noi dung mo ta"
        # Window rong (0 -> 10^6 giay) de KHOP moi keyframe cua video do bat
        # ke timestamp_sec that dang dung hay dang bi bug ve 0 (xem docstring
        # dau file) - muc tieu la test duoc pipeline fusion/enrich, khong
        # phai gia lap timing chuan xac cua ASR that.
        corpus.append(
            {
                "dataset": metadata.loc[metadata["video_id"] == vid, "dataset"].iloc[0],
                "video_id": vid,
                "start_time": 0.0,
                "end_time": 1_000_000.0,
                "text": f"{planted} - doan thoai mau cho video {vid}",
            }
        )
    return ASRSearchPipeline(MockASRRepository(corpus))


def show_df(label: str, df: pd.DataFrame, cols_priority: list[str]) -> None:
    print(f"\n--- {label} : {0 if df is None else len(df)} dong ---")
    if df is None or df.empty:
        print("  (rong)")
        return
    cols = [c for c in cols_priority if c in df.columns]
    extra = [c for c in ["search_mode", "rrf_score", "matched_sources", "source_models"] if c in df.columns and c not in cols]
    print(df[cols + extra].head(10).to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/app.yaml")
    parser.add_argument("--query", default="con hà mã đen; cảnh bảng đầy giấy note")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--duration", type=float, default=-1.0)
    parser.add_argument("--semantic-model", default=None, help="model_key cu the; mac dinh dung tat ca model available")
    parser.add_argument("--no-temporal", action="store_true")
    parser.add_argument("--no-ocr", action="store_true")
    parser.add_argument("--no-asr", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path

    line(f"[0] LOAD CONFIG + BUILD SYSTEM THAT: {config_path}")
    from src.api.legacy.paths import load_yaml
    from src.retrieval.system import build_system
    from src.retrieval.retriever.common.orchestrator import Orchestrator

    config = load_yaml(config_path)
    system = build_system(config)
    real_orch = system.orchestrator
    metadata: pd.DataFrame = real_orch.index.metadata

    semantic_models = [args.semantic_model] if args.semantic_model else real_orch.available_semantic_models()
    print(f"semantic_models dung de test = {semantic_models}")
    print(f"metadata: {len(metadata)} dong, {metadata['video_id'].nunique()} video")

    query_terms = [t for t in args.query.lower().split() if len(t) > 2]

    line("[1] DUNG MOCK OCR/ASR REPOSITORY (that su goi len tren SearchPipeline that)")
    mock_ocr_pipeline = None if args.no_ocr else build_mock_ocr_pipeline(metadata, query_terms)
    mock_asr_pipeline = None if args.no_asr else build_mock_asr_pipeline(metadata, query_terms)
    if mock_ocr_pipeline:
        print("OCR corpus mau (5 dong dau):")
        for c in mock_ocr_pipeline.repository.corpus[:5]:
            print(f"  video_id={c['video_id']} keyframe_id={c['keyframe_id']} texts={c['texts']}")
    if mock_asr_pipeline:
        print("ASR corpus mau:")
        for c in mock_asr_pipeline.repository.corpus:
            print(f"  video_id={c['video_id']} [{c['start_time']}, {c['end_time']}] text={c['text']!r}")

    line("[2] DUNG lai mot Orchestrator MOI: semantic/temporal THAT + ocr/asr MOCK")
    test_orch = Orchestrator(
        index=real_orch.index_manager or real_orch.index,
        semantic_search=real_orch.semantic_search,
        temporal_search=real_orch.temporal_search,
        ocr_search_pipeline=mock_ocr_pipeline,
        asr_search_pipeline=mock_asr_pipeline,
    )

    line(f"[3] KIEM TRA TUNG NGUON RIENG LE (truoc khi fuse) - query={args.query!r}")
    plan = test_orch.build_query_plan(args.query, mode="semantic", use_split=True)
    show_cols = ["video_id", "keyframe_id", "rank", "score", "retrieval_score", "search_mode"]

    for mk in semantic_models:
        df_sem = test_orch.semantic_search.search(plan.events, args.top_k, args.top_k * 5, model_key=mk)
        show_df(f"semantic[{mk}]", df_sem, show_cols)

    if not args.no_temporal:
        temporal_plan = test_orch.build_query_plan(args.query, mode="temporal", use_split=True)
        df_temp = test_orch.temporal_search.search(
            temporal_plan.events, args.top_k, args.top_k * 10, args.duration, model_key=semantic_models[0] if semantic_models else None
        )
        show_df("temporal", df_temp, ["video_id", "rank", "temporal_start_time", "temporal_end_time", "temporal_duration_sec"])

    if mock_ocr_pipeline:
        df_ocr = test_orch.ocr_search(args.query, args.top_k)
        show_df("ocr (mock)", df_ocr, show_cols + ["matched_texts"])

    if mock_asr_pipeline:
        df_asr = test_orch.asr_search(args.query, args.top_k)
        show_df("asr (mock)", df_asr, ["video_id", "rank", "temporal_start_time", "temporal_end_time", "matched_texts"])

    line("[4] ADVANCED_SEARCH THAT (fusion RRF toan bo nguon)")
    fused, per_source = test_orch.advanced_search(
        args.query,
        semantic_models=semantic_models,
        temporal=not args.no_temporal,
        use_ocr=not args.no_ocr,
        use_asr=not args.no_asr,
        top_k=args.top_k,
        duration_limit=args.duration,
    )

    print(f"\nper_source keys = {list(per_source.keys())}")
    for key, df in per_source.items():
        show_df(f"per_source[{key}]", df, show_cols)

    show_df("FUSED (ket qua cuoi cung tra ve frontend)", fused, show_cols)
    if not fused.empty and "search_mode" in fused.columns:
        print("\nPhan bo search_mode trong ket qua fuse cuoi:")
        print(fused["search_mode"].value_counts().to_string())
        print(
            "\n(Neu chi thay DUY NHAT 1 search_mode o day du ban da bat ca "
            "4 nguon -> fusion dang khong that su gop nguon nao khac, kiem "
            "tra lai weights/rrf_k hoac 1 nguon dang tra ve rong.)"
        )

    line("XONG - gui toan bo output nay lai de phan tich tiep")


if __name__ == "__main__":
    main()
