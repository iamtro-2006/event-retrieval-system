"""Debug advanced fusion without Elasticsearch.

Reads the real local ASR/OCR post-process files, wraps them in in-memory
repositories with the same ``search()`` contract as Elasticsearch, then runs
the real Orchestrator.advanced_search() implementation.

Examples (from backend/):
  python scripts/debug_mock_fusion_data.py --query "sụt lún; cháy" --temporal
  python scripts/debug_mock_fusion_data.py --query "sụt lún; cháy" --temporal --ocr --no-asr
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parent
sys.path.insert(0, str(BACKEND))


def _tokens(text: str) -> list[str]:
    return [x for x in re.findall(r"\w+", str(text).casefold()) if len(x) > 1]


def _score(query: str, text: str) -> float:
    q = _tokens(query)
    hay = str(text).casefold()
    return float(sum(hay.count(token) for token in q))


class MockRepository:
    def __init__(self, documents: list[dict[str, Any]], text_field: str) -> None:
        self.documents = documents
        self.text_field = text_field

    def search(self, text: str, top_k: int = 10) -> list[dict[str, Any]]:
        hits = []
        for document in self.documents:
            value = document.get(self.text_field, "")
            if isinstance(value, list):
                value = " ".join(map(str, value))
            score = _score(text, value)
            if score > 0:
                hits.append({"_score": score, "_source": document})
        hits.sort(key=lambda item: item["_score"], reverse=True)
        return hits[: int(top_k)]


def load_asr(root: Path) -> list[dict[str, Any]]:
    documents = []
    for path in sorted((root / "data" / "asr").glob("*/*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[skip ASR] {path}: {exc}")
            continue
        for row in payload if isinstance(payload, list) else []:
            if not isinstance(row, dict):
                continue
            documents.append({
                "dataset": path.parent.name,
                "video_id": str(row.get("video_id") or path.stem),
                "start_time": float(row.get("start", row.get("start_time", 0))),
                "end_time": float(row.get("end", row.get("end_time", 0))),
                "text": str(row.get("transcript", row.get("text", ""))),
            })
    return documents


def load_ocr(root: Path) -> list[dict[str, Any]]:
    documents = []
    for path in sorted((root / "data" / "ocr_post").glob("*/*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[skip OCR] {path}: {exc}")
            continue
        for frame_id, value in payload.items() if isinstance(payload, dict) else []:
            texts: list[str] = []
            # OCR post files contain nested detection records; collect all
            # textual leaves and leave coordinates untouched/irrelevant.
            def collect(item: Any) -> None:
                if isinstance(item, str) and item.strip():
                    texts.append(item)
                elif isinstance(item, list):
                    for child in item:
                        collect(child)
                elif isinstance(item, dict):
                    for child in item.values():
                        collect(child)
            collect(value)
            if texts:
                documents.append({
                    "dataset": path.parent.name,
                    "video_id": path.stem,
                    "keyframe_id": str(frame_id),
                    "texts": texts,
                })
    return documents


def show(label: str, df: pd.DataFrame) -> None:
    print(f"\n[{label}] rows={0 if df is None else len(df)}")
    if df is None or df.empty:
        print("  <empty>")
        return
    cols = [
        "video_id", "keyframe_id", "timestamp_sec", "sub_query_idx",
        "search_mode", "rrf_score", "temporal_start_time",
        "temporal_end_time", "matched_sequence",
    ]
    cols = [column for column in cols if column in df.columns]
    view = df[cols].head(10).copy()
    if "matched_sequence" in view.columns:
        view["matched_sequence"] = view["matched_sequence"].map(
            lambda x: len(x) if isinstance(x, list) else x
        )
    print(view.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="sụt lún; cháy")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-multiplier", type=int, default=5)
    parser.add_argument("--temporal", action="store_true")
    parser.add_argument("--ocr", action="store_true")
    parser.add_argument("--no-asr", action="store_true")
    parser.add_argument("--no-ocr", action="store_true")
    args = parser.parse_args()

    from src.retrieval.retriever.asr_search.pipeline.search import SearchPipeline as ASRPipeline
    from src.retrieval.retriever.ocr_search.pipeline.search import SearchPipeline as OCRPipeline
    from src.retrieval.system import build_system
    from src.api.legacy.paths import load_yaml
    from src.retrieval.retriever.common.orchestrator import Orchestrator

    asr_docs = load_asr(ROOT)
    ocr_docs = load_ocr(ROOT)
    print(f"Loaded mock ASR={len(asr_docs)} segments, OCR={len(ocr_docs)} frames")

    config = load_yaml(BACKEND / "configs" / "app.yaml")
    real = build_system(config).orchestrator
    asr = None if args.no_asr else ASRPipeline(MockRepository(asr_docs, "text"))
    ocr = None if args.no_ocr else OCRPipeline(MockRepository(ocr_docs, "texts"))
    orchestrator = Orchestrator(
        index=real.index_manager or real.index,
        semantic_search=real.semantic_search,
        temporal_search=real.temporal_search,
        ocr_search_pipeline=ocr,
        asr_search_pipeline=asr,
    )

    models = orchestrator.available_semantic_models()
    print(f"Models={models}; temporal={args.temporal}; OCR={ocr is not None}; ASR={asr is not None}")
    result, sources = orchestrator.advanced_search(
        args.query,
        semantic_models=models[:1],
        temporal=args.temporal,
        use_ocr=ocr is not None,
        use_asr=asr is not None,
        top_k=args.top_k,
        candidate_multiplier=args.candidate_multiplier,
    )
    for label, frame in sources.items():
        show(label, frame)
    show("FINAL", result)


if __name__ == "__main__":
    main()
