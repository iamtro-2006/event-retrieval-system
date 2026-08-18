"""Step-by-step audit for fusion identity, translation paths, RRF and temporal.

Run from backend/:
  python scripts/debug_fusion_steps.py --query "sụt lún; cháy" --temporal
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parent
sys.path.insert(0, str(BACKEND))

from scripts.debug_mock_fusion_data import MockRepository, load_asr, load_ocr  # noqa: E402
from src.api.legacy.paths import load_yaml  # noqa: E402
from src.retrieval.retriever.asr_search.pipeline.search import SearchPipeline as ASRPipeline  # noqa: E402
from src.retrieval.retriever.common.orchestrator import Orchestrator  # noqa: E402
from src.retrieval.retriever.common.scoring import reciprocal_rank_fusion  # noqa: E402
from src.retrieval.retriever.ocr_search.pipeline.search import SearchPipeline as OCRPipeline  # noqa: E402
from src.retrieval.system import build_system  # noqa: E402


def flatten(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, parent in df.iterrows():
        seq = parent.get("matched_sequence")
        if not isinstance(seq, (list, tuple)) or not seq:
            rows.append(parent.to_dict())
            continue
        for frame in seq:
            if isinstance(frame, dict):
                row = {**parent.to_dict(), **frame}
                row.pop("matched_sequence", None)
                row.pop("sub_query_idx", None)
                rows.append(row)
    out = pd.DataFrame.from_records(rows)
    if not out.empty:
        out["rank"] = range(1, len(out) + 1)
    return out


def identity(row: pd.Series) -> tuple[str, str, str]:
    dataset = str(row.get("dataset", "")).strip().casefold()
    video = str(row.get("video_id", "")).strip().casefold()
    try:
        keyframe = str(int(float(row.get("keyframe_id"))))
    except (TypeError, ValueError):
        keyframe = str(row.get("keyframe_id", "")).strip().casefold()
    return dataset, video, keyframe


def report(label: str, df: pd.DataFrame) -> None:
    print(f"\n[{label}] rows={len(df)}")
    if df.empty:
        return
    print(df[[c for c in ["video_id", "keyframe_id", "timestamp_sec", "rank"] if c in df]].head(8).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="sụt lún; cháy")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--temporal", action="store_true")
    args = parser.parse_args()

    real = build_system(load_yaml(BACKEND / "configs" / "app.yaml")).orchestrator
    ocr = OCRPipeline(MockRepository(load_ocr(ROOT), "texts"))
    asr = ASRPipeline(MockRepository(load_asr(ROOT), "text"))
    orch = Orchestrator(real.index_manager or real.index, real.semantic_search, real.temporal_search, ocr, asr)
    model = orch.available_semantic_models()[0]
    plan = orch.build_query_plan(args.query, mode="semantic", use_split=True)
    print(f"QUERY RAW={args.query!r}")
    print(f"EVENTS={plan.event_queries}")
    print(f"MODEL={model}; TEMPORAL={args.temporal}")

    for event_idx, event in enumerate(plan.events):
        semantic = orch.semantic_search.search([event], args.top_k, args.top_k * 5, model_key=model)
        ocr_df = orch.ocr_search(event[0], args.top_k * 5)
        asr_df = orch.asr_search(event[0], args.top_k * 5)
        sources = [flatten(semantic), flatten(ocr_df), flatten(asr_df)]
        for name, frame_df in zip(("semantic", "ocr", "asr"), sources):
            frame_df["rank"] = range(1, len(frame_df) + 1)
            report(f"event={event_idx} {name} frames", frame_df)
        all_ids = {}
        for name, frame_df in zip(("semantic", "ocr", "asr"), sources):
            for _, row in frame_df.iterrows():
                all_ids.setdefault(identity(row), []).append(name)
        collisions = {key: sorted(set(methods)) for key, methods in all_ids.items() if len(set(methods)) > 1}
        print(f"event={event_idx} CROSS-METHOD FRAME COLLISIONS={len(collisions)}")
        for key, methods in list(collisions.items())[:10]:
            print(f"  {key} <- {methods}")
        fused = reciprocal_rank_fusion(sources, top_k=args.top_k * 5)
        report(f"event={event_idx} RRF frames", fused)

    final, _ = orch.advanced_search(args.query, semantic_models=[model], temporal=args.temporal,
                                    use_ocr=True, use_asr=True, top_k=args.top_k)
    report("FINAL", final)
    if not final.empty and "matched_sequence" in final.columns:
        print("FINAL sequence lengths:", final["matched_sequence"].map(lambda x: len(x) if isinstance(x, list) else 0).tolist())


if __name__ == "__main__":
    main()
