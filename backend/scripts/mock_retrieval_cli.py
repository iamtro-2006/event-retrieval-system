from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import dataclass
from typing import Any


MOCK_MODELS = ["siglip2-so400m", "vitH-378-quickgelu", "BLIP2"]

MOCK_FRAMES = [
    {
        "dataset": "L21",
        "video_id": "L21_V001",
        "keyframe_id": "000120",
        "frame_idx": 120,
        "timestamp_sec": 12.0,
        "keyframe_path": "data/keyframes/L21/L21_V001/000120.jpg",
        "caption": "a person walking near a red car in the street",
        "ocr_text": "bien bao duong pho",
        "asr_text": "hom nay chung ta noi ve giao thong do thi",
    },
    {
        "dataset": "L21",
        "video_id": "L21_V002",
        "keyframe_id": "000245",
        "frame_idx": 245,
        "timestamp_sec": 24.5,
        "keyframe_path": "data/keyframes/L21/L21_V002/000245.jpg",
        "caption": "people cooking food in a kitchen",
        "ocr_text": "nha hang mon ngon",
        "asr_text": "nguoi dau bep dang chuan bi bua an",
    },
    {
        "dataset": "L22",
        "video_id": "L22_V003",
        "keyframe_id": "000360",
        "frame_idx": 360,
        "timestamp_sec": 36.0,
        "keyframe_path": "data/keyframes/L22/L22_V003/000360.jpg",
        "caption": "a crowd watching a football match",
        "ocr_text": "tran dau bong da truc tiep",
        "asr_text": "khong khi san van dong rat soi dong",
    },
    {
        "dataset": "L22",
        "video_id": "L22_V004",
        "keyframe_id": "000480",
        "frame_idx": 480,
        "timestamp_sec": 48.0,
        "keyframe_path": "data/keyframes/L22/L22_V004/000480.jpg",
        "caption": "a train arrives at a busy station",
        "ocr_text": "ga tau trung tam",
        "asr_text": "chuyen tau tiep theo sap den san ga",
    },
]


@dataclass(frozen=True)
class QueryPlan:
    query: str
    mode: str
    use_split: bool
    events: list[list[str]]


def audit(step: str, **fields: Any) -> None:
    payload = " ".join(f"{key}={json.dumps(value, ensure_ascii=False)}" for key, value in fields.items())
    print(f"[AUDIT] {step}" + (f" | {payload}" if payload else ""))


def clean_parts(values: list[str]) -> list[str]:
    return [value.strip() for value in values if value and value.strip()]


def build_query_plan(query: str, mode: str, use_split: bool) -> QueryPlan:
    events: list[list[str]] = []
    for event in clean_parts(re.split(r"[.;]+", query.replace("\n", " "))):
        if use_split:
            parts = clean_parts([event, *event.split(",")])
        else:
            parts = [event]
        if parts:
            events.append(parts)
    return QueryPlan(query=query, mode=mode, use_split=use_split, events=events)


def token_score(query: str, text: str) -> float:
    query_tokens = set(re.findall(r"\w+", query.lower()))
    text_tokens = set(re.findall(r"\w+", text.lower()))
    if not query_tokens:
        return 0.0
    overlap = len(query_tokens & text_tokens)
    coverage = overlap / max(1, len(query_tokens))
    density = overlap / max(1, len(text_tokens))
    return round((coverage * 0.75) + (density * 0.25), 4)


def rank_rows(rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    rows = sorted(rows, key=lambda row: row["retrieval_score"], reverse=True)[:top_k]
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx
        row["display_rank"] = idx
    return rows


def base_result(frame: dict[str, Any], mode: str, score: float, model_key: str | None = None) -> dict[str, Any]:
    return {
        "dataset": frame["dataset"],
        "video_id": frame["video_id"],
        "keyframe_id": frame["keyframe_id"],
        "frame_idx": frame["frame_idx"],
        "timestamp_sec": frame["timestamp_sec"],
        "keyframe_path": frame["keyframe_path"],
        "retrieval_score": score,
        "search_mode": mode,
        "model_key": model_key,
    }


def semantic_search(query: str, model_key: str, top_k: int) -> list[dict[str, Any]]:
    audit("semantic.encode_text", model_key=model_key, query=query, model_loaded=False)
    rows = []
    model_bias = (MOCK_MODELS.index(model_key) + 1) * 0.015 if model_key in MOCK_MODELS else 0.0
    for frame in MOCK_FRAMES:
        score = min(1.0, token_score(query, frame["caption"]) + model_bias)
        rows.append(base_result(frame, "semantic", score, model_key))
    audit("semantic.faiss_search", model_key=model_key, candidates=len(rows))
    return rank_rows(rows, top_k)


def temporal_search(plan: QueryPlan, model_key: str, top_k: int, duration_limit: float) -> list[dict[str, Any]]:
    audit("temporal.plan", events=plan.events, model_key=model_key, duration_limit=duration_limit)
    event_queries = [event[0] for event in plan.events]
    rows = []
    for frame in MOCK_FRAMES:
        event_scores = [token_score(event, frame["caption"]) for event in event_queries]
        avg_score = sum(event_scores) / max(1, len(event_scores))
        if len(event_queries) > 1:
            avg_score += 0.08
        matched_sequence = []
        for idx, event in enumerate(event_queries):
            seq_frame = dict(frame)
            seq_frame["sub_query_idx"] = idx
            seq_frame["sub_query"] = event
            seq_frame["candidate_score"] = round(avg_score, 4)
            matched_sequence.append(seq_frame)
        row = base_result(frame, "temporal", round(min(1.0, avg_score), 4), model_key)
        row.update(
            matched_sequence=matched_sequence,
            temporal_start_time=frame["timestamp_sec"],
            temporal_end_time=frame["timestamp_sec"] + max(1.0, len(event_queries) * 2.5),
        )
        rows.append(row)
    audit("temporal.dp_align", candidates=len(rows))
    return rank_rows(rows, top_k)


def ocr_search(query: str, top_k: int) -> list[dict[str, Any]]:
    audit("ocr.elasticsearch_query", index="ocr_db", query=query)
    rows = []
    for frame in MOCK_FRAMES:
        score = token_score(query, frame["ocr_text"])
        row = base_result(frame, "ocr", score)
        row["matched_texts"] = [frame["ocr_text"]]
        row["ocr_score"] = round(score * 10, 4)
        rows.append(row)
    return rank_rows(rows, top_k)


def asr_search(query: str, top_k: int) -> list[dict[str, Any]]:
    audit("asr.elasticsearch_query", index="asr_db", query=query)
    rows = []
    for frame in MOCK_FRAMES:
        score = token_score(query, frame["asr_text"])
        row = base_result(frame, "asr", score)
        row["matched_texts"] = [frame["asr_text"]]
        row["asr_score"] = round(score * 10, 4)
        row["matched_sequence"] = [
            {**frame, "sub_query_idx": 0, "sub_query": frame["asr_text"], "candidate_score": score}
        ]
        rows.append(row)
    return rank_rows(rows, top_k)


def reciprocal_rank_fusion(sources: dict[str, list[dict[str, Any]]], top_k: int, rrf_k: int = 60) -> list[dict[str, Any]]:
    audit("fusion.rrf", sources=list(sources), rrf_k=rrf_k)
    fused: dict[tuple[str, str], dict[str, Any]] = {}
    for source, rows in sources.items():
        for rank, row in enumerate(rows, start=1):
            key = (row["video_id"], row["keyframe_id"])
            item = fused.setdefault(key, dict(row, source_models=[]))
            item["source_models"].append(source)
            item["rrf_score"] = item.get("rrf_score", 0.0) + 1.0 / (rrf_k + rank)
    output = list(fused.values())
    for item in output:
        item["retrieval_score"] = round(item["rrf_score"], 6)
    return rank_rows(output, top_k)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    query = args.query.strip()
    if not query:
        raise ValueError("query must not be empty")

    mode = args.mode
    plan = build_query_plan(query, mode, args.use_split)
    audit("request.accepted", mode=mode, top_k=args.top_k, use_split=args.use_split)
    audit("query.plan_built", events=plan.events)

    if mode == "auto":
        mode = "temporal" if len(plan.events) > 1 else "semantic"
        audit("auto.resolved_mode", effective_mode=mode)

    if mode == "semantic":
        results = semantic_search(query, args.model_key, args.top_k)
        per_source = None
    elif mode == "temporal":
        results = temporal_search(plan, args.model_key, args.top_k, args.duration_limit)
        per_source = None
    elif mode == "ocr":
        results = ocr_search(query, args.top_k)
        per_source = None
    elif mode == "asr":
        results = asr_search(query, args.top_k)
        per_source = None
    elif mode == "advanced":
        sources: dict[str, list[dict[str, Any]]] = {}
        for model_key in args.semantic_models:
            sources[f"semantic:{model_key}"] = semantic_search(query, model_key, args.top_k)
        if args.temporal and args.semantic_models:
            sources["temporal"] = temporal_search(plan, args.semantic_models[0], args.top_k, args.duration_limit)
        if args.use_ocr:
            sources["ocr"] = ocr_search(query, args.top_k)
        if args.use_asr:
            sources["asr"] = asr_search(query, args.top_k)
        results = reciprocal_rank_fusion(sources, args.top_k)
        per_source = sources if args.include_per_source else None
    else:
        raise ValueError(f"unsupported mode: {mode}")

    latency_ms = math.ceil((time.perf_counter() - started) * 1000)
    audit("response.ready", count=len(results), latency_ms=latency_ms)
    return {
        "query": query,
        "mode": args.mode,
        "effective_mode": mode,
        "latency_ms": latency_ms,
        "query_plan": {"mode": plan.mode, "use_split": plan.use_split, "events": plan.events},
        "count": len(results),
        "results": results,
        "per_source": per_source,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mock retrieval CLI: no model loading, audit logs only.")
    parser.add_argument("query", help="Search query.")
    parser.add_argument(
        "--mode",
        choices=["semantic", "temporal", "ocr", "asr", "auto", "advanced"],
        default="auto",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--use-split", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model-key", default=MOCK_MODELS[0], choices=MOCK_MODELS)
    parser.add_argument("--semantic-models", nargs="*", default=[MOCK_MODELS[0]])
    parser.add_argument("--temporal", action="store_true")
    parser.add_argument("--use-ocr", action="store_true")
    parser.add_argument("--use-asr", action="store_true")
    parser.add_argument("--include-per-source", action="store_true")
    parser.add_argument("--duration-limit", type=float, default=-1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    response = run(args)
    print(json.dumps(response, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
