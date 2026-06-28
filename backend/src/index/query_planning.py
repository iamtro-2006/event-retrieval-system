from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd


SearchMode = Literal["semantic", "temporal", "ocr", "asr", "auto"]


def clean_queries(queries: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for query in queries:
        query = re.sub(r"\s+", " ", str(query or "").strip())
        key = query.casefold()
        if query and key not in seen:
            seen.add(key)
            cleaned.append(query)
    return cleaned


def split_temporal_events(query: str) -> list[str]:
    return clean_queries(re.split(r"[.;]+", str(query or "").replace("\n", " ")))


def split_semantic_queries(event: str) -> list[str]:
    event = str(event or "").strip()
    return (
        clean_queries([event, *(part.strip() for part in event.split(","))])
        if event
        else []
    )


@dataclass(frozen=True)
class QueryPlan:
    query: str
    mode: SearchMode
    use_split: bool
    events: list[list[str]]

    @property
    def event_queries(self) -> list[str]:
        return [event[0] for event in self.events if event]

    @property
    def flat_queries(self) -> list[str]:
        return clean_queries([query for event in self.events for query in event])


def build_query_plan(
    query: str, mode: SearchMode = "semantic", use_split: bool = True
) -> QueryPlan:
    query = str(query or "").strip()
    events = []
    for text in split_temporal_events(query):
        parts = split_semantic_queries(text) if use_split else clean_queries([text])
        if parts:
            events.append(parts)
    return QueryPlan(query=query, mode=mode, use_split=use_split, events=events)


def resolve_effective_mode(mode: SearchMode, plan: QueryPlan) -> SearchMode:
    if mode == "auto":
        return "temporal" if len(plan.events) > 1 else "semantic"
    return mode


def aggregate_multi_query(
    scores: np.ndarray,
    indices: np.ndarray,
    queries: list[str],
    top_k: int,
    metadata_records: list[dict] | None = None,
) -> pd.DataFrame:
    """Aggregate multi-query search results into a single ranked DataFrame.

    This is vector-store-agnostic: it operates on scores/indices arrays.
    When ``metadata_records`` is provided (list of dicts indexed by the IDs
    in ``indices``), each result row is enriched with the matching metadata.
    """
    if indices.size == 0:
        return pd.DataFrame()

    n_queries = len(queries)
    valid = indices >= 0
    flat_ids = indices[valid].astype(np.int32, copy=False)
    flat_scores = scores[valid].astype(np.float32, copy=False)
    query_ids = np.repeat(np.arange(n_queries, dtype=np.int32), indices.shape[1])[
        valid.ravel()
    ]

    order = np.argsort(flat_ids, kind="stable")
    ids_sorted, scores_sorted, q_sorted = (
        flat_ids[order],
        flat_scores[order],
        query_ids[order],
    )
    unique_ids, starts = np.unique(ids_sorted, return_index=True)
    counts = np.diff(np.r_[starts, len(ids_sorted)])
    score_sum = np.add.reduceat(scores_sorted, starts)
    max_score = np.maximum.reduceat(scores_sorted, starts)

    matched = np.empty(len(unique_ids), dtype=np.int32)
    for i, (start, count) in enumerate(zip(starts, counts)):
        matched[i] = np.unique(q_sorted[start : start + count]).size

    avg_score = score_sum / counts
    coverage = matched.astype(np.float32) / max(1, n_queries)
    alignment = 0.90 * avg_score + 0.10 * coverage

    rank_order = np.argsort(-alignment, kind="stable")[:top_k]

    rows: list[dict[str, Any]] = []
    for display_rank, pos in enumerate(rank_order, 1):
        idx = int(unique_ids[pos])
        if metadata_records is not None:
            item: dict[str, Any] = dict(metadata_records[idx])
        else:
            item = {"_id": idx}
        item["avg_score"] = float(avg_score[pos])
        item["max_score"] = float(max_score[pos])
        item["matched_queries"] = int(matched[pos])
        item["coverage_score"] = float(coverage[pos])
        item["alignment_score"] = float(alignment[pos])
        item["retrieval_score"] = float(alignment[pos])
        item["display_rank"] = display_rank
        item["rank"] = display_rank
        rows.append(item)

    return pd.DataFrame.from_records(rows)
