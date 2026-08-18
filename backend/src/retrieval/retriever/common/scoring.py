from __future__ import annotations

import pandas as pd

# Identity of a "result" across different retrievers/models: same keyframe,
# regardless of which model/method found it. Used by `reciprocal_rank_fusion`
# to merge ranked lists coming from different semantic models and/or
# semantic/temporal/ocr/asr methods (`advanced_search`).
_IDENTITY_COLUMNS = ("dataset", "video_id", "keyframe_id")


def _canonical_identity_value(column: str, value) -> str:
    """Canonicalize IDs so heterogeneous backends identify the same frame.

    FAISS metadata commonly exposes numeric keyframe IDs while Elasticsearch
    returns strings (sometimes zero-padded). Without canonicalization, 291,
    "291", and "000291" incorrectly become three different RRF entities.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if column == "keyframe_id":
        try:
            return str(int(float(text)))
        except (TypeError, ValueError):
            pass
    return text.casefold()


def reciprocal_rank_fusion(
    ranked_lists: list[pd.DataFrame],
    weights: list[float] | None = None,
    rrf_k: int = 60,
    top_k: int | None = None,
) -> pd.DataFrame:
    """Merge several already-ranked result DataFrames (each sorted best-first,
    with a `rank`/`display_rank` column, or just row order) into ONE ranked
    DataFrame using weighted Reciprocal Rank Fusion.

    This is what `advanced_search` uses to combine results coming from
    several different semantic models and/or several search methods
    (semantic/temporal/ocr/asr) that ticked "on" for the request — each list
    can be on a totally different, non-comparable raw score scale (cosine
    sim vs BM25 vs DP alignment score), so raw scores are never averaged
    directly; only each list's *rank order* is used.

    RRF formula per source list: `score = weight / (rrf_k + rank)`, summed
    across every list a given result appears in (higher = better). This is a
    standard, parameter-light way to fuse heterogeneous rankers (used by
    Elasticsearch, most hybrid-search systems, etc.) — a result that ranks
    highly in *several* lists outranks one that ranks #1 in only one.

    Args:
        ranked_lists: One DataFrame per source (model and/or method), each
            already sorted best-first (row order = rank; a `rank` column is
            used instead if present).
        weights: Optional per-list weight (same length as `ranked_lists`,
            default 1.0 each) — e.g. give semantic more weight than OCR.
        rrf_k: RRF damping constant (60 is the standard default from the
            original RRF paper / most implementations).
        top_k: Optional cap on the number of fused rows returned.

    Returns:
        A single DataFrame sorted by fused `rrf_score` descending, with a
        `source_models` column listing which model_key/search_mode(s)
        contributed each row and a `matched_sources` count.
    """
    if not ranked_lists:
        return pd.DataFrame()

    weights = weights or [1.0] * len(ranked_lists)
    fused: dict[tuple, dict] = {}

    for df, weight in zip(ranked_lists, weights):
        if df is None or df.empty:
            continue
        identity_cols = [c for c in _IDENTITY_COLUMNS if c in df.columns]
        if not identity_cols:
            continue

        ranks = df["rank"] if "rank" in df.columns else (pd.Series(range(1, len(df) + 1), index=df.index))
        source_label = df["search_mode"].iloc[0] if "search_mode" in df.columns else None
        model_label = df["model_key"].iloc[0] if "model_key" in df.columns else None
        label = " / ".join(str(x) for x in (source_label, model_label) if x) or "unknown"

        for (_, row), rank in zip(df.iterrows(), ranks):
            key = tuple(_canonical_identity_value(c, row.get(c)) for c in identity_cols)
            contribution = float(weight) / (rrf_k + float(rank))
            if key not in fused:
                fused[key] = {
                    "row": row.to_dict(),
                    "rrf_score": 0.0,
                    "source_labels": [],
                }
            fused[key]["rrf_score"] += contribution
            fused[key]["source_labels"].append(label)

    if not fused:
        return pd.DataFrame()

    rows = []
    for entry in fused.values():
        item = dict(entry["row"])
        item["rrf_score"] = entry["rrf_score"]
        item["matched_sources"] = len(entry["source_labels"])
        item["source_models"] = sorted(set(entry["source_labels"]))
        rows.append(item)

    result = pd.DataFrame.from_records(rows).sort_values("rrf_score", ascending=False).reset_index(drop=True)
    if top_k is not None:
        result = result.head(int(top_k)).reset_index(drop=True)
    result["display_rank"] = range(1, len(result) + 1)
    result["rank"] = result["display_rank"]
    return result


def rerank_multi_query(results: list[pd.DataFrame]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()

    rows = []

    for query_idx, df in enumerate(results):
        if df.empty:
            continue

        for _, row in df.iterrows():
            item = row.to_dict()
            item["sub_query_idx"] = query_idx
            rows.append(item)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    group_cols = [
        "dataset",
        "video_id",
        "keyframe_id",
        "source_name",
        "frame_idx",
        "timestamp_sec",
        "fps",
        "keyframe_path",
        "embedding_path",
        "map_path",
    ]

    existing_group_cols = [c for c in group_cols if c in df.columns]
    n_queries = max(1, df["sub_query_idx"].nunique())

    ranked = (
        df.groupby(existing_group_cols, as_index=False)
        .agg(
            avg_score=("score", "mean"),
            max_score=("score", "max"),
            matched_queries=("sub_query_idx", "nunique"),
        )
    )

    ranked["coverage_score"] = ranked["matched_queries"] / n_queries

    ranked["alignment_score"] = (
        0.90 * ranked["avg_score"]
        + 0.10 * ranked["coverage_score"]
    )

    return ranked.sort_values("alignment_score", ascending=False).reset_index(drop=True)
