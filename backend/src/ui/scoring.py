from __future__ import annotations

import pandas as pd


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