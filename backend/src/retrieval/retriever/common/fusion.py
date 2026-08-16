from __future__ import annotations

import numpy as np
import pandas as pd

# Columns that identity a keyframe across modalities (used as the merge key).
FUSION_KEY_COLUMNS = ("video_id", "keyframe_id_int")


def min_max_normalize(scores: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Min-max scale a mode's candidate scores to [0, 1].

    Spreads a mode's score distribution over the interval [0, 1] so that
    different modalities (cosine similarity vs BM25) become comparable before
    the weighted combination. A degenerate pool (all scores equal) collapses to
    1.0 rather than dividing by zero.
    """
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size == 0:
        return scores
    lo = float(np.min(scores))
    hi = float(np.max(scores))
    if hi - lo <= epsilon:
        return np.ones_like(scores, dtype=np.float32)
    return (scores - lo) / (hi - lo)


def reciprocal_rank_fusion(
    ranks_by_mode: dict[str, dict[tuple, int]], k: int = 60
) -> dict[tuple, float]:
    """Reciprocal Rank Fusion (RRF) across modes.

    Args:
        ranks_by_mode: mode -> {merge_key -> 1-based rank in that mode's pool}.
        k: RRF damping constant (60 is the standard value).

    Returns:
        merge_key -> summed RRF score across all modes that contain the key.
    """
    fused: dict[tuple, float] = {}
    for mode_ranks in ranks_by_mode.values():
        for key, rank in mode_ranks.items():
            fused[key] = fused.get(key, 0.0) + 1.0 / (k + rank)
    return fused


def fuse(
    semantic_df: pd.DataFrame,
    text_df: pd.DataFrame,
    weights: dict[str, float],
    rrf_weight: float = 0.1,
    rrf_k: int = 60,
    top_k: int = 20,
    min_max_epsilon: float = 1e-6,
) -> pd.DataFrame:
    """Fuse semantic + text candidate pools into a single ranked result set.

    Strategy (hybrid score + RRF):
      1. Normalize each mode's raw score over its own candidate pool
         (semantic -> `retrieval_score`, text -> raw `es_score` BM25).
      2. Weighted sum of the normalized per-mode scores.
      3. Add an RRF bonus for keyframes agreed on by multiple modalities.

    Keyframes are merged on `(video_id, keyframe_id_int)`. A keyframe present
    in both modes carries the semantic metadata plus the text provenance
    fields (`matched_texts`, `matched_ocr`, `matched_asr`, `es_score`); one
    present in a single mode is carried through unchanged.

    Args:
        semantic_df: semantic candidate pool (oversampled, not yet truncated).
        text_df: text candidate pool (oversampled, not yet truncated).
        weights: per-mode combination weights, e.g. {"semantic": 0.6, "text": 0.4}.
        rrf_weight: scale of the additive RRF consensus bonus.
        rrf_k: RRF damping constant.
        top_k: number of results to return.
        min_max_epsilon: floor for min-max normalization denominator.

    Returns:
        A ranked DataFrame with `retrieval_score` = fused score, plus
        `display_rank`/`rank` and `fusion_components` in each row.
    """
    available_modes: list[str] = []
    pools: list[tuple[str, pd.DataFrame, str]] = []

    if semantic_df is not None and not semantic_df.empty:
        available_modes.append("semantic")
        pools.append(("semantic", semantic_df, "retrieval_score"))
    if text_df is not None and not text_df.empty:
        available_modes.append("text")
        pools.append(("text", text_df, "es_score"))

    if not pools:
        return pd.DataFrame()

    # Renormalize weights over the modes that actually returned candidates, so
    # `similarity` remains meaningful when one arm is empty.
    total_weight = sum(weights.get(m, 0.0) for m in available_modes) or 1.0

    # ------------------------------------------------------------------
    # Normalize each mode's raw score over its own pool + build rank maps.
    # ------------------------------------------------------------------
    normalized: dict[str, pd.Series] = {}
    rank_maps: dict[str, dict[tuple, int]] = {}

    for mode, pool, score_col in pools:
        raw = pool[score_col].astype(float).to_numpy()
        pool = pool.reset_index(drop=True)
        norm = min_max_normalize(raw, min_max_epsilon)
        normalized[mode] = pd.Series(norm, index=pool.index)

        order = np.argsort(-raw, kind="stable")
        rank_map: dict[tuple, int] = {}
        for rank, pos in enumerate(order, 1):
            key = _key_for(pool, pos)
            if key is not None:
                rank_map.setdefault(key, rank)
        rank_maps[mode] = rank_map

    # ------------------------------------------------------------------
    # Merge all pools keyed on (video_id, keyframe_id_int).
    # ------------------------------------------------------------------
    merged: dict[tuple, dict] = {}
    source_order: dict[tuple, str] = {}
    for mode, pool, _score_col in pools:
        pool = pool.reset_index(drop=True)
        for pos in range(len(pool)):
            key = _key_for(pool, pos)
            if key is None:
                continue
            row = pool.iloc[pos].to_dict()
            if key not in merged:
                merged[key] = row
                merged[key]["fusion_components"] = {}
                source_order[key] = mode
            # Text provenance fields overwrite/merge onto the semantic row.
            for field in (
                "matched_texts",
                "matched_ocr",
                "matched_asr",
                "es_score",
                "ocr_score",
                "asr_score",
            ):
                if field in row and row[field] not in (None, "", []):
                    merged[key][field] = row[field]
            merged[key]["fusion_components"][mode] = {
                "score": float(row.get("retrieval_score", 0.0) or 0.0),
                "raw_score": float(
                    row.get(("es_score" if mode == "text" else "retrieval_score"), 0.0)
                    or 0.0
                ),
                "normalized": float(normalized[mode].iloc[pos]),
                "rank": rank_maps[mode].get(key),
            }

    # ------------------------------------------------------------------
    # Compute RRF + weighted combination.
    # ------------------------------------------------------------------
    rrf = reciprocal_rank_fusion(rank_maps, rrf_k)

    rows: list[dict] = []
    for key, item in merged.items():
        score_sum = 0.0
        for mode in available_modes:
            w = weights.get(mode, 0.0) / total_weight
            comp = item["fusion_components"].get(mode)
            if comp is not None:
                score_sum += w * comp["normalized"]
        fused = score_sum + rrf_weight * rrf.get(key, 0.0)
        item["retrieval_score"] = float(fused)
        item["score"] = float(fused)
        item["search_mode"] = "fusion"
        rows.append(item)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame.from_records(rows).sort_values("retrieval_score", ascending=False)
    df = df.head(int(top_k)).reset_index(drop=True)
    df["display_rank"] = np.arange(1, len(df) + 1)
    df["rank"] = df["display_rank"]
    return df


def _key_for(pool: pd.DataFrame, pos: int) -> tuple | None:
    video_id = pool.iloc[pos].get("video_id")
    keyframe_id_int = pool.iloc[pos].get("keyframe_id_int")
    if keyframe_id_int is None:
        keyframe_id_int = pool.iloc[pos].get("keyframe_id")
    if video_id is None or keyframe_id_int is None:
        return None
    try:
        return (str(video_id), int(keyframe_id_int))
    except (TypeError, ValueError):
        return None
