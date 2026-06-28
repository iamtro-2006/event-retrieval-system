from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def format_keyframe_id(item: dict[str, Any]) -> str:
    """Format a keyframe identifier as a zero-padded 6-digit string.

    Accepts a dict (the canonical form used by both backends) and prefers
    ``keyframe_id`` (string), falling back to ``keyframe_id_int`` then
    ``frame_idx``.
    """
    value = item.get("keyframe_id", None)
    if value is not None:
        value_text = str(value)
        if value_text.isdigit():
            return value_text.zfill(6)
        return value_text

    if "keyframe_id_int" in item:
        return f"{safe_int(item.get('keyframe_id_int'), 0):06d}"
    if "frame_idx" in item:
        return f"{safe_int(item.get('frame_idx'), 0):06d}"
    return "000000"


def normalize_rel_path(path_value: str | Path) -> str:
    return str(path_value or "").replace("\\", "/").lstrip("/")


def resolve_keyframe_path(item: dict[str, Any]) -> str:
    keyframe_path = str(item.get("keyframe_path", "") or "")
    if keyframe_path:
        return keyframe_path.replace("\\", "/")

    dataset = str(item.get("dataset", "") or "")
    video_id = str(item.get("video_id", "") or "")
    frame_id_text = format_keyframe_id(item)

    if dataset and video_id:
        return f"data/processed/keyframes/{dataset}/{video_id}/{frame_id_text}.jpg"
    return f"data/processed/keyframes/{video_id}/{frame_id_text}.jpg"


def find_video_path(item: dict[str, Any]) -> str:
    video_path = str(item.get("video_path", "") or "")
    if video_path:
        return video_path.replace("\\", "/")

    dataset = str(item.get("dataset", "") or "")
    video_id = str(item.get("video_id", "") or "")

    if not video_id:
        return ""

    if dataset:
        return f"data/processed/videos/{dataset}/{video_id}.mp4"
    return f"data/processed/videos/{video_id}.mp4"


def serialize_matched_sequence(sequence: Any) -> list[dict[str, Any]]:
    if not isinstance(sequence, list):
        return []

    rows: list[dict[str, Any]] = []

    for idx, item in enumerate(sequence):
        if not isinstance(item, dict):
            continue

        raw_k_path = resolve_keyframe_path(item)
        if "keyframes/" in raw_k_path:
            image_rel_path = raw_k_path.split("keyframes/", 1)[1]
        else:
            dataset = str(item.get("dataset", "") or "")
            video_id = str(item.get("video_id", "") or "")
            frame_id_text = format_keyframe_id(item)
            image_rel_path = (
                f"{dataset}/{video_id}/{frame_id_text}.jpg"
                if dataset
                else f"{video_id}/{frame_id_text}.jpg"
            )

        frame_id_text = format_keyframe_id(item)
        timestamp = safe_float(
            item.get("timestamp_sec", item.get("timestamp", 0.0)), 0.0
        )
        score = safe_float(item.get("score", item.get("candidate_score", 0.0)), 0.0)

        rows.append(
            {
                "video_id": str(json_safe(item.get("video_id", "")) or ""),
                "source_name": str(json_safe(item.get("source_name", "")) or ""),
                "keyframe_id": frame_id_text,
                "frame_id": safe_int(frame_id_text, 0),
                "frame_idx": safe_int(item.get("frame_idx", 0), 0),
                "timestamp_sec": timestamp,
                "fps": safe_float(item.get("fps", 0), 0.0),
                "keyframe_path": raw_k_path,
                "image_url": f"/static/keyframes/{image_rel_path}"
                if image_rel_path
                else "",
                "image_rel_path": image_rel_path,
                "sub_query_idx": safe_int(item.get("sub_query_idx", idx), idx),
                "sub_query": str(json_safe(item.get("sub_query", "")) or ""),
                "score": score,
                "candidate_score": safe_float(
                    item.get("candidate_score", score), score
                ),
                "candidate_rank": safe_int(item.get("candidate_rank", 0), 0),
            }
        )

    return rows


def serialize(item: dict[str, Any]) -> dict[str, Any]:
    """Shape a retrieval hit dict into the HTTP response shape.

    This is the single source of truth for the response contract shared by
    the FastAPI app and the mock API. Pure: no I/O, no globals.
    """
    dataset = str(item.get("dataset", "") or "")
    video_id = str(item.get("video_id", "unknown_video") or "unknown_video")

    frame_id_text = format_keyframe_id(item)
    frame_id_number = safe_int(frame_id_text, 0)

    raw_k_path = resolve_keyframe_path(item)
    if "keyframes/" in raw_k_path:
        image_rel_path = raw_k_path.split("keyframes/", 1)[1]
    else:
        image_rel_path = (
            f"{dataset}/{video_id}/{frame_id_text}.jpg"
            if dataset
            else f"{video_id}/{frame_id_text}.jpg"
        )
    image_url = f"/static/keyframes/{image_rel_path}"

    raw_v_path = find_video_path(item)
    if "videos/" in raw_v_path:
        video_rel_path = raw_v_path.split("videos/", 1)[1]
    else:
        video_rel_path = f"{dataset}/{video_id}.mp4" if dataset else f"{video_id}.mp4"
    video_url = f"/static/videos/{video_rel_path}"

    map_rel_path = f"{dataset}/{video_id}.csv" if dataset else f"{video_id}.csv"
    map_url = f"/static/map-keyframes/{map_rel_path}"

    timestamp = safe_float(item.get("timestamp_sec", item.get("timestamp", 0.0)), 0.0)
    if timestamp == 0.0 and "pts_time" in item:
        timestamp = safe_float(item.get("pts_time"), 0.0)

    retrieval_score = safe_float(
        item.get(
            "retrieval_score",
            item.get("alignment_score", item.get("avg_score", item.get("score", 0.0))),
        ),
        0.0,
    )

    avg_score = safe_float(item.get("avg_score", retrieval_score), retrieval_score)
    matched_sequence = serialize_matched_sequence(item.get("matched_sequence", []))

    temporal_start = safe_float(item.get("temporal_start_time", timestamp), timestamp)
    temporal_end = safe_float(item.get("temporal_end_time", timestamp), timestamp)
    temporal_duration = safe_float(
        item.get("temporal_duration_sec", max(0.0, temporal_end - temporal_start)),
        max(0.0, temporal_end - temporal_start),
    )

    return {
        "id": f"{video_id}_{frame_id_text}",
        "video_id": video_id,
        "frame_id": frame_id_number,
        "frame_name": f"{frame_id_text}.jpg",
        "path": f"{video_id}/{frame_id_text}",
        "keyframe_path": raw_k_path,
        "image_url": image_url,
        "image_rel_path": image_rel_path,
        "video_url": video_url,
        "video_rel_path": video_rel_path,
        "map_url": map_url,
        "map_rel_path": map_rel_path,
        "timestamp": timestamp,
        "similarity": retrieval_score,
        "caption": str(item.get("caption", "") or ""),
        "rank": safe_int(item.get("display_rank", item.get("rank", 0)), 0),
        "matched_sequence": matched_sequence,
        "temporal": {
            "video_score": safe_float(item.get("video_score", 0), 0.0),
            "start_time": temporal_start,
            "end_time": temporal_end,
            "duration_sec": temporal_duration,
            "avg_score": avg_score,
        },
        "raw": item,
    }
