"""Serialize metadata rows (DataFrame records) thành response dict cho
frontend cũ — port nguyên `dict_to_result_FAST` + các hàm phụ trợ từ
`main.py` gốc, KHÔNG đổi shape output (frontend đang render trực tiếp field
`image_url`, `video_url`, `map_url`, `matched_sequence`, `temporal`, ...).

Khác biệt duy nhất so với bản gốc: `KEYFRAMES_ROOT` không còn là biến global
ở top-level module (vì giờ nhiều router dùng chung code này, và root path
chỉ có sau khi config được load trong lifespan) — được truyền vào tường minh
qua tham số `keyframes_root` thay vì đọc closure global.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import HTTPException

from src.api.legacy.paths import resolve_backend_path
from src.retrieval.index.faiss_index import FaissIndex


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert a value to float, handling NaNs and exceptions efficiently."""
    if value is None:
        return default
    if isinstance(value, float) and np.isnan(value):
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert a value to int, handling NaNs and exceptions efficiently."""
    if value is None:
        return default
    if isinstance(value, float) and np.isnan(value):
        return default
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default


def json_safe(value: Any) -> Any:
    """Convert numpy/pandas types to native Python types for JSON serialization.

    Recurses into dict/list/tuple/set so nested structures (e.g. the raw
    metadata row echoed back under `"raw"` in `dict_to_result_FAST`) are
    fully sanitized too — a single un-sanitized `nan`/`inf` anywhere in the
    tree makes the whole response fail to serialize (`ValueError: Out of
    range float values are not JSON compliant: nan`).
    """
    if value is None:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        f = float(value)
        return None if (np.isnan(f) or np.isinf(f)) else f
    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def format_keyframe_id_from_dict(item: dict[str, Any]) -> str:
    """Extract and format the keyframe ID as a 6-digit zero-padded string."""
    value = item.get("keyframe_id")
    if value is not None:
        value_text = str(value)
        return value_text.zfill(6) if value_text.isdigit() else value_text

    if "keyframe_id_int" in item:
        return f"{safe_int(item.get('keyframe_id_int'), 0):06d}"
    if "frame_idx" in item:
        return f"{safe_int(item.get('frame_idx'), 0):06d}"
    return "000000"


def resolve_keyframe_path_from_dict(item: dict[str, Any], keyframes_root: Path, backend_dir: Path) -> str:
    """Resolve the absolute path for a keyframe image.

    Always prefer rebuilding the path from the *current* `keyframes_root`
    using dataset/video_id/frame_id. The `keyframe_path` column in
    metadata.csv can contain a stale absolute path baked in at index-build
    time (e.g. pointing at an old data location before a folder move), so
    it is only used as a last-resort fallback when dataset/video_id are
    missing.
    """
    dataset = str(item.get("dataset", "") or "")
    video_id = str(item.get("video_id", "") or "")
    frame_id_text = format_keyframe_id_from_dict(item)

    if dataset and video_id:
        return str(keyframes_root / dataset / video_id / f"{frame_id_text}.jpg")
    if video_id:
        return str(keyframes_root / video_id / f"{frame_id_text}.jpg")

    keyframe_path = str(item.get("keyframe_path", "") or "")
    if keyframe_path:
        keyframe_path = keyframe_path.replace("\\", "/")
        resolved = resolve_backend_path(backend_dir, keyframe_path)
        if resolved.exists():
            return str(resolved)
        # Stale absolute path from an old data location: retry against the
        # current keyframes root using just the filename tail.
        return str(keyframes_root / Path(keyframe_path).name)

    return ""


def find_video_path_from_dict(item: dict[str, Any]) -> str:
    """Resolve the relative path for a video file."""
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


def serialize_matched_sequence(
    sequence: list[dict[str, Any]] | Any, keyframes_root: Path, backend_dir: Path
) -> list[dict[str, Any]]:
    """Serialize a list of matched temporal sequence frames.

    `sequence` comes straight from a DataFrame record (`get("matched_sequence", [])`),
    so it is NOT guaranteed to be a list: when rows from a `matched_sequence`-less
    source (semantic/ocr/asr) are merged with temporal rows via
    `pd.DataFrame.from_records()` (see `common/scoring.reciprocal_rank_fusion`),
    pandas fills the missing cells with `NaN` (a `float`) instead of leaving the
    key absent — so `.get(..., [])`'s default never kicks in. `bool(nan)` is
    `True`, so `if not sequence` alone does NOT catch this and `enumerate(nan)`
    used to raise `TypeError: 'float' object is not iterable`. Explicitly type-check
    first.
    """
    if not isinstance(sequence, (list, tuple)):
        return []
    if not sequence:
        return []

    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(sequence):
        if not isinstance(item, dict):
            continue

        get = item.get  # Localize method lookup for C-speed access
        raw_k_path = resolve_keyframe_path_from_dict(item, keyframes_root, backend_dir)

        if "keyframes/" in raw_k_path:
            image_rel_path = raw_k_path.split("keyframes/", 1)[1]
        else:
            dataset = str(get("dataset", "") or "")
            video_id = str(get("video_id", "") or "")
            frame_id_text = format_keyframe_id_from_dict(item)
            image_rel_path = f"{dataset}/{video_id}/{frame_id_text}.jpg" if dataset else f"{video_id}/{frame_id_text}.jpg"

        frame_id_text = format_keyframe_id_from_dict(item)
        timestamp = safe_float(get("timestamp_sec", get("timestamp", 0.0)), 0.0)
        score = safe_float(get("score", get("candidate_score", 0.0)), 0.0)

        rows.append({
            "video_id": str(json_safe(get("video_id", "")) or ""),
            "source_name": str(json_safe(get("source_name", "")) or ""),
            "keyframe_id": frame_id_text,
            "frame_id": safe_int(frame_id_text, 0),
            "frame_idx": safe_int(get("frame_idx", 0), 0),
            "timestamp_sec": timestamp,
            "fps": safe_float(get("fps", 0), 0.0),
            "keyframe_path": raw_k_path,
            "image_url": f"/static/keyframes/{image_rel_path}" if image_rel_path else "",
            "image_rel_path": image_rel_path,
            "sub_query_idx": safe_int(get("sub_query_idx", idx), idx),
            "sub_query": str(json_safe(get("sub_query", "")) or ""),
            "score": score,
            "candidate_score": safe_float(get("candidate_score", score), score),
            "candidate_rank": safe_int(get("candidate_rank", 0), 0),
        })
    return rows


def dict_to_result_FAST(item: dict[str, Any], keyframes_root: Path, backend_dir: Path) -> dict[str, Any]:
    """High-performance serialization of a metadata row to an API response dictionary."""
    get = item.get  # Localize method lookup for C-speed access

    dataset = str(get("dataset", "") or "")
    video_id = str(get("video_id", "unknown_video") or "unknown_video")
    frame_id_text = format_keyframe_id_from_dict(item)
    frame_id_number = safe_int(frame_id_text, 0)

    raw_k_path = resolve_keyframe_path_from_dict(item, keyframes_root, backend_dir)
    if "keyframes/" in raw_k_path:
        image_rel_path = raw_k_path.split("keyframes/", 1)[1]
    else:
        image_rel_path = f"{dataset}/{video_id}/{frame_id_text}.jpg" if dataset else f"{video_id}/{frame_id_text}.jpg"

    raw_v_path = find_video_path_from_dict(item)
    if "videos/" in raw_v_path:
        video_rel_path = raw_v_path.split("videos/", 1)[1]
    else:
        video_rel_path = f"{dataset}/{video_id}.mp4" if dataset else f"{video_id}.mp4"

    timestamp = safe_float(get("timestamp_sec", get("timestamp", 0.0)), 0.0)
    if timestamp == 0.0 and "pts_time" in item:
        timestamp = safe_float(get("pts_time"), 0.0)

    retrieval_score = safe_float(
        get("retrieval_score", get("alignment_score", get("avg_score", get("score", 0.0)))), 0.0
    )
    avg_score = safe_float(get("avg_score", retrieval_score), retrieval_score)

    temporal_start = safe_float(get("temporal_start_time", timestamp), timestamp)
    temporal_end = safe_float(get("temporal_end_time", timestamp), timestamp)

    # OCR-only field: on-screen text strings that matched the query (absent/empty
    # for semantic/temporal results). Kept at top-level, like `caption`, so the
    # frontend can render OCR hits with the exact same result card component.
    matched_texts_raw = get("matched_texts")
    matched_texts = [str(t) for t in matched_texts_raw] if isinstance(matched_texts_raw, (list, tuple)) else []

    return {
        "id": f"{video_id}_{frame_id_text}",
        "video_id": video_id,
        "frame_id": frame_id_number,
        "frame_name": f"{frame_id_text}.jpg",
        "path": f"{video_id}/{frame_id_text}",
        "keyframe_path": raw_k_path,
        "image_url": f"/static/keyframes/{image_rel_path}",
        "image_rel_path": image_rel_path,
        "video_url": f"/static/videos/{video_rel_path}",
        "video_rel_path": video_rel_path,
        "map_url": f"/static/map-keyframes/{dataset}/{video_id}.csv" if dataset else f"/static/map-keyframes/{video_id}.csv",
        "map_rel_path": f"{dataset}/{video_id}.csv" if dataset else f"{video_id}.csv",
        "timestamp": timestamp,
        "similarity": retrieval_score,
        "caption": str(get("caption", "") or ""),
        "rank": safe_int(get("display_rank", get("rank", 0)), 0),
        "matched_sequence": serialize_matched_sequence(get("matched_sequence", []), keyframes_root, backend_dir),
        "matched_texts": matched_texts,
        "ocr_score": safe_float(get("ocr_score"), 0.0) if get("ocr_score") is not None else None,
        "asr_score": safe_float(get("asr_score"), 0.0) if get("asr_score") is not None else None,
        "temporal": {
            "video_score": safe_float(get("video_score", 0), 0.0),
            "start_time": temporal_start,
            "end_time": temporal_end,
            "duration_sec": safe_float(get("temporal_duration_sec", max(0.0, temporal_end - temporal_start)), max(0.0, temporal_end - temporal_start)),
            "avg_score": avg_score,
        },
        "raw": json_safe(item),
    }


def find_metadata_row(clip_index: FaissIndex, video_id: str, keyframe_id: int) -> pd.Series:
    """Fetch a specific metadata row using the pre-computed hash map."""
    row_idx = clip_index._row_by_video_frame.get((str(video_id), int(keyframe_id)))
    if row_idx is None:
        raise HTTPException(status_code=404, detail=f"Frame not found: {video_id}/{keyframe_id}")
    return clip_index.metadata.iloc[int(row_idx)]