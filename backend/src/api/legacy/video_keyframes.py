"""Lightweight keyframe-strip contract shared by real and mock media APIs."""
from math import isfinite


def finite_number(value):
    try:
        number = float(value)
        return number if isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def build_video_keyframes(records, serialize):
    items = []
    seen = set()
    for record in records:
        timestamp = next((number for key in ("timestamp_sec", "timestamp", "time_sec", "pts_time")
                          if (number := finite_number(record.get(key))) is not None and number >= 0), None)
        fps = finite_number(record.get("fps"))
        frame_idx = finite_number(record.get("frame_idx"))
        if timestamp is None and fps and fps > 0 and frame_idx is not None and frame_idx >= 0:
            timestamp = frame_idx / fps
        if timestamp is None:
            continue
        source = serialize(record)
        identifier = record.get("keyframe_id")
        if identifier is None or str(identifier) in {"", "nan"}:
            identifier = record.get("keyframe_id_int", source.get("frame_id"))
        if identifier is None or str(identifier) in {"", "nan"} or str(identifier) in seen:
            continue
        seen.add(str(identifier))
        items.append({
            "keyframe_id": str(identifier),
            "frame_idx": int(frame_idx) if frame_idx is not None and frame_idx >= 0 else None,
            "timestamp": timestamp,
            "fps": fps if fps and fps > 0 else None,
            "image_url": source.get("image_url", ""),
            "image_rel_path": source.get("image_rel_path", ""),
            "synthetic_timing": bool(record.get("synthetic_timing", False)),
        })
    return sorted(items, key=lambda item: item["timestamp"])
