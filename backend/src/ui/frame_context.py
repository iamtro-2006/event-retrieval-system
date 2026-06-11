from __future__ import annotations

from pathlib import Path


IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]


def get_surrounding_frames(keyframe_path: str | Path, radius: int = 5) -> list[dict]:
    keyframe_path = Path(keyframe_path)

    if not keyframe_path.exists():
        return []

    folder = keyframe_path.parent
    stem = keyframe_path.stem

    try:
        current_id = int(stem)
    except ValueError:
        return [{"keyframe_id": stem, "path": keyframe_path}]

    frames = []

    for idx in range(current_id - radius, current_id + radius + 1):
        if idx < 0:
            continue

        for ext in IMAGE_EXTENSIONS:
            path = folder / f"{idx:06d}{ext}"

            if path.exists():
                frames.append({
                    "keyframe_id": f"{idx:06d}",
                    "path": path,
                    "is_current": idx == current_id,
                })
                break

    return frames


def get_timestamp_from_row(row) -> float | None:
    if "timestamp_sec" in row and row["timestamp_sec"] == row["timestamp_sec"]:
        return float(row["timestamp_sec"])

    if (
        "frame_idx" in row
        and "fps" in row
        and row["frame_idx"] == row["frame_idx"]
        and row["fps"] == row["fps"]
        and float(row["fps"]) > 0
    ):
        return float(row["frame_idx"]) / float(row["fps"])

    return None