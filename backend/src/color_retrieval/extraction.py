from __future__ import annotations

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from .palette import palette_indices

GRID_SIZE = 5


def frame_identity(row: dict) -> str:
    frame_id = row.get("keyframe_id_int", row.get("keyframe_id", row.get("frame_idx", "")))
    try:
        frame_id = str(int(float(frame_id)))
    except (TypeError, ValueError):
        frame_id = str(frame_id)
    return f"{row.get('video_id', '')}\0{frame_id}"


def extract_grid(image: Image.Image, grid_size: int = GRID_SIZE) -> np.ndarray:
    # BOX keeps regional colour proportions while a fixed divisible size
    # bounds CPU cost for large source images.
    sample_size = grid_size * 10
    sampled = ImageOps.exif_transpose(image).convert("RGB").resize((sample_size, sample_size), Image.Resampling.BOX)
    labels = palette_indices(np.asarray(sampled))
    cell_size = sample_size // grid_size
    return np.asarray([
        np.bincount(
            labels[row * cell_size:(row + 1) * cell_size, col * cell_size:(col + 1) * cell_size].ravel(),
            minlength=11,
        ).argmax()
        for row in range(grid_size) for col in range(grid_size)
    ], dtype=np.uint8)


def resolve_frame_path(row: dict, metadata_path: Path, keyframes_root: Path | None = None) -> Path:
    raw = str(row.get("keyframe_path") or row.get("path") or "").strip()
    candidate = Path(raw)
    choices = [candidate]
    if not candidate.is_absolute():
        choices.extend([metadata_path.parent / candidate, Path.cwd() / candidate])
        if keyframes_root:
            choices.append(keyframes_root / candidate)
    if keyframes_root:
        video_id = str(row.get("video_id") or "")
        filename = candidate.name or f"{int(row.get('keyframe_id_int', row.get('keyframe_id', 0))):06d}.jpg"
        choices.append(keyframes_root / video_id / filename)
    for choice in choices:
        if choice.is_file():
            return choice.resolve()
    raise FileNotFoundError(raw or frame_identity(row))


def extract_metadata(metadata_path: str | Path, output_path: str | Path, keyframes_root: str | Path | None = None, workers: int | None = None) -> dict:
    metadata_path = Path(metadata_path).resolve()
    output_path = Path(output_path).resolve()
    root = Path(keyframes_root).resolve() if keyframes_root else None
    metadata = pd.read_csv(metadata_path)
    grids = np.empty((len(metadata), GRID_SIZE * GRID_SIZE), dtype=np.uint8)
    identities = []
    failures = []
    records = metadata.to_dict(orient="records")
    identities = [frame_identity(row) for row in records]

    def process(item):
        index, row = item
        try:
            with Image.open(resolve_frame_path(row, metadata_path, root)) as image:
                return index, extract_grid(image), None
        except Exception as exc:
            return index, np.full(GRID_SIZE * GRID_SIZE, 255, dtype=np.uint8), str(exc)

    worker_count = max(1, int(workers or min(8, os.cpu_count() or 1)))
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
      for completed, (index, grid, error) in enumerate(pool.map(process, enumerate(records)), 1):
        grids[index] = grid
        if error:
            failures.append({"row": index, "identity": identities[index], "error": error})
        if completed % 1000 == 0:
            print(f"Extracted {completed}/{len(metadata)} frames", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, colors=grids, identities=np.asarray(identities), grid_size=GRID_SIZE)
    temporary.replace(output_path)
    return {"frames": len(metadata), "failures": failures, "output": str(output_path)}
