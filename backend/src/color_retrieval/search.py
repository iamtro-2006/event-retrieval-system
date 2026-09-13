from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .extraction import GRID_SIZE, frame_identity
from .palette import COLOR_NAMES


class ColorIndex:
    def __init__(self, path: str | Path, metadata_records: list[dict]):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Color index not found: {self.path}. Run scripts/color/extract.py first.")
        archive = np.load(self.path, allow_pickle=False)
        self.colors = np.asarray(archive["colors"], dtype=np.uint8)
        if self.colors.shape != (len(metadata_records), GRID_SIZE * GRID_SIZE):
            raise ValueError(f"Color index shape {self.colors.shape} does not match {len(metadata_records)} metadata rows")
        if "identities" in archive:
            expected = np.asarray([frame_identity(row) for row in metadata_records])
            if not np.array_equal(archive["identities"].astype(str), expected):
                raise ValueError("Color index row identities do not match the loaded metadata")

    def search(self, metadata_records: list[dict], cells: list[dict], top_k: int, video_ids: list[str] | None = None) -> pd.DataFrame:
        positions = np.asarray([int(cell["row"]) * GRID_SIZE + int(cell["col"]) for cell in cells], dtype=np.intp)
        targets = np.asarray([COLOR_NAMES.index(str(cell["color"])) for cell in cells], dtype=np.uint8)
        scores = np.mean(self.colors[:, positions] == targets[None, :], axis=1, dtype=np.float32)
        valid = np.all(self.colors[:, positions] != 255, axis=1)
        if video_ids:
            allow = frozenset(str(value) for value in video_ids)
            valid &= np.fromiter((str(row.get("video_id")) in allow for row in metadata_records), dtype=bool)
        candidate_ids = np.flatnonzero(valid & (scores > 0))
        order = candidate_ids[np.argsort(-scores[candidate_ids], kind="stable")[:int(top_k)]]
        rows = []
        for rank, index in enumerate(order, 1):
            item = metadata_records[int(index)].copy()
            item.update(rank=rank, display_rank=rank, score=float(scores[index]), retrieval_score=float(scores[index]), color_score=float(scores[index]), search_mode="color")
            rows.append(item)
        return pd.DataFrame.from_records(rows)


@lru_cache(maxsize=4)
def load_color_index(path: str, modified_ns: int, metadata_count: int, identities: tuple[str, ...]) -> ColorIndex:
    # Cache key includes file mtime and metadata identities, so replacing an
    # offline index is picked up without restarting the API.
    records = []
    for identity in identities:
        video_id, frame_id = identity.split("\0", 1)
        records.append({"video_id": video_id, "keyframe_id": frame_id})
    return ColorIndex(path, records)


def get_color_index(path: str | Path, metadata_records: list[dict]) -> ColorIndex:
    resolved = Path(path).resolve()
    identities = tuple(frame_identity(row) for row in metadata_records)
    return load_color_index(str(resolved), resolved.stat().st_mtime_ns, len(metadata_records), identities)
