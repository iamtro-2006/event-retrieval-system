from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Candidate:
    shot_id: int
    frame_idx: int
    timestamp_sec: float
    source: str
    quality: float = 0.0
    valid: bool = True
    rejection_reason: str = ""
    blur: float = 0.0
    brightness: float = 0.0
    entropy: float = 0.0
    edge_density: float = 0.0
    clipped_fraction: float = 0.0
    feature: np.ndarray | None = None
    is_anchor: bool = False
    selection_source: str = ""
    mmr_score: float = 0.0
    spread: float = 0.0
    cluster_id: int = -1
    cluster_count: int = 1
    cluster_aucc: float = float("nan")
    representativeness: float = 0.0
    typicality: float = float("nan")
    volatility: float = float("nan")
    unique_score: float = float("nan")


@dataclass(frozen=True)
class Shot:
    shot_id: int
    start_frame: int
    end_frame: int
    fps: float

    @property
    def duration_sec(self) -> float:
        return (self.end_frame - self.start_frame + 1) / self.fps if self.fps > 0 else 0.0


@dataclass
class DedupRecord:
    dropped: Candidate
    kept: Candidate
    reason: str
    phash_hamming: int | None = None
    dense_cosine: float | None = None


@dataclass
class SelectionResult:
    selected: list[Candidate]
    dedup_dropped: list[DedupRecord] = field(default_factory=list)
    shot_diagnostics: dict[int, dict[str, Any]] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)



