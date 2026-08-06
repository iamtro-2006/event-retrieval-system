from __future__ import annotations

from dataclasses import dataclass

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
@dataclass
class DedupRecord:
    dropped: Candidate
    kept: Candidate
    reason: str
    phash_hamming: int | None = None
    dense_cosine: float | None = None


