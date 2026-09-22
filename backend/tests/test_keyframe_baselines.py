from __future__ import annotations

import numpy as np

from src.keyframe_extraction.baselines import (
    SelectedFrame,
    _beta_region_features,
    _robust_threshold,
    _select_with_temporal_nms,
)


def test_beta_region_features_are_finite_and_normalized() -> None:
    gray = np.tile(np.arange(64, dtype=np.uint8), (64, 1)) * 4
    features = _beta_region_features(gray, grid_rows=4, grid_cols=4)
    assert features.shape == (16, 16)
    assert np.isfinite(features).all()
    np.testing.assert_allclose(np.linalg.norm(features, axis=1), 1.0, atol=1e-5)


def test_robust_threshold_rejects_regular_scores() -> None:
    scores = np.asarray([1.0] * 99 + [20.0])
    threshold = _robust_threshold(scores, mad_scale=3.5, percentile=98.0)
    assert threshold > 1.0
    assert threshold <= 20.0


def test_temporal_nms_keeps_strongest_candidate_in_gap() -> None:
    scored = [
        SelectedFrame(10, 0.4, 5.0),
        SelectedFrame(12, 0.48, 8.0),
        SelectedFrame(40, 1.6, 6.0),
    ]
    selected = _select_with_temporal_nms(scored, threshold=4.0, min_gap_frames=5)
    assert [item.frame_idx for item in selected] == [12, 40]
