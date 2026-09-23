from __future__ import annotations

import numpy as np

from src.keyframe_extraction.models.detector import predictions_to_scenes
from src.keyframe_extraction.models.selector import split_large_scenes


def test_predictions_to_scenes_collapses_transition_run_to_midpoint() -> None:
    predictions = np.asarray([0.0, 0.1, 0.8, 0.9, 0.7, 0.1, 0.0], dtype=np.float32)

    scenes = predictions_to_scenes(predictions, threshold=0.5)

    np.testing.assert_array_equal(scenes, np.asarray([[0, 3], [4, 6]], dtype=np.int32))


def test_predictions_to_scenes_keeps_single_full_scene_without_transition() -> None:
    predictions = np.zeros(5, dtype=np.float32)

    scenes = predictions_to_scenes(predictions, threshold=0.5)

    np.testing.assert_array_equal(scenes, np.asarray([[0, 4]], dtype=np.int32))


def test_predictions_to_scenes_handles_empty_input() -> None:
    scenes = predictions_to_scenes(np.asarray([], dtype=np.float32), threshold=0.5)

    assert scenes.shape == (0, 2)
    assert scenes.dtype == np.int32


def test_split_large_scenes_preserves_all_frame_indexes() -> None:
    scenes = np.asarray([[0, 9]], dtype=np.int32)

    chunks = split_large_scenes(scenes, max_gap=3)

    np.testing.assert_array_equal(
        chunks,
        np.asarray([[0, 3], [4, 7], [8, 9]], dtype=np.int32),
    )
