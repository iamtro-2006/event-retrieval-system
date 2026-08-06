from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

import numpy as np
import pandas as pd

from src.keyframe_extraction.models.candidate_sampler import phash_hamming, sample_candidates
from src.keyframe_extraction.models.deduplicator import deduplicate, dense_cosine
from src.keyframe_extraction.models.detector import repair_and_split_scenes
from src.keyframe_extraction.models.p3_selector import select_common_anchor, select_p3, unique_gate
from src.keyframe_extraction.models.quality_filter import evaluate_quality
from src.keyframe_extraction.models.selector import save_keyframe_map
from src.keyframe_extraction.pipeline.extract_keyframes import KeyframeExtractionPipeline
from src.keyframe_extraction.schemas import Candidate
from src.utils.config import CandidateConfig, ClusteringConfig, DedupConfig, QualityConfig, SelectorConfig


def candidate(frame_idx: int, feature: list[float], quality: float = 0.8, shot_id: int = 0) -> Candidate:
    value = np.asarray(feature, dtype=np.float32)
    value /= max(float(np.linalg.norm(value)), 1e-12)
    return Candidate(
        shot_id=shot_id,
        frame_idx=frame_idx,
        timestamp_sec=float(frame_idx),
        source="test",
        quality=quality,
        feature=value,
    )


class P3AlgorithmTests(unittest.TestCase):
    def test_video_id_filter_is_case_insensitive_and_reports_missing(self):
        videos = [Path("group/L21_V001.mp4"), Path("group/L22_V001.mp4")]
        self.assertEqual(
            KeyframeExtractionPipeline._filter_videos(videos, Path("."), ["l21_v001"], None),
            [videos[0]],
        )
        with self.assertRaisesRegex(FileNotFoundError, "L99_V999"):
            KeyframeExtractionPipeline._filter_videos(videos, Path("."), ["L99_V999"], None)

    def test_group_filter_preserves_input_root(self):
        videos = [Path("Videos_L21_a/video/L21_V001.mp4"), Path("Videos_L22_a/video/L22_V001.mp4")]
        self.assertEqual(
            KeyframeExtractionPipeline._filter_videos(videos, Path("."), None, ["videos_l22_a"]),
            [videos[1]],
        )

    def test_quality_filter_rejects_black_frame(self):
        item = Candidate(0, 0, 0.0, "test")
        evaluate_quality(item, np.zeros((32, 32, 3), dtype=np.uint8), QualityConfig())
        self.assertFalse(item.valid)
        self.assertEqual(item.rejection_reason, "severe_blur")

    def test_scene_repair_splits_long_inclusive_range(self):
        repaired = repair_and_split_scenes(np.asarray([[0, 99]]), fps=10.0, max_duration_sec=3.0)
        self.assertEqual(repaired.tolist(), [[0, 29], [30, 59], [60, 89], [90, 99]])

    def test_phash_hamming(self):
        self.assertEqual(phash_hamming(0b1010, 0b0011), 2)

    def test_candidate_sampler_accepts_frame_iterator(self):
        frames = [(index, np.zeros((16, 16, 3), dtype=np.uint8)) for index in range(5)]
        sampled = sample_candidates(
            frames,
            np.asarray([[0, 4]], dtype=np.int32),
            fps=4.0,
            cfg=CandidateConfig(min_gap_sec=0.25, max_gap_sec=0.5, boundary_margin_sec=0.0),
        )
        self.assertEqual([item.frame_idx for item in sampled], [0, 2, 4])

    def test_rejected_candidate_does_not_advance_sampler_state(self):
        frames = [(index, np.full((16, 16, 3), index, dtype=np.uint8)) for index in range(5)]
        images: dict[int, np.ndarray] = {}

        def reject_frame_two(item: Candidate, _rgb: np.ndarray) -> None:
            item.valid = item.frame_idx != 2

        sampled = sample_candidates(
            frames,
            np.asarray([[0, 4]], dtype=np.int32),
            fps=2.0,
            cfg=CandidateConfig(
                min_gap_sec=0.5,
                max_gap_sec=1.0,
                phash_min_distance=65,
                pixel_change_threshold=2.0,
                boundary_margin_sec=0.0,
            ),
            observer=reject_frame_two,
            image_cache=images,
        )
        self.assertEqual([item.frame_idx for item in sampled], [0, 3, 4])
        self.assertEqual(sorted(images), [0, 3, 4])

    def test_dense_cosine_normalizes_inputs(self):
        self.assertAlmostEqual(dense_cosine(np.asarray([2.0, 0.0]), np.asarray([5.0, 0.0])), 1.0)

    def test_common_anchor_prefers_typical_stable_candidate(self):
        items = [candidate(0, [1.0, 0.0]), candidate(1, [0.99, 0.01]), candidate(2, [0.0, 1.0])]
        selected = select_common_anchor(items, SelectorConfig())
        self.assertIn(selected.frame_idx, {0, 1})

    def test_unique_gate_handles_single_candidate(self):
        only = candidate(0, [1.0, 0.0])
        unique, passed, reason = unique_gate([only], only, 0.0, SelectorConfig())
        self.assertIsNone(unique)
        self.assertFalse(passed)
        self.assertEqual(reason, "too_few_candidates")

    def test_unique_gate_accepts_distinct_interior_candidate(self):
        items = [
            candidate(0, [1.0, 0.0]),
            candidate(2, [0.0, 1.0]),
            candidate(4, [0.99, 0.01]),
        ]
        cfg = SelectorConfig(
            min_unique_quality=0.0,
            max_common_cosine=0.95,
            min_shot_spread=0.0,
            min_volatility_range=0.0,
            min_unique_volatility=0.0,
            min_unique_atypicality=0.0,
            unique_boundary_margin_sec=0.1,
        )
        unique, passed, reason = unique_gate(items, items[0], spread=0.2, cfg=cfg)
        self.assertIsNotNone(unique)
        self.assertEqual(unique.frame_idx, 2)
        self.assertTrue(passed)
        self.assertEqual(reason, "")

    def test_p3_single_candidate_shot_returns_anchor(self):
        only = candidate(10, [1.0, 0.0])
        selected, diagnostics, _ = select_p3(
            {0: [only]}, ClusteringConfig(), SelectorConfig(), DedupConfig()
        )
        self.assertEqual(selected, [only])
        self.assertEqual(selected[0].selection_source, "shot_common_anchor")
        self.assertEqual(diagnostics[0]["unique_reason"], "too_few_candidates")

    def test_mmr_respects_temporal_gap(self):
        items = [
            candidate(0, [1.0, 0.0]),
            candidate(1, [0.0, 1.0]),
            candidate(4, [-1.0, 0.0]),
            candidate(8, [0.0, -1.0]),
        ]
        selector_cfg = SelectorConfig(
            min_temporal_gap_sec=2.0,
            global_novelty_cosine=0.99,
            min_unique_quality=0.0,
            max_common_cosine=1.1,
            min_shot_spread=0.0,
            min_volatility_range=0.0,
            min_unique_volatility=0.0,
            min_unique_atypicality=0.0,
            unique_boundary_margin_sec=0.0,
        )
        selected, _, _ = select_p3(
            {0: items},
            ClusteringConfig(min_cluster_size=1, min_spread=0.0),
            selector_cfg,
            DedupConfig(dense_cosine_threshold=1.1),
        )
        for left_index, left in enumerate(selected):
            for right in selected[left_index + 1 :]:
                self.assertGreaterEqual(abs(left.timestamp_sec - right.timestamp_sec), 2.0)

    def test_dedup_drops_dense_duplicate(self):
        first = candidate(0, [1.0, 0.0])
        second = candidate(1, [1.0, 0.0])
        image = np.full((32, 32, 3), 127, dtype=np.uint8)
        kept, dropped = deduplicate(
            [first, second],
            {0: image, 1: image.copy()},
            DedupConfig(phash_hamming_threshold=-1, dense_cosine_threshold=0.97, temporal_window_sec=4.0),
        )
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(dropped), 1)
        self.assertTrue(dropped[0].reason.startswith("dense_cosine_"))

    def test_mapping_csv_schema(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "map.csv"
            with patch("src.keyframe_extraction.models.selector.get_video_fps", return_value=25.0):
                save_keyframe_map([0, 25], Path("L01_V001.mp4"), output)
            frame = pd.read_csv(output)
            self.assertEqual(
                frame.columns.tolist(), ["keyframe_id", "video_id", "frame_idx", "timestamp_sec", "fps"]
            )
            self.assertEqual(frame["timestamp_sec"].tolist(), [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
