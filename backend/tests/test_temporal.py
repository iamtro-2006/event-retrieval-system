from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.index.temporal import (
    Candidates,
    TemporalMatch,
    matches_to_dataframe,
    search,
)


def _make_candidates(
    video_ids: list[str],
    timestamps: list[float],
    records: list[dict],
    embedding_matrix: np.ndarray,
) -> Candidates:
    return Candidates(
        video_id=np.array(video_ids, dtype=object),
        timestamp_sec=np.array(timestamps, dtype=np.float32),
        row_index=np.arange(len(video_ids), dtype=np.int64),
        records=records,
        embedding_matrix=embedding_matrix.astype(np.float32),
    )


def _unit(vec: np.ndarray) -> np.ndarray:
    vec = vec.astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-12 else vec


class TestSearchBasic:
    def test_single_video_optimal_path(self):
        """Two-query temporal search on 5 frames where the optimal path
        is frame 0 -> frame 3."""
        dim = 4
        q0 = _unit(np.array([1, 0, 0, 0], dtype=np.float32))
        q1 = _unit(np.array([0, 1, 0, 0], dtype=np.float32))

        frame_embs = np.zeros((5, dim), dtype=np.float32)
        frame_embs[0] = _unit(np.array([1, 0, 0, 0], dtype=np.float32))
        frame_embs[3] = _unit(np.array([0, 1, 0, 0], dtype=np.float32))

        records = [{"video_id": "V001", "keyframe_id": f"{i:06d}"} for i in range(5)]
        timestamps = [float(i) for i in range(5)]

        cands = _make_candidates(["V001"] * 5, timestamps, records, frame_embs)

        matches = search(
            candidates=cands,
            query_embeddings=np.stack([q0, q1]),
            sub_queries=["a person walking", "then sitting"],
            top_k_videos=10,
        )
        assert len(matches) >= 1
        m = matches[0]
        assert m.video_id == "V001"
        assert m.selected_indices == [0, 3]
        assert m.score == pytest.approx(2.0, abs=0.01)
        assert m.avg_score == pytest.approx(1.0, abs=0.01)

    def test_empty_candidates(self):
        cands = _make_candidates([], [], [], np.zeros((0, 4), dtype=np.float32))
        matches = search(
            candidates=cands,
            query_embeddings=np.zeros((2, 4), dtype=np.float32),
            sub_queries=["q0", "q1"],
        )
        assert matches == []

    def test_video_too_short_skipped(self):
        """A video with fewer frames than queries is skipped."""
        q0 = _unit(np.array([1, 0], dtype=np.float32))
        q1 = _unit(np.array([0, 1], dtype=np.float32))
        frame_embs = np.array([[1, 0]], dtype=np.float32)
        records = [{"video_id": "V001", "keyframe_id": "000000"}]

        cands = _make_candidates(["V001"], [0.0], records, frame_embs)
        matches = search(
            candidates=cands,
            query_embeddings=np.stack([q0, q1]),
            sub_queries=["q0", "q1"],
        )
        assert matches == []

    def test_multiple_videos_ranked(self):
        """Two videos; the one with a better alignment should rank first."""
        dim = 2
        q0 = _unit(np.array([1, 0], dtype=np.float32))
        q1 = _unit(np.array([0, 1], dtype=np.float32))

        emb_good = np.stack(
            [
                _unit(np.array([1, 0], dtype=np.float32)),
                _unit(np.array([0, 1], dtype=np.float32)),
            ]
        )
        emb_bad = np.stack(
            [
                _unit(np.array([0.5, 0.5], dtype=np.float32)),
                _unit(np.array([0.5, 0.5], dtype=np.float32)),
            ]
        )

        records_good = [
            {"video_id": "V_good", "keyframe_id": f"{i:06d}"} for i in range(2)
        ]
        records_bad = [
            {"video_id": "V_bad", "keyframe_id": f"{i:06d}"} for i in range(2)
        ]

        video_ids = ["V_good", "V_good", "V_bad", "V_bad"]
        timestamps = [0.0, 1.0, 0.0, 1.0]
        records = records_good + records_bad
        matrix = np.vstack([emb_good, emb_bad])

        cands = _make_candidates(video_ids, timestamps, records, matrix)
        matches = search(
            candidates=cands,
            query_embeddings=np.stack([q0, q1]),
            sub_queries=["q0", "q1"],
            top_k_videos=10,
        )
        assert len(matches) == 2
        assert matches[0].video_id == "V_good"
        assert matches[0].avg_score > matches[1].avg_score


class TestDurationLimit:
    def test_duration_limit_windows(self):
        """With duration_limit, only windows within the limit are searched."""
        dim = 2
        q0 = _unit(np.array([1, 0], dtype=np.float32))
        q1 = _unit(np.array([0, 1], dtype=np.float32))

        frame_embs = np.zeros((6, dim), dtype=np.float32)
        frame_embs[0] = _unit(np.array([1, 0], dtype=np.float32))
        frame_embs[5] = _unit(np.array([0, 1], dtype=np.float32))

        records = [{"video_id": "V001", "keyframe_id": f"{i:06d}"} for i in range(6)]
        timestamps = [float(i) for i in range(6)]

        cands = _make_candidates(["V001"] * 6, timestamps, records, frame_embs)

        # With a 2-second window, frame 0 -> frame 5 (5s apart) won't be found.
        matches = search(
            candidates=cands,
            query_embeddings=np.stack([q0, q1]),
            sub_queries=["q0", "q1"],
            duration_limit=2.0,
        )
        # The best in-window alignment has low score; still may return something
        # but it won't be the frame 0 -> 5 path.
        if matches:
            for m in matches:
                assert m.duration_sec <= 2.0 + 0.01


class TestMatchesToDataFrame:
    def test_dataframe_shape(self):
        match = TemporalMatch(
            video_id="V001",
            score=2.0,
            avg_score=1.0,
            selected_indices=[0, 3],
            selected_keyframes=[
                {"video_id": "V001", "keyframe_id": "000000", "sub_query": "q0"},
                {"video_id": "V001", "keyframe_id": "000003", "sub_query": "q1"},
            ],
            duration_sec=3.0,
            start_time=0.0,
            end_time=3.0,
        )
        df = matches_to_dataframe([match])
        assert len(df) == 1
        row = df.iloc[0]
        assert row["video_id"] == "V001"
        assert row["rank"] == 1
        assert row["avg_score"] == pytest.approx(1.0)
        assert row["temporal_start_time"] == 0.0
        assert row["temporal_end_time"] == 3.0
        assert len(row["matched_sequence"]) == 2

    def test_empty_matches(self):
        df = matches_to_dataframe([])
        assert df.empty


class TestLengthMismatch:
    def test_mismatched_queries_raises(self):
        cands = _make_candidates(
            ["V001"],
            [0.0],
            [{"video_id": "V001"}],
            np.zeros((1, 4), dtype=np.float32),
        )
        with pytest.raises(ValueError, match="length mismatch"):
            search(
                candidates=cands,
                query_embeddings=np.zeros((2, 4), dtype=np.float32),
                sub_queries=["only one"],
            )
