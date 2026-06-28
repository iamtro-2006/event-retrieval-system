from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.index.embedding_index import (
    collect_embedding_files,
    parse_embedding_path,
    parse_keyframe_stem,
)


class TestParseKeyframeStem:
    def test_pure_digit(self):
        result = parse_keyframe_stem("000003")
        assert result["keyframe_id_int"] == 3
        assert result["keyframe_id_str"] == "000003"
        assert result["source_name"] == "000003"
        assert result["frame_idx_from_name"] is None

    def test_short_digit(self):
        result = parse_keyframe_stem("42")
        assert result["keyframe_id_int"] == 42
        assert result["keyframe_id_str"] == "000042"

    def test_debug_frame_pattern(self):
        result = parse_keyframe_stem("debug_5_frame_4521")
        assert result["keyframe_id_int"] == 5
        assert result["frame_idx_from_name"] == 4521

    def test_shot_frame_pattern(self):
        result = parse_keyframe_stem("shot_3_frame_12")
        assert result["keyframe_id_int"] == 3
        assert result["frame_idx_from_name"] == 12

    def test_no_digits_raises(self):
        with pytest.raises(ValueError, match="Cannot parse keyframe stem"):
            parse_keyframe_stem("no_numbers_here")


class TestParseEmbeddingPath:
    def test_standard_path(self, tmp_path):
        root = tmp_path / "embeddings"
        emb_path = root / "L23" / "V001" / "000003.npy"
        emb_path.parent.mkdir(parents=True)

        dataset, video_id, parsed = parse_embedding_path(emb_path, root)
        assert dataset == "L23"
        assert video_id == "V001"
        assert parsed["keyframe_id_int"] == 3

    def test_short_path_raises(self, tmp_path):
        root = tmp_path / "embeddings"
        emb_path = root / "000003.npy"
        emb_path.parent.mkdir(parents=True)

        with pytest.raises(ValueError, match="Invalid embedding path"):
            parse_embedding_path(emb_path, root)


class TestCollectEmbeddingFiles:
    def test_finds_npy_files(self, tmp_path):
        root = tmp_path / "embeddings"
        (root / "dataset_A" / "V001").mkdir(parents=True)
        (root / "dataset_A" / "V002").mkdir(parents=True)

        for vid in ["V001", "V002"]:
            for i in range(3):
                np.save(root / "dataset_A" / vid / f"{i:06d}.npy", np.zeros(4))

        paths = collect_embedding_files(root)
        assert len(paths) == 6
        assert all(p.suffix == ".npy" for p in paths)

    def test_missing_root_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            collect_embedding_files(tmp_path / "nonexistent")

    def test_empty_root_returns_empty(self, tmp_path):
        root = tmp_path / "empty"
        root.mkdir()
        paths = collect_embedding_files(root)
        assert paths == []
