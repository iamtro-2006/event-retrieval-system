from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.api.serialization import (
    format_keyframe_id,
    safe_float,
    safe_int,
    serialize,
    serialize_matched_sequence,
)


class TestSafeFloat:
    def test_none(self):
        assert safe_float(None) == 0.0

    def test_nan(self):
        assert safe_float(float("nan")) == 0.0

    def test_string(self):
        assert safe_float("3.14") == 3.14

    def test_custom_default(self):
        assert safe_float(None, -1.0) == -1.0


class TestSafeInt:
    def test_none(self):
        assert safe_int(None) == 0

    def test_string_float(self):
        assert safe_int("42.7") == 42

    def test_custom_default(self):
        assert safe_int(None, 99) == 99


class TestFormatKeyframeId:
    def test_string_digit(self):
        assert format_keyframe_id({"keyframe_id": "3"}) == "000003"

    def test_string_non_digit(self):
        assert format_keyframe_id({"keyframe_id": "abc"}) == "abc"

    def test_keyframe_id_int_fallback(self):
        assert format_keyframe_id({"keyframe_id_int": 42}) == "000042"

    def test_frame_idx_fallback(self):
        assert format_keyframe_id({"frame_idx": 7}) == "000007"

    def test_empty_dict(self):
        assert format_keyframe_id({}) == "000000"


class TestSerialize:
    def test_basic_keys(self):
        item = {
            "video_id": "V001",
            "dataset": "L23",
            "keyframe_id": "000003",
            "keyframe_id_int": 3,
            "timestamp_sec": 12.5,
            "retrieval_score": 0.95,
            "rank": 1,
        }
        result = serialize(item)
        assert result["video_id"] == "V001"
        assert result["frame_id"] == 3
        assert result["frame_name"] == "000003.jpg"
        assert result["timestamp"] == 12.5
        assert result["similarity"] == 0.95
        assert result["rank"] == 1
        assert "image_url" in result
        assert "video_url" in result
        assert "map_url" in result
        assert "temporal" in result
        assert "matched_sequence" in result

    def test_keyframe_path_split(self):
        item = {
            "video_id": "V001",
            "dataset": "L23",
            "keyframe_id": "5",
            "keyframe_path": "data/processed/keyframes/L23/V001/000005.jpg",
            "retrieval_score": 0.8,
        }
        result = serialize(item)
        assert result["image_rel_path"] == "L23/V001/000005.jpg"
        assert result["image_url"] == "/static/keyframes/L23/V001/000005.jpg"

    def test_matched_sequence_serialized(self):
        item = {
            "video_id": "V001",
            "keyframe_id": "3",
            "retrieval_score": 0.9,
            "matched_sequence": [
                {
                    "video_id": "V001",
                    "keyframe_id": "1",
                    "timestamp_sec": 5.0,
                    "score": 0.8,
                },
                {
                    "video_id": "V001",
                    "keyframe_id": "5",
                    "timestamp_sec": 25.0,
                    "score": 0.9,
                },
            ],
        }
        result = serialize(item)
        assert len(result["matched_sequence"]) == 2
        assert result["matched_sequence"][0]["frame_id"] == 1
        assert result["matched_sequence"][1]["frame_id"] == 5

    def test_nan_score_handled(self):
        item = {
            "video_id": "V001",
            "keyframe_id": "3",
            "retrieval_score": float("nan"),
        }
        result = serialize(item)
        assert result["similarity"] == 0.0


class TestSerializeMatchedSequence:
    def test_empty(self):
        assert serialize_matched_sequence([]) == []

    def test_non_list(self):
        assert serialize_matched_sequence(None) == []
        assert serialize_matched_sequence("not a list") == []

    def test_non_dict_items_skipped(self):
        assert serialize_matched_sequence([1, 2, 3]) == []
