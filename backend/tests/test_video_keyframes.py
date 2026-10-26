import unittest

from src.api.legacy.video_keyframes import build_video_keyframes


class VideoKeyframeContractTests(unittest.TestCase):
    def serialize(self, record):
        return {"image_url": f"/static/v3c/keyframes/V3C/video/{record['keyframe_id']}.jpg",
                "image_rel_path": f"V3C/video/{record['keyframe_id']}.jpg"}

    def test_metadata_keeps_sparse_keyframe_identity_separate_from_decoded_frame(self):
        records = [
            {"keyframe_id": 8, "frame_idx": 240, "timestamp_sec": 12.0, "fps": 20},
            {"keyframe_id": 2, "frame_idx": 40, "timestamp_sec": 2.0, "fps": 20},
        ]
        items = build_video_keyframes(records, self.serialize)
        self.assertEqual([item["keyframe_id"] for item in items], ["2", "8"])
        self.assertEqual(items[0]["frame_idx"], 40)
        self.assertEqual(items[0]["timestamp"], 2.0)
        self.assertEqual(items[0]["image_rel_path"], "V3C/video/2.jpg")
        self.assertTrue(items[0]["image_url"].startswith("/static/v3c/"))

    def test_missing_timestamp_uses_metadata_fps_without_defaulting_to_zero(self):
        for missing in (None, float("nan"), -1):
            with self.subTest(timestamp=missing):
                items = build_video_keyframes([
                    {"keyframe_id": "000013", "frame_idx": 120, "fps": 24, "timestamp_sec": missing}
                ], self.serialize)
                self.assertEqual(items[0]["timestamp"], 5)
                self.assertEqual(items[0]["keyframe_id"], "000013")

    def test_bad_times_are_omitted_and_duplicate_keyframes_are_not_repeated(self):
        records = [
            {"keyframe_id": "1", "timestamp": 0},
            {"keyframe_id": "1", "timestamp": 0},
            {"keyframe_id": "2", "timestamp": float("inf")},
            {"keyframe_id": "3", "timestamp": -2},
            {"keyframe_id": "4"},
        ]
        items = build_video_keyframes(records, self.serialize)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["timestamp"], 0)


if __name__ == "__main__":
    unittest.main()
