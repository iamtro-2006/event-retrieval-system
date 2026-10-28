"""CPU checks for frame mapping, model dispatch and fresh per-shot chat state."""
import copy
import logging
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from src.caption_extraction.pipeline.extract_caption import ExtractCaptionPipeline, apply_speed_prefix
from src.caption_extraction.pipeline.internvl import prepare_video, video_question


class InternVLTests(unittest.TestCase):
    def pipeline(self, model="OpenGVLab/InternVL3_5-2B"):
        cfg = {"extraction": dict(
            caption_model=model, shot_model="unused", shot_revision="unused",
            max_shot_seconds=10, min_caption_seconds=1, frame_sample_fps=1.5,
            caption_profiles=[dict(max_frames=16, short_side=256)], max_new_tokens=220,
            duplicate_max_gap=.25, duplicate_similarity=.9, prompt="Describe.",
            language="English", speed_predictor={}, internvl={"max_tiles": 1},
        )}
        return ExtractCaptionPipeline(cfg, Path("."), Path("."), Path("."), logging.getLogger("test"))

    def test_video_normalization_and_temporal_order(self):
        frames = [np.zeros((3, 20, 40), dtype=np.uint8),
                  np.full((3, 20, 40), 255, dtype=np.uint8)]
        pixels, counts = prepare_video(frames, 32)
        self.assertEqual(tuple(pixels.shape), (2, 3, 32, 32))
        self.assertEqual(counts, [1, 1])
        torch.testing.assert_close(pixels[0, :, 0, 0],
                                   -torch.tensor([.485, .456, .406]) / torch.tensor([.229, .224, .225]))
        self.assertTrue(torch.all(pixels[1] > pixels[0]))
        question = video_question([1.1, 2.2], "Describe.")
        self.assertEqual(question.count("<image>"), 2)
        self.assertTrue(question.endswith("Describe."))

    def test_tiles_are_row_major_and_thumbnail_last(self):
        frame = np.zeros((3, 32, 64), dtype=np.uint8)
        frame[0, :, :32] = 255
        frame[2, :, 32:] = 255
        pixels, counts = prepare_video([frame], 32, max_tiles=2)
        self.assertEqual(counts, [3])
        self.assertGreater(pixels[0, 0].mean(), pixels[0, 2].mean())
        self.assertGreater(pixels[1, 2].mean(), pixels[1, 0].mean())

    def test_backend_selection_and_reject_hf_format(self):
        self.assertEqual(self.pipeline().caption_backend, "internvl")
        self.assertEqual(self.pipeline("DAMO-NLP-SG/VideoLLaMA3-2B").caption_backend, "videollama3")
        with self.assertRaisesRegex(ValueError, "without -HF"):
            self.pipeline("OpenGVLab/InternVL3_5-2B-HF")

    def test_sampling_actual_fps_keeps_native_internvl_pixels(self):
        pipeline = self.pipeline()
        indices = []
        class Reader:
            def get_avg_fps(self): return 60
            def __len__(self): return 600
            def get_batch(self, selected):
                indices.extend(selected)
                return SimpleNamespace(asnumpy=lambda: np.zeros((len(selected), 36, 64, 3), dtype=np.uint8))
        frames, timestamps = pipeline._sample_frames(Reader(), 1, 6, {"max_frames": 16, "short_side": 448})
        self.assertEqual(len(frames), 8)  # ceil(5s * 1.5fps)
        self.assertEqual(frames[0].shape, (3, 36, 64))
        self.assertEqual(indices[0], 60)
        self.assertEqual(indices[-1], 359)
        self.assertEqual(timestamps[-1], 359 / 60)

    def test_chat_oom_fallback_and_no_history_between_shots(self):
        pipeline = self.pipeline()
        pipeline.internvl_image_size = 448
        pipeline.caption_dtype = torch.float16
        pipeline.processor = object()
        pipeline.profiles = [{"max_frames": n, "short_side": 448} for n in (16, 12, 8)]
        pipeline._sample_frames = Mock(return_value=([np.zeros((3, 2, 2), dtype=np.uint8)], [0.0]))
        calls = []
        def chat(tokenizer, pixels, question, generation_config, **kwargs):
            calls.append((copy.deepcopy(generation_config), kwargs))
            generation_config["eos_token_id"] = 42  # upstream chat mutates this dict
            if len(calls) == 1:
                raise torch.cuda.OutOfMemoryError("synthetic")
            return "The video segment shows a person walking."
        pipeline.model = SimpleNamespace(chat=chat)
        tensor = Mock()
        tensor.to.return_value = tensor
        with patch("src.caption_extraction.pipeline.internvl.prepare_video", return_value=(tensor, [1])), \
             patch("torch.cuda.empty_cache"):
            text = pipeline._caption(None, 0, 5)
            pipeline._caption(None, 5, 10)
        self.assertEqual(len(calls), 3)
        for config, kwargs in calls:
            self.assertNotIn("eos_token_id", config)
            self.assertIsNone(kwargs["history"])
            self.assertEqual(kwargs["num_patches_list"], [1])
        self.assertTrue(apply_speed_prefix(text, "timelapse").startswith("The timelapse video segment"))


if __name__ == "__main__":
    unittest.main()
