"""CPU-only contracts for batching, frame boundaries, vLLM and resumable JSON."""
import json
import logging
import tempfile
import threading
import types
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from src.caption_extraction.pipeline.extract_caption import ExtractCaptionPipeline
from src.caption_extraction.pipeline.internvl import InternVLCaptioner, prepared_batches, validate_settings


class InternVLTests(unittest.TestCase):
    def pipeline(self, root=Path("."), **overrides):
        cfg = dict(
            caption_model="OpenGVLab/InternVL3_5-2B", shot_model="unused", shot_revision="unused",
            max_shot_seconds=10, min_caption_seconds=1, frame_sample_fps=1.5,
            caption_max_frames=16, caption_batch_size=2, max_new_tokens=220,
            duplicate_max_gap=.25, duplicate_similarity=.9, prompt="Describe.",
            language="English", speed_predictor={},
        )
        cfg.update(overrides)
        return ExtractCaptionPipeline({"extraction": cfg}, root, root, root / "out", logging.getLogger("test"))

    def test_validate_and_remove_legacy_backend(self):
        for overrides in ({"caption_model": "legacy-model"}, {"caption_model": "OpenGVLab/InternVL3_5-2B-HF"},
                          {"caption_batch_size": 0}, {"caption_max_frames": True},
                          {"vllm": {"typo": 1}}, {"vllm": {"gpu_memory_utilization": 1}},
                          {"caption_profiles": []}, {"max_new_tokens": 0}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.pipeline(**overrides)

    def test_sampling_actual_fps_and_max_frames(self):
        pipeline = self.pipeline()
        pipeline.captioner = SimpleNamespace(image_size=32)
        indices = []
        class Reader:
            def get_avg_fps(self): return 60
            def __len__(self): return 600
            def get_batch(self, selected):
                indices.extend(selected)
                rgb = np.zeros((len(selected), 36, 64, 3), dtype=np.uint8)
                rgb[..., 0] = 255
                return SimpleNamespace(asnumpy=lambda: rgb)
        frames, timestamps = pipeline._sample_frames(Reader(), 1, 6)
        self.assertEqual(len(frames), 8)
        self.assertEqual(frames[0].size, (32, 32))
        self.assertEqual(frames[0].getpixel((0, 0)), (255, 0, 0))
        self.assertEqual((indices[0], indices[-1]), (60, 359))
        self.assertEqual(timestamps[-1], 359 / 60)
        pipeline.max_frames = 4
        self.assertEqual(len(pipeline._sample_frames(Reader(), 0, 10)[0]), 4)

    def test_exact_shot_boundary_does_not_include_next_shot(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            pipeline = self.pipeline(root)
            pipeline.captioner = SimpleNamespace(image_size=8)
            video = root / "video.mp4"
            pipeline.scene_path(video).write_text("0 31\n32 70\n", encoding="utf-8")
            fps = 30000 / 1001
            with patch("src.caption_extraction.pipeline.extract_caption._probe_video", return_value=(fps, 5)):
                windows = pipeline._windows(video)
            selected = []
            reader = SimpleNamespace(get_avg_fps=lambda: fps,
                                     get_batch=lambda ids: selected.extend(ids) or SimpleNamespace(
                                         asnumpy=lambda: np.zeros((len(ids), 8, 8, 3), dtype=np.uint8)))
            class Reader:
                def __len__(self): return 150
                def get_avg_fps(self): return reader.get_avg_fps()
                def get_batch(self, ids): return reader.get_batch(ids)
            pipeline._sample_frames(Reader(), **windows[0])
            self.assertEqual(selected[-1], 31)

    def test_bounded_prefetch_overlaps_consumer_and_preserves_order(self):
        prepared = []
        ahead = threading.Event()
        def prepare(offset, batch):
            prepared.append(offset)
            if offset == 2:
                ahead.set()
            return [item["id"] for item in batch]
        windows = [{"id": i} for i in range(7)]
        with closing(prepared_batches(windows, 2, prepare)) as batches:
            first = next(batches)
            self.assertEqual(first[2], [0, 1])
            self.assertTrue(ahead.wait(timeout=2))
            self.assertEqual(prepared, [0, 2])  # no unbounded whole-video decode
            rest = [item[2] for item in batches]
        self.assertEqual(rest, [[2, 3], [4, 5], [6]])

    def test_prefetch_errors_propagate_and_thread_is_joined(self):
        def prepare(offset, batch):
            raise ValueError("bad frame")
        with self.assertRaisesRegex(ValueError, "bad frame"):
            list(prepared_batches([{}], 2, prepare))
        self.assertFalse(any(t.name.startswith("caption-decode") for t in threading.enumerate()))

    def make_captioner(self):
        tokenizer = Mock()
        tokenizer.apply_chat_template.side_effect = lambda messages, **kw: messages[-1]["content"]
        tokenizer.encode.return_value = [1] * 50
        config = SimpleNamespace(force_image_size=448, vision_config=SimpleNamespace(image_size=448, patch_size=14),
                                 downsample_ratio=.5)
        engine = Mock()
        engine.llm_engine.model_config.hf_config = config
        fake_vllm = types.ModuleType("vllm")
        fake_vllm.LLM = Mock(return_value=engine)
        fake_vllm.SamplingParams = Mock(side_effect=lambda **kwargs: kwargs)
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoTokenizer = SimpleNamespace(from_pretrained=Mock(return_value=tokenizer))
        with patch.dict("sys.modules", {"vllm": fake_vllm, "transformers": fake_transformers}), \
             patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.is_bf16_supported", return_value=False):
            captioner = InternVLCaptioner("OpenGVLab/InternVL3_5-2B", 16, 220,
                                         validate_settings("InternVL", 16, 8, {}), logging.getLogger("test"))
        return captioner, engine, tokenizer, fake_vllm

    def test_vllm_load_contract_and_ordered_frame_placeholders(self):
        captioner, engine, tokenizer, vllm = self.make_captioner()
        kwargs = vllm.LLM.call_args.kwargs
        self.assertEqual(kwargs["dtype"], "float16")
        self.assertEqual(kwargs["limit_mm_per_prompt"], {"image": 16, "video": 0})
        self.assertEqual(kwargs["mm_processor_kwargs"]["max_dynamic_patch"], 1)
        self.assertTrue(kwargs["enable_chunked_prefill"])
        self.assertEqual(captioner.image_tokens, 256)
        images = [object(), object()]
        request = captioner.request(images, [0, 1], "Describe.")
        self.assertEqual(request["multi_modal_data"]["image"], images)
        self.assertEqual(request["prompt"].count("<image>"), 2)
        self.assertLess(request["prompt"].index("Frame1"), request["prompt"].index("Frame2"))
        self.assertFalse(tokenizer.apply_chat_template.call_args.kwargs["enable_thinking"])
        self.assertEqual(captioner.sampling["temperature"], 0)
        captioner.max_model_len = 256
        with self.assertRaisesRegex(ValueError, "increase vllm.max_model_len"):
            captioner.request(images, [0, 1], "Describe.")

    def test_batch_generation_rejects_missing_or_empty_results(self):
        captioner, engine, _, _ = self.make_captioner()
        def result(text):
            return SimpleNamespace(outputs=[SimpleNamespace(text=text, token_ids=[1, 2], finish_reason="stop")])
        engine.generate.return_value = [result("First"), result("Second")]
        captions, tokens = captioner.generate([{}, {}])
        self.assertEqual(captions, ["First", "Second"])
        self.assertEqual(tokens, 4)
        self.assertEqual(len(engine.generate.call_args.args[0]), 2)
        engine.generate.return_value = [result("First")]
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            captioner.generate([{}, {}])
        engine.generate.return_value = [result("")]
        with self.assertRaisesRegex(RuntimeError, "empty"):
            captioner.generate([{}])

    def test_pipeline_batch_checkpoint_resume_labels_and_json_schema(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            video = root / "001.mp4"
            video.write_bytes(b"fake video")
            pipeline = self.pipeline(root)
            pipeline.scene_path(video).write_text("0 449\n", encoding="utf-8")
            windows = [dict(start=i * 5., end=(i + 1) * 5.) for i in range(3)]
            pipeline._windows = Mock(return_value=windows)
            pipeline._sample_frames = Mock(return_value=([object()], [0.]))
            captioner = Mock(image_size=448)
            captioner.request.side_effect = lambda *args: {}
            captioner.generate.side_effect = [(["The video segment shows A.", "The video segment shows B."], 4),
                                               RuntimeError("synthetic GPU failure")]
            pipeline._load_caption_model = lambda: setattr(pipeline, "captioner", captioner)
            decord = types.ModuleType("decord")
            decord.VideoReader = Mock(return_value=object())
            decord.cpu = Mock(return_value=0)
            speed = {str(video): ["timelapse", "slow-motion", "unknown"]}
            with patch.dict("sys.modules", {"decord": decord}), \
                 patch("src.caption_extraction.pipeline.extract_caption._probe_video", return_value=(30, 15)), \
                 patch("src.caption_extraction.pipeline.extract_caption.predict_speed_in_worker", return_value=speed):
                with self.assertRaisesRegex(RuntimeError, "Completed batches remain resumable"):
                    pipeline.caption_videos([video])
                partial = root / "out/001.partial.json"
                state = json.loads(partial.read_text(encoding="utf-8"))
                self.assertEqual(len(state["records"]), 2)
                captioner.generate.reset_mock(side_effect=True)
                captioner.generate.return_value = (["The video segment shows C."], 2)
                # Runtime memory tuning must not invalidate already completed captions.
                pipeline.vllm_config["max_num_seqs"] = 1
                pipeline.vllm_config["enforce_eager"] = True
                pipeline.caption_videos([video])
                self.assertEqual(len(captioner.generate.call_args.args[0]), 1)
            records = json.loads((root / "out/001.json").read_text(encoding="utf-8"))
            self.assertFalse(partial.exists())
            self.assertEqual(len(records), 3)
            self.assertTrue(all(set(r) == {"video_id", "start", "end", "caption"} for r in records))
            self.assertTrue(records[0]["caption"].startswith("The timelapse video segment"))
            self.assertTrue(records[1]["caption"].startswith("The slow-motion video segment"))
            self.assertTrue(records[2]["caption"].startswith("The video segment"))
            self.assertEqual(sorted(p.name for p in (root / "out").iterdir()), ["001.json"])


if __name__ == "__main__":
    unittest.main()
