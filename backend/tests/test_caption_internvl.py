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
from src.caption_extraction.pipeline.internvl import (InternVLCaptioner, occupied_color_bins,
                                                     prepared_batches, resolve_gpu_settings, validate_settings)


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
                          {"caption_profiles": []}, {"max_new_tokens": 0},
                          {"caption_min_color_bins": 0}, {"caption_min_color_bins": True}):
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
                # Eight occupied bins, with red in the first band.
                for band, value in enumerate((255, 0, 32, 64, 96, 128, 160, 192)):
                    rgb[:, :, band * 8:(band + 1) * 8, 0] = value
                return SimpleNamespace(asnumpy=lambda: rgb)
        frames, timestamps = pipeline._sample_frames(Reader(), 1, 6)
        self.assertEqual(len(frames), 8)
        self.assertEqual(frames[0].size, (32, 32))
        self.assertEqual(frames[0].getpixel((0, 0)), (255, 0, 0))
        self.assertEqual((indices[0], indices[-1]), (60, 359))
        self.assertEqual(timestamps[-1], 359 / 60)
        pipeline.max_frames = 4
        self.assertEqual(len(pipeline._sample_frames(Reader(), 0, 10)[0]), 4)

    def test_flat_filter_strict_boundary_and_no_interpolation_colors(self):
        for color in (0, 255):
            self.assertEqual(occupied_color_bins(np.full((8, 8, 3), color, dtype=np.uint8)), 1)
        pipeline = self.pipeline(frame_sample_fps=4)
        pipeline.captioner = SimpleNamespace(image_size=448)
        raw = np.zeros((4, 8, 8, 3), dtype=np.uint8)
        for i, bins in enumerate((1, 4, 5, 6)):
            raw[i, :, :, 0] = ((np.arange(8) % bins) * 32)[None, :]
            self.assertEqual(occupied_color_bins(raw[i]), bins)
        class Reader:
            def get_avg_fps(self): return 4
            def __len__(self): return 4
            def get_batch(self, ids): return SimpleNamespace(asnumpy=lambda: raw[ids])
        frames, timestamps = pipeline._sample_frames(Reader(), 0, 1)
        self.assertEqual(len(frames), 2)  # <5 rejected, exactly 5 retained
        self.assertEqual(timestamps, [.5, .75])
        self.assertTrue(all(frame.size == (448, 448) for frame in frames))

    def test_prepare_batch_does_not_make_request_for_empty_shot(self):
        pipeline = self.pipeline()
        pipeline.captioner = Mock(image_size=448)
        pipeline._sample_frames = Mock(side_effect=[([], []), ([object()], [1.5])])
        windows = [dict(start=0, end=1), dict(start=1, end=2)]
        items = pipeline._prepare_batch(None, Path("video.mp4"), 2, 0, 0, windows)
        self.assertIsNone(items[0])
        pipeline.captioner.request.assert_called_once()

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

    def test_long_shot_split_has_exact_shared_edges(self):
        from src.caption_extraction.pipeline.speed import speed_segment_bounds
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            pipeline = self.pipeline(root)
            video = root / "video.mp4"
            pipeline.scene_path(video).write_text("0 656\n657 899\n", encoding="utf-8")
            fps = 30000 / 1001
            with patch("src.caption_extraction.pipeline.extract_caption._probe_video", return_value=(fps, 40)):
                windows = pipeline._windows(video)
            self.assertEqual(len(windows), 4)
            for left, right in zip(windows, windows[1:]):
                self.assertEqual(left["end"], right["start"])
            bounds = speed_segment_bounds(windows, fps)
            for left, right in zip(bounds, bounds[1:]):
                self.assertEqual(left[1], right[0])
            pipeline.scene_path(video).write_text("0 656\n650 899\n", encoding="utf-8")
            with patch("src.caption_extraction.pipeline.extract_caption._probe_video", return_value=(fps, 40)):
                with self.assertRaisesRegex(ValueError, "Scenes overlap or are unsorted"):
                    pipeline._windows(video)

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

    def make_captioner(self, gpu_count=1):
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
             patch("torch.cuda.device_count", return_value=gpu_count), \
             patch("torch.cuda.is_bf16_supported", return_value=False):
            captioner = InternVLCaptioner("OpenGVLab/InternVL3_5-2B", 16, 220,
                                         validate_settings("InternVL", 16, 8, {}), logging.getLogger("test"))
        return captioner, engine, tokenizer, fake_vllm

    def test_gpu_autodetect_one_and_two_gpu_engine_contract(self):
        for count, tp, concurrency, tokens in ((1, 1, 3, 4096), (2, 2, 6, 8192)):
            captioner, _, _, vllm = self.make_captioner(count)
            kwargs = vllm.LLM.call_args.kwargs
            self.assertEqual(kwargs['tensor_parallel_size'], tp)
            self.assertEqual(kwargs['max_num_seqs'], concurrency)
            self.assertEqual(kwargs['max_num_batched_tokens'], tokens)
            self.assertEqual(captioner.settings['max_num_seqs'] * 2, 6 * tp)
        settings = validate_settings('InternVL', 16, 'auto', {})
        with self.assertRaisesRegex(RuntimeError, 'No visible CUDA GPU'):
            resolve_gpu_settings(0, settings)
        settings['tensor_parallel_size'] = 2
        with self.assertRaisesRegex(ValueError, 'detected 1'):
            resolve_gpu_settings(1, settings)

    def test_pipeline_resolves_automatic_batch_after_engine_load(self):
        for count, expected in ((1, 6), (2, 12)):
            pipeline = self.pipeline(caption_batch_size='auto')
            captioner = SimpleNamespace(settings=resolve_gpu_settings(count, pipeline.vllm_config))
            with patch('src.caption_extraction.pipeline.extract_caption.InternVLCaptioner', return_value=captioner):
                pipeline._load_caption_model()
            self.assertEqual(pipeline.batch_size, expected)

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

    def test_skipped_windows_keep_label_alignment_and_resume_position(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            video = root / "001.mp4"
            video.write_bytes(b"fake video")
            pipeline = self.pipeline(root)
            pipeline.scene_path(video).write_text("0 449\n", encoding="utf-8")
            windows = [dict(start=i * 5., end=(i + 1) * 5.) for i in range(3)]
            pipeline._windows = Mock(return_value=windows)
            pipeline._sample_frames = Mock(side_effect=[([], []), ([object()], [5.]), ([object()], [10.])])
            captioner = Mock(image_size=448)
            captioner.request.side_effect = lambda *args: {}
            captioner.generate.side_effect = [(["The video segment shows B."], 2), RuntimeError("stop")]
            pipeline._load_caption_model = lambda: setattr(pipeline, "captioner", captioner)
            decord = types.ModuleType("decord")
            decord.VideoReader = Mock(return_value=object())
            decord.cpu = Mock(return_value=0)
            with patch.dict("sys.modules", {"decord": decord}), \
                 patch("src.caption_extraction.pipeline.extract_caption._probe_video", return_value=(30, 15)), \
                 patch("src.caption_extraction.pipeline.extract_caption.predict_speed_in_worker",
                       return_value={str(video): ["timelapse", "slow-motion", "normal"]}):
                with self.assertRaises(RuntimeError):
                    pipeline.caption_videos([video])
                partial = json.loads((root / "out/001.partial.json").read_text(encoding="utf-8"))
                self.assertEqual(len(partial["records"]), 2)
                self.assertEqual(partial["records"][0]["skip_reason"], "uniform_frames")
                captioner.generate.side_effect = None
                captioner.generate.return_value = (["The video segment shows C."], 2)
                pipeline._sample_frames.side_effect = None
                pipeline._sample_frames.return_value = ([object()], [10.])
                pipeline.caption_videos([video])
                pipeline._sample_frames.assert_called_with(decord.VideoReader.return_value, 10., 15.)
            records = json.loads((root / "out/001.json").read_text(encoding="utf-8"))
            self.assertEqual([r["start"] for r in records], [5., 10.])
            self.assertTrue(records[0]["caption"].startswith("The slow-motion video segment"))
            self.assertTrue(records[1]["caption"].startswith("The video segment"))
            self.assertTrue(all(set(r) == {"video_id", "start", "end", "caption"} for r in records))

    def test_all_flat_video_never_calls_generation_and_writes_empty_list(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            video = root / "001.mp4"
            video.write_bytes(b"fake video")
            pipeline = self.pipeline(root)
            pipeline.scene_path(video).write_text("0 149\n", encoding="utf-8")
            pipeline._windows = Mock(return_value=[dict(start=0., end=5.)])
            pipeline._sample_frames = Mock(return_value=([], []))
            captioner = Mock(image_size=448)
            pipeline._load_caption_model = lambda: setattr(pipeline, "captioner", captioner)
            decord = types.ModuleType("decord")
            decord.VideoReader = Mock(return_value=object())
            decord.cpu = Mock(return_value=0)
            with patch.dict("sys.modules", {"decord": decord}), \
                 patch("src.caption_extraction.pipeline.extract_caption._probe_video", return_value=(30, 5)), \
                 patch("src.caption_extraction.pipeline.extract_caption.predict_speed_in_worker",
                       return_value={str(video): ["normal"]}):
                pipeline.caption_videos([video])
            captioner.generate.assert_not_called()
            captioner.request.assert_not_called()
            self.assertEqual(json.loads((root / "out/001.json").read_text(encoding="utf-8")), [])
            self.assertFalse((root / "out/001.partial.json").exists())


if __name__ == "__main__":
    unittest.main()
