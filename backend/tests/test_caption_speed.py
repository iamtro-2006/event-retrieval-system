"""Inference parity and streaming regressions; synthetic CPU tensors, no downloads."""
import logging
import math
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.caption_extraction.pipeline.speed import ShotSpeedPredictor, original_window_slices, speed_label
from src.caption_extraction.pipeline.extract_caption import apply_speed_prefix, merge_duplicate_captions, predict_speed_in_worker
from src.utils.omni_progress import inference_with_progress
from src.utils.gpu_video import CudaVideoReader


class OriginalSpeedTests(unittest.TestCase):
    def test_worker_stream_protocol_and_native_failure(self):
        import io
        class Pipe(io.StringIO):
            def close(self): pass
        class Process:
            pid = 123
            def __init__(self, **kw):
                self.stdin = Pipe()
                self.stdout = io.StringIO('diagnostic line\n__SPEED_LABELS__={"v.mp4":["normal"]}\n')
            def __enter__(self): return self
            def __exit__(self,*a): pass
            def wait(self,**kw): return 0
            def poll(self): return 0
        videos = [dict(path="v.mp4",windows=[dict(start=0,end=5)],fps=30)]
        with patch('src.caption_extraction.pipeline.extract_caption.subprocess.Popen',side_effect=lambda *a,**kw: Process(**kw)):
            self.assertEqual(predict_speed_in_worker({},videos,logging.getLogger("test")),{"v.mp4":["normal"]})
        class Crash(Process):
            def __init__(self,**kw):
                super().__init__(**kw)
                self.stdout = io.StringIO('Fatal Python error: Segmentation fault\n')
            def wait(self,**kw): return -11
        with patch('src.caption_extraction.pipeline.extract_caption.subprocess.Popen',side_effect=lambda *a,**kw: Crash(**kw)):
            with self.assertRaisesRegex(RuntimeError,'-11'):
                predict_speed_in_worker({},videos,logging.getLogger("test"))

    def test_simple_decoder_is_gpu_only_and_does_not_construct_threaded(self):
        calls = []
        class Simple:
            def __init__(self, **kwargs): calls.append(kwargs)
            def get_stream_metadata(self): return types.SimpleNamespace(num_frames=150)
        def forbidden(**kwargs): self.fail("ThreadedDecoder must not be called")
        codec = types.ModuleType("PyNvVideoCodec")
        codec.SimpleDecoder = Simple
        codec.ThreadedDecoder = forbidden
        codec.OutputColorType = types.SimpleNamespace(RGBP="RGBP")
        with patch.dict(sys.modules, {"PyNvVideoCodec":codec}), \
             patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.current_device", return_value=0), \
             patch("torch.cuda.synchronize"):
            with CudaVideoReader(Path("video.mp4"), backend="simple") as reader:
                self.assertEqual(reader.total,150)
        self.assertTrue(calls[0]["use_device_memory"])
        self.assertEqual(calls[0]["cuda_context"],0)
        self.assertEqual(calls[0]["output_color_type"],"RGBP")

    def predictor(self):
        p = object.__new__(ShotSpeedPredictor)
        p.count, p.sample_fps, p.span, p.recursion = 16, 7.5, 16/7.5, 3
        p.logger = logging.getLogger("speed-test")
        p.cfg = {"timelapse_threshold": 1.8, "slow_motion_threshold": .6}
        return p

    def test_upstream_window_geometry_and_actual_fps(self):
        for fps in [25., 30., 60., 30000/1001]:
            for length in [1., 2.1334, 5., 7., 10.]:
                n, span = int(length*fps), 16/7.5
                # Independent transcription of dataset.SlidingWindowDataset.
                count = math.ceil((n/fps-span)//span)+1 if n/fps >= span else 0
                expected = []
                for i in range(count):
                    clip_length = int(span*fps)
                    start = int(span*i*fps) if i < count-1 else n-clip_length
                    expected.append((start, start+clip_length))
                self.assertEqual(original_window_slices(n, fps, span), expected)
        self.assertEqual(original_window_slices(150, 30, 16/7.5), [(0,64),(86,150)])

    def test_three_pass_correction_returns_last_score_and_16_frames(self):
        p = self.predictor()
        outputs = iter([.5, .5, .96, .98])
        calls = []
        def model(x):
            self.assertEqual(tuple(x.shape), (1, 3, 16, 1, 1))
            calls.append(x[0,0,:,0,0].tolist())
            return torch.tensor(math.log(next(outputs)), dtype=torch.float64)
        p.model = model
        frames = torch.arange(150.).view(150,1,1,1).expand(-1,3,1,1)
        result = p._estimate(frames, 30., "test")
        self.assertAlmostEqual(result, .98/(1/.48))
        self.assertEqual(calls[0], torch.linspace(0,63,16).long().float().tolist())
        self.assertEqual(calls[1][0], 86)
        self.assertEqual(calls[2][0], 22)
        self.assertEqual(calls[3][0], 17)

    def test_short_shot_is_unknown_without_model_or_added_prefix(self):
        p = self.predictor()
        p.model = lambda _: self.fail("Short shot must not invoke model")
        score = p._estimate(torch.zeros(30,3,1,1), 30., "short")
        self.assertIsNone(score)
        self.assertEqual(speed_label(score, p.cfg), "unknown")
        self.assertEqual(apply_speed_prefix("The video segment shows a tree.", "unknown"),
                         "The video segment shows a tree.")

    def test_factor_never_expands_past_shot_for_slow_predictions(self):
        p = self.predictor()
        p.model = lambda _: torch.tensor(math.log(.1), dtype=torch.float64)
        result = p._estimate(torch.zeros(150,3,1,1), 30., "clamp")
        self.assertAlmostEqual(result, .1/(5/p.span-1e-9))

    def test_threshold_boundaries_and_no_merge_across_labels(self):
        cfg = self.predictor().cfg
        self.assertEqual([speed_label(x,cfg) for x in [.5999,.6,.6001,1.8,1.8001]],
                         ["slow-motion","normal","normal","normal","timelapse"])
        records = [dict(video_id="v",start=0,end=5,caption=apply_speed_prefix("The video segment shows a tree.","normal")),
                   dict(video_id="v",start=5,end=10,caption=apply_speed_prefix("The video segment shows a tree.","timelapse"))]
        self.assertEqual(len(merge_duplicate_captions(records,.25,.8)),2)

    def test_gpu_stream_partition_with_gaps_and_shared_decode_batch(self):
        class Reader:
            total = 300
            def __init__(self,*a,**kw): pass
            def __enter__(self): return self
            def __exit__(self,*a): pass
            def __iter__(self):
                for start in range(0,300,8):
                    batch = torch.arange(start,min(start+8,300),dtype=torch.uint8)
                    yield start,batch.view(-1,1,1,1).expand(-1,3,2,2)
        p = self.predictor()
        p.size, p.mean, p.std = 2, torch.zeros(1,3,1,1), torch.ones(1,3,1,1)
        seen = []
        def estimate(frames,fps,context):
            seen.append(frames[:,0,0,0].mul(255).round().long().tolist())
            return 1.
        p._estimate = estimate
        windows = [dict(start=0,end=2.5),dict(start=2.5,end=5),dict(start=6,end=7)]
        with patch("src.caption_extraction.pipeline.speed.CudaVideoReader",Reader):
            self.assertEqual(p.predict(Path("fake.mp4"),windows,30.),["normal"]*3)
        self.assertEqual(seen,[list(range(75)),list(range(75,150)),list(range(180,210))])

    def test_omni_live_progress_and_restore_after_failure(self):
        engine = types.SimpleNamespace()
        engine.split_videos = lambda *a,**kw: [[None,0,0,0,30,30],[None,0,20,25,50,30]]
        engine._decode_video = lambda *a,**kw: list(range(50))
        original_split, original_decode = engine.split_videos, engine._decode_video
        class Detector:
            def inference(self,path,mode,overlap):
                self.mode = mode
                engine._decode_video(path)
                for i,_ in enumerate(engine.split_videos(None)):
                    if i == 1: raise RuntimeError("intentional failure")
        detector = Detector()
        package = types.ModuleType("omnishotcut")
        package.engine = engine
        with patch.dict(sys.modules,{"omnishotcut":package}):
            with self.assertLogs("omni-test",level="INFO") as log:
                with self.assertRaisesRegex(RuntimeError,"intentional"):
                    inference_with_progress(detector,"test.mp4",mode="clean_shot",fps=30,
                                            logger=logging.getLogger("omni-test"))
        self.assertEqual(detector.mode,"clean_shot")
        self.assertIn("window=1/2 DONE", "\n".join(log.output))
        self.assertIn("window=2/2 START", "\n".join(log.output))
        self.assertIs(engine.split_videos,original_split)
        self.assertIs(engine._decode_video,original_decode)


if __name__ == "__main__":
    unittest.main()
