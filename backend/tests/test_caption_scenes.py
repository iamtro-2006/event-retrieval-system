import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import yaml

from src.caption_extraction.pipeline.extract_caption import ExtractCaptionPipeline


class CaptionSceneTests(unittest.TestCase):
    def pipeline(self, root, profile='16x448'):
        cfg = yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/caption_extraction.yaml').read_text())
        cfg['extraction']['caption_profile'] = profile
        cfg['extraction']['caption_min_color_bins'] = 1
        return ExtractCaptionPipeline(cfg, root, root, root/'out', logging.getLogger('test'))

    def test_whole_scenes_preserve_long_and_subsecond_intervals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'video.scenes.txt').write_text('0 0\n1 5000\n')
            pipeline = self.pipeline(root)
            with patch('src.caption_extraction.pipeline.extract_caption._probe_video', return_value=(25., 300.)):
                self.assertEqual(pipeline._windows(root/'video.mp4'),
                    [{'start':0.,'end':.04}, {'start':.04,'end':200.04}])

    def test_scene_reuse_does_not_overwrite_and_missing_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pipeline = self.pipeline(root)
            with self.assertRaises(FileNotFoundError):
                pipeline.prepare_shots([root/'video.mp4'])
            scene = root/'video.scenes.txt'
            scene.write_text('0 5000\n')
            pipeline.prepare_shots([root/'video.mp4'], overwrite=True)
            self.assertEqual(scene.read_text(), '0 5000\n')

    def test_profiles_sample_full_scene_at_448(self):
        class Reader:
            def get_avg_fps(self): return 25.
            def __len__(self): return 10000
            def get_batch(self, ids):
                return SimpleNamespace(asnumpy=lambda: np.zeros((len(ids),8,8,3), dtype=np.uint8))
        with tempfile.TemporaryDirectory() as tmp:
            for maximum in (12,16,24):
                pipeline = self.pipeline(Path(tmp), f'{maximum}x448')
                pipeline.captioner = SimpleNamespace(image_size=448)
                images, times = pipeline._sample_frames(Reader(), 0., 200.)
                self.assertEqual(len(images), maximum)
                self.assertEqual(times[0], 0.)
                self.assertEqual(times[-1], 199.96)
                self.assertTrue(all(image.size == (448,448) for image in images))
                if maximum == 24:
                    self.assertGreaterEqual(pipeline.vllm_config['max_model_len'], 8192)


if __name__ == '__main__':
    unittest.main()
