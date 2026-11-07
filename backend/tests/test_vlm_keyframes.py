"""Merged VLM/V3C selection, notebook packaging and persistent worker regressions."""
import ast
import base64
import importlib.util
import io
import json
import logging
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

import numpy as np
import torch

from src.keyframe_extraction.model_cache import KeyframeModels
from src.keyframe_extraction.models.p3 import Candidate, aucc_score, cosine_dedup, select_sampled_p3
from src.keyframe_extraction.models.quality import score_batch
from src.keyframe_extraction.worker_session import VlmWorkerSession
from src.utils.config import load_config
from src.utils.frame_sampling import build_sample_plan

ROOT = Path(__file__).resolve().parents[1]


class VlmP3Tests(unittest.TestCase):
    def test_legacy_sampling_and_decode_paths_are_rejected_before_loading_models(self):
        from src.keyframe_extraction.pipeline.extract_keyframes import KeyframeExtractionPipeline

        for field, value, message in [('sample_fps', None, 'positive'),
                                      ('backend', 'decord', 'NVDEC')]:
            with self.subTest(field=field):
                cfg = load_config(ROOT/'configs/kf_extraction.yaml')
                section = cfg.embedding if field == 'sample_fps' else cfg.frame_loader
                setattr(section, field, value)
                with patch('src.keyframe_extraction.model_cache.KeyframeModels.embedding') as load:
                    with self.assertRaisesRegex(ValueError, message):
                        KeyframeExtractionPipeline(cfg, logging.getLogger('test')).run([Path('test.mp4')])
                    load.assert_not_called()

    def test_clean_shot_matches_upstream_label_filter(self):
        from src.keyframe_extraction.models.omni_detector import clean_shot_ranges
        ranges = [[0, 10], [10, 15], [15, 30]]
        self.assertEqual(clean_shot_ranges(ranges, [0, 1, 0], 0), [[0, 10], [15, 30]])
        self.assertEqual(clean_shot_ranges(ranges, [1, 2, 1], 0), [])

    def test_timestamp_regions_are_30_seconds(self):
        times = np.arange(1650) / 30.
        plan = build_sample_plan(np.array([[0, 1649]]), times, 30.)
        self.assertEqual(len(plan['sample_ranges']), 2)
        self.assertEqual(len(plan['frame_indexes']), 165)
        first, last = plan['sample_ranges'][0]
        self.assertLess(plan['timestamps'][last] - plan['timestamps'][first], 30.)

    def test_aucc_matches_pairwise_with_ties(self):
        distance = np.array([[0, .2, .2, .8], [.2, 0, .5, .8],
                             [.2, .5, 0, .8], [.8, .8, .8, 0]])
        labels = np.array([0, 0, 1, 1])
        a, b = np.triu_indices(4, 1)
        same = labels[a] == labels[b]
        positive, negative = distance[a, b][same], distance[a, b][~same]
        expected = np.mean(positive[:, None] < negative) + .5 * np.mean(positive[:, None] == negative)
        self.assertAlmostEqual(aucc_score(distance, labels), expected)

    def test_no_dedup_within_same_shot_even_across_sampling_regions(self):
        def item(fid, vector, quality):
            return Candidate(0, fid, float(fid), 'test', quality=quality,
                             feature=np.array(vector, dtype=np.float64))
        first = item(0, [1., 0.], .8)
        boundary = item(1000, [.96, .28], .7)
        duplicate = item(2000, [1., 0.], .9)
        duplicate.scene_id = 2
        self.assertEqual(cosine_dedup([first, boundary, duplicate], [[0, 2999]], .96), [0, 1000, 2000])

    def test_adjacent_shots_dedup_and_reject_outside_candidates(self):
        selected = [Candidate(0, 1, 0., 'test', quality=.4, feature=np.array([1., 0.])),
                    Candidate(0, 10, 1., 'test', quality=.9, feature=np.array([1., 0.]))]
        self.assertEqual(cosine_dedup(selected, [[0, 4], [8, 15]], .96), [10])
        with self.assertRaisesRegex(ValueError, 'outside'):
            cosine_dedup(selected, [[0, 4], [11, 15]], .96)

    def test_nonadjacent_shots_survive_and_threshold_is_strict(self):
        selected = [Candidate(i, i * 10, float(i), 'test', quality=1. - i * .1,
                              feature=np.array([1., 0.])) for i in range(3)]
        shots = [[0, 9], [10, 19], [20, 29]]
        self.assertEqual(cosine_dedup(selected, shots, .96), [0, 20])
        selected[1].feature = np.array([.96, .28])
        self.assertEqual(cosine_dedup(selected[:2], shots, .96), [0, 10])

    def test_p3_preserves_common_for_each_region(self):
        plan = dict(frame_indexes=np.array([0, 10, 900, 910]),
                    timestamps=np.array([0., 1/3, 30., 30+1/3]),
                    sample_ranges=np.array([[0, 1], [2, 3]]))
        features = np.array([[1., 0.], [1., 0.], [0., 1.], [0., 1.]], dtype=np.float32)
        selected, _ = select_sampled_p3(plan, features, np.ones(4))
        self.assertEqual([c.frame_id for c in selected], [0, 900])

    def test_quality_rejects_uniform_and_accepts_texture(self):
        torch.manual_seed(42)
        frames = torch.randint(20, 235, (2, 3, 48, 64), dtype=torch.uint8)
        frames[0].fill_(128)
        valid, quality, nonflat, sharp = score_batch(frames)
        self.assertEqual(nonflat.tolist(), [False, True])
        self.assertEqual(valid.tolist(), [False, True])
        self.assertTrue(np.isfinite(quality).all())
        self.assertGreater(sharp[1], sharp[0])

    def test_uniform_shot_dropped_and_nonflat_fallback(self):
        from src.embedding_extraction.models import gpu_encoder as encoder
        class Reader:
            total = 6
            def __init__(self, *args): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def __iter__(self):
                yield 0, torch.zeros((6, 3, 8, 8), dtype=torch.uint8)
        class Model:
            def encode_image(self, images):
                return torch.ones((len(images), 1024))
        plan = dict(frame_indexes=np.arange(6), timestamps=np.arange(6, dtype=float),
                    sample_ranges=np.array([[0, 1], [2, 3], [4, 5]]), source_total_frames=6)
        decisions = (np.array([0, 0, 0, 0, 1, 1], bool),
                     np.array([0, 0, .2, .4, .8, .9]),
                     np.array([0, 0, 1, 1, 1, 1], bool), np.arange(6, dtype=float))
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(encoder, 'CudaVideoReader', Reader), \
                patch.object(encoder, 'pe_preprocess_cuda', lambda x, size: x.float()), \
                patch('src.keyframe_extraction.models.quality.score_batch', return_value=decisions):
            result = encoder.encode_sampled_with_quality(Model(), Path('test.mp4'),
                Path(tmp)/'features.pkl', 2, 8, plan, np.array([[0, 1], [2, 3], [4, 5]]), 8, 5)
        self.assertEqual(result['frame_indexes'].tolist(), [3, 4, 5])
        self.assertEqual(result['kept_scenes'].tolist(), [[2, 3], [4, 5]])
        np.testing.assert_allclose(result['quality'], [.4, .8, .9])

    def test_scene_publication_follows_export_and_dedup(self):
        from src.keyframe_extraction.pipeline import extract_keyframes as pipeline
        from src.utils.config import load_config
        for fail in (False, True):
            with self.subTest(export_fails=fail), tempfile.TemporaryDirectory() as tmp:
                cfg = load_config(ROOT/'configs/kf_extraction.yaml')
                cfg.paths.output_dir = Path(tmp)
                cfg.embedding.skip_existing = False
                scene = Path(tmp)/'scenes/LHE/test.scenes.txt'
                raw_shots = np.array([[0, 1], [2, 3], [4, 5]])
                def encode(*args, **kwargs):
                    self.assertFalse(scene.exists())
                    vector = np.zeros((2, 1024), dtype=np.float32)
                    vector[:, 0] = 1.
                    return dict(features=vector, frame_indexes=np.array([2, 4]),
                        timestamps=np.array([2., 4.]), sample_ranges=np.array([[0, 0], [1, 1]]),
                        source_total_frames=6, source_fps=1., quality=np.array([.4, .9]),
                        kept_scenes=raw_shots[1:])
                def export(*args):
                    self.assertFalse(scene.exists())
                    if fail: raise RuntimeError('write failed')
                with patch.object(pipeline, 'detect_omni_scenes', return_value=raw_shots), \
                        patch('src.utils.frame_sampling.probe_video_timeline', return_value=(np.arange(6.), 1.)), \
                        patch.object(pipeline, 'load_clip_model', return_value=(None, None, None)), \
                        patch.object(pipeline, 'encode_video_frames', side_effect=encode), \
                        patch.object(pipeline, 'save_keyframe_map'), \
                        patch('src.keyframe_extraction.models.export.export_selected_cuda', side_effect=export):
                    task = pipeline.KeyframeExtractionPipeline(cfg, logging.getLogger('test'))
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, 'write failed'):
                            task.run([Path('test.mp4')], Path('LHE'))
                        self.assertFalse(scene.exists())
                    else:
                        task.run([Path('test.mp4')], Path('LHE'))
                        self.assertEqual(scene.read_text().splitlines(), ['4 5'])

    def test_notebook_bundle_matches_repository(self):
        nb = json.loads((ROOT/'notebooks/vlm_keyframes.ipynb').read_text(encoding='utf-8'))
        for cell in nb['cells']:
            if cell['cell_type'] == 'code': ast.parse(''.join(cell['source']))
        self.assertEqual(''.join(nb['cells'][9]['source']),
                         (ROOT/'scripts/keyframe_extraction/vlm_run_cell.py').read_text(encoding='utf-8'))
        source = ''.join(nb['cells'][7]['source'])
        tree = ast.parse(''.join(nb['cells'][5]['source']))
        batch = next(ast.literal_eval(node.value) for node in tree.body
                     if isinstance(node, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'BATCH_SOURCE' for t in node.targets))
        self.assertEqual(batch, (ROOT/'src/keyframe_extraction/remote_jobs.py').read_text(encoding='utf-8'))
        data = base64.b64decode(re.search(r"SOURCE_ZIP = '([^']+)'", source)[1])
        with zipfile.ZipFile(io.BytesIO(data)) as bundle:
            from scripts.keyframe_extraction.build_vlm_keyframes import SOURCES
            self.assertEqual(set(bundle.namelist()), set(SOURCES) | {'vlm_worker.py'})
            for name in bundle.namelist():
                if (ROOT/name).is_file():
                    self.assertEqual(bundle.read(name), (ROOT/name).read_bytes(), name)
            self.assertIn(b'vlm-p3-quality-v2', bundle.read('vlm_worker.py'))
            self.assertEqual(bundle.read('vlm_worker.py').decode(),
                             (ROOT/'scripts/keyframe_extraction/vlm_worker.py').read_text(encoding='utf-8'))



class ResidentModelTests(unittest.TestCase):
    def test_omni_detection_reuses_loaded_detector_across_videos(self):
        from src.keyframe_extraction.models import omni_detector as omni
        models = KeyframeModels()
        detector = object()
        codec = SimpleNamespace(engine=object())
        labels = SimpleNamespace(unique_intra_label_mapping=dict(general=0))
        cfg = load_config(ROOT/'configs/kf_extraction.yaml').omni
        with tempfile.TemporaryDirectory() as tmp, \
                patch.dict(sys.modules, {'omnishotcut': codec, 'omnishotcut.label_correspondence': labels}), \
                patch.object(omni, 'load_omni_detector', return_value=detector) as load, \
                patch('src.utils.omni_progress.inference_with_progress', return_value=[[0, 2]]) as infer, \
                patch('torch.cuda.empty_cache') as empty:
            for i in range(2):
                result = omni.detect_omni_scenes(Path(f'{i}.mp4'), Path(tmp)/f'{i}.txt',
                                                  cfg, backend='decord', model_cache=models)
                np.testing.assert_array_equal(result, [[0, 1]])
            load.assert_called_once_with(cfg)
            self.assertEqual(infer.call_count, 2)
            self.assertTrue(all(call.args[0] is detector for call in infer.call_args_list))
            empty.assert_not_called()

    def test_two_job_pipelines_share_both_models_and_release_at_session_end(self):
        from src.keyframe_extraction.pipeline import extract_keyframes as pipeline
        models = KeyframeModels()
        omni, embedding = object(), object()
        omni_loader = Mock(return_value=omni)
        embedding_loader = Mock(return_value=(embedding, None, None))
        shots = np.array([[0, 1]])

        def detect(video, output, cfg, backend, model_cache):
            self.assertIs(model_cache.omni(cfg, omni_loader), omni)
            return shots

        def encode(model, *args, **kwargs):
            self.assertIs(model, embedding)
            # Loading embeddings must leave Omni resident at the same time.
            self.assertIs(models._omni, omni)
            features = np.zeros((2, 1024), dtype=np.float32)
            features[:, 0] = 1
            return dict(features=features, frame_indexes=np.array([0, 1]),
                        timestamps=np.array([0., 1.]), sample_ranges=np.array([[0, 1]]),
                        source_total_frames=2, source_fps=1., quality=np.array([.4, .9]),
                        kept_scenes=shots)

        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(pipeline, 'detect_omni_scenes', side_effect=detect), \
                patch.object(pipeline, 'load_clip_model', embedding_loader), \
                patch.object(pipeline, 'encode_video_frames', side_effect=encode), \
                patch('src.utils.frame_sampling.probe_video_timeline', return_value=(np.array([0., 1.]), 1.)), \
                patch('src.keyframe_extraction.models.export.export_selected_cuda'), \
                patch('torch.cuda.empty_cache') as empty:
            for video_id in ('00128', '00129'):
                cfg = load_config(ROOT/'configs/kf_extraction.yaml')
                cfg.paths.output_dir = Path(tmp)/video_id/'output'
                task = pipeline.KeyframeExtractionPipeline(cfg, logging.getLogger('resident-test'), models)
                task.run([Path(video_id + '.mp4')], Path('V3C'))
                self.assertTrue((cfg.paths.output_dir/'map_keyframes/V3C'/f'{video_id}.csv').is_file())
            omni_loader.assert_called_once()
            embedding_loader.assert_called_once()
            empty.assert_not_called()
            with patch('torch.cuda.is_available', return_value=True):
                models.close()
            empty.assert_called_once()
            self.assertIsNone(models._omni)
            self.assertIsNone(models._embedding)

    def test_changed_model_configuration_is_not_silently_reused(self):
        cfg = load_config(ROOT/'configs/kf_extraction.yaml')
        models = KeyframeModels()
        loader = Mock(return_value=(object(), None, None))
        models.embedding(cfg.embedding, loader, None)
        cfg.embedding.precision = 'fp32'
        with self.assertRaisesRegex(ValueError, 'configuration changed'):
            models.embedding(cfg.embedding, loader, None)
        loader.assert_called_once()



class WorkerProtocolTests(unittest.TestCase):
    def test_two_requests_share_one_process_and_stream_logs(self):
        source = '''import json, os, sys
for line in sys.stdin:
    request = json.loads(line)
    print('worker pid=' + str(os.getpid()), flush=True)
    print('__VLM_RESULT__=' + json.dumps(dict(config=request['config'], ok=True)), flush=True)
'''
        lines = []
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            script = path/'worker.py'
            script.write_text(source, encoding='utf-8')
            session = VlmWorkerSession([sys.executable, '-u', script], on_line=lines.append)
            with session:
                process = session.process
                for i in range(2):
                    session.extract(dict(video=str(i)), path/f'{i}.json')
                self.assertEqual(json.loads((path/'1.json').read_text()), dict(video='1'))
            self.assertEqual(len(lines), 2)
            self.assertEqual(lines[0], lines[1])
            self.assertEqual(process.returncode, 0)
            self.assertTrue(process.stdout.closed)

    def test_failure_or_native_crash_never_counts_as_completed(self):
        for statement, message in [
            ("print('__VLM_RESULT__=' + json.dumps(dict(config=request['config'], ok=False, error='decode failed')), flush=True)", 'decode failed'),
            ('sys.exit(7)', 'exit code 7'),
            ("print('__VLM_RESULT__=' + json.dumps(dict(config='wrong.json', ok=True)), flush=True)", 'wrong video'),
        ]:
            source = 'import json, sys\nrequest=json.loads(sys.stdin.readline())\n' + statement + '\n'
            with self.subTest(message=message), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)
                script = path/'worker.py'
                script.write_text(source, encoding='utf-8')
                session = VlmWorkerSession([sys.executable, '-u', script])
                with self.assertRaisesRegex(RuntimeError, message):
                    with session:
                        process = session.process
                        session.extract({}, path/'config.json')
                self.assertIsNotNone(process.poll())
                self.assertTrue(process.stdout.closed)

    def test_server_passes_same_models_and_stops_on_first_failed_job(self):
        spec = importlib.util.spec_from_file_location('vlm_worker', ROOT/'scripts/keyframe_extraction/vlm_worker.py')
        worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker)
        models = object()
        incoming = ''.join(json.dumps(dict(config=f'{i}.json')) + '\n' for i in range(3))
        output = io.StringIO()
        with patch.object(worker.sys, 'stdin', io.StringIO(incoming)), \
                patch.object(worker.sys, 'stdout', output), \
                patch.object(worker.traceback, 'print_exc'), \
                patch.object(worker, 'process_job', side_effect=[None, RuntimeError('bad frame')]) as job:
            self.assertEqual(worker.serve(models), 1)
        self.assertEqual(job.call_count, 2)
        self.assertTrue(all(call.args[1] is models for call in job.call_args_list))
        results = [json.loads(line.split('=', 1)[1]) for line in output.getvalue().splitlines()]
        self.assertEqual([result['ok'] for result in results], [True, False])



if __name__ == '__main__':
    unittest.main()
