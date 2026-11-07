import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from threading import RLock
from types import SimpleNamespace

import faiss
import numpy as np
from pydantic import ValidationError

from src.api.routers.plans import SearchPlan, ExecuteRequest
from src.api.routers import plans
import test_mapped_search
from src.retrieval.search_scope import resolve_scope, pool_frame_keys, SCOPE_MODES
from src.retrieval.plan_search import search_plan
from src.retrieval.retriever.common.candidate_filter import (
    with_video_filter, with_frame_filter, filtered_faiss_search, frame_keys_context,
)


class ScopeTests(unittest.TestCase):
    def setUp(self):
        self.session = {'dataset': 'v3c', 'plans': [
            {'id': 'p', 'status': 'completed', 'records': [
                {'video_id': 'v1', 'keyframe_id': '1'},
                {'video_id': 'v2', 'keyframe_id': '9', 'matched_sequence': [
                    {'video_id': 'v2', 'keyframe_id': '2'}]}]},
            {'id': 'empty', 'status': 'completed', 'records': []},
            {'id': 'pending', 'status': 'queued'}]}

    def plan(self, **kwargs):
        return SearchPlan(query='q', **kwargs)

    def test_scope_intersection_and_parent_is_lineage_only(self):
        self.assertIsNone(resolve_scope(self.plan(parent_pool_id='p'), self.session).videos)
        scope = resolve_scope(self.plan(scope='videos', video_ids=['v1', 'v3']), self.session, ['v2'])
        self.assertTrue(scope.empty)
        # Empty scopes never invoke any retrieval backend.
        self.assertTrue(search_plan(self.plan(), None, None, scope).empty)

    def test_pool_sequence_uses_mapped_frames_only(self):
        keys = pool_frame_keys(self.session['plans'][0])
        self.assertEqual(keys, {('v1', 1), ('v2', 2)})
        scope = resolve_scope(self.plan(scope='pool_frames', scope_pool_id='p'), self.session, ['v2'])
        self.assertEqual(scope.frames, {('v2', 2)})

    def test_empty_and_invalid_pool_never_fall_back(self):
        for scope in ('pool_frames', 'pool_videos'):
            self.assertTrue(resolve_scope(self.plan(scope=scope, scope_pool_id='empty'), self.session).empty)
            for pool in ('unknown', 'pending'):
                with self.assertRaises(ValueError):
                    resolve_scope(self.plan(scope=scope, scope_pool_id=pool), self.session)
        with self.assertRaises(ValueError):
            resolve_scope(self.plan(scope='pool_frames', scope_pool_id='p'))

    def test_legacy_mapping_and_conflicts(self):
        legacy = self.plan(mode='hierarchical', direction='bottom_up', parent_pool_id='p', restrict_to_parent=True)
        self.assertEqual((legacy.scope, legacy.scope_pool_id), ('pool_frames', 'p'))
        self.assertEqual(SearchPlan.model_validate(legacy.model_dump()).scope, 'pool_frames')
        self.assertEqual(ExecuteRequest(dataset='v3c', query='q', video_ids=['v1']).scope, 'videos')
        self.assertEqual(ExecuteRequest(dataset='v3c', query='q', video_ids=[]).scope, 'dataset')
        for args in (
            {'scope': 'videos'}, {'scope': 'videos', 'video_ids': []},
            {'scope': 'dataset', 'video_ids': ['v1']},
            {'scope': 'dataset', 'scope_pool_id': 'p'},
            {'scope': 'pool_frames'},
            {'restrict_to_parent': True, 'parent_pool_id': 'p', 'scope': 'dataset'},
        ):
            with self.assertRaises(ValidationError):
                self.plan(**args)

    def test_advertised_pool_modes_are_enforced(self):
        for mode in ('semantic', 'hierarchical', 'asr', 'caption', 'ocr', 'temporal'):
            if mode in SCOPE_MODES['pool_frames']:
                self.plan(mode=mode, scope='pool_frames', scope_pool_id='p')
            else:
                with self.assertRaises(ValidationError):
                    self.plan(mode=mode, scope='pool_frames', scope_pool_id='p')

    def test_subset_search_finds_candidates_outside_global_top_k(self):
        index = faiss.IndexFlatIP(2)
        index.add(np.array([[1, 0], [.2, 0], [.8, 0]], dtype='float32'))
        rows = [dict(video_id=v, keyframe_id=f) for v, f in [('v0', 0), ('v1', 1), ('v1', 3)]]
        query = np.array([[1, 0]], dtype='float32')
        def search(scope):
            return with_video_filter(scope.videos, lambda: with_frame_filter(scope.frames,
                lambda: filtered_faiss_search(index, RLock(), query, 1, rows)))[1][0].tolist()
        frames = resolve_scope(self.plan(scope='pool_frames', scope_pool_id='p'), self.session, ['v1'])
        videos = resolve_scope(self.plan(scope='pool_videos', scope_pool_id='p'), self.session, ['v1'])
        self.assertEqual(search(frames), [1])
        self.assertEqual(search(videos), [2])  # New frame from same video.
        self.assertIsNone(frame_keys_context.get())

    def test_context_restored_on_failure(self):
        def fail():
            raise RuntimeError('failure')
        with self.assertRaises(RuntimeError):
            with_frame_filter(frozenset(), fail)
        self.assertIsNone(frame_keys_context.get())

    def test_hierarchical_directions_keep_frame_scope_and_scoring(self):
        engine = test_mapped_search.MappedSearchTests().engine()
        scope = frozenset({('v0', 1)})
        rows = [engine.frame(('v0', 1)), engine.frame(('v1', 2))]
        top = with_frame_filter(scope, engine.search, query='context')
        bottom = with_frame_filter(scope, engine.search, query='context',
            direction='bottom_up', initial_frames=rows)
        self.assertEqual(top.video_id.tolist(), ['v0'])
        self.assertEqual(bottom.video_id.tolist(), ['v0'])
        self.assertAlmostEqual(top.retrieval_score.iloc[0], .95)
        self.assertAlmostEqual(bottom.retrieval_score.iloc[0], .7 * .95 + .3 * .5)

    def test_capabilities_and_registration_validate_pool_references(self):
        system = SimpleNamespace(text_search_health=lambda: {}, available_models=lambda: ['qwen'])
        session = {**self.session, 'id': 'scope-test', 'video_ids': ['v1'], 'config': {}}
        session['plans'] = [{**p, 'plan': {'label': p['id']}} for p in session['plans']]
        with tempfile.TemporaryDirectory() as tmp, patch.object(plans, 'ROOT', Path(tmp)):
            plans.save(session)
            tools = plans.RealTools(session, system, SimpleNamespace(), [])
            capabilities, _ = tools.execute('list_capabilities', {})
            self.assertEqual(capabilities['current_scope']['video_ids'], ['v1'])
            self.assertEqual(capabilities['scope_modes']['pool_frames'], ['semantic'])
            self.assertEqual(len(capabilities['referenceable_pools']), 2)
            with self.assertRaises(ValueError):
                tools.execute('register_plans', {'plans': [dict(query='q', scope='pool_videos', scope_pool_id='other')]})
            self.assertEqual(len(session['plans']), 3)


if __name__ == '__main__':
    unittest.main()
