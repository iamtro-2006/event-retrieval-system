import csv
import json
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import faiss
import numpy as np

from src.retrieval.segment_data import build_maps, load_segments, SegmentMaps, choose_asr_fallback
from src.retrieval.hierarchical import HierarchicalSearch, weighted_scores
from src.retrieval.retriever.common.video_filter import with_video_filter


class MappedSearchTests(unittest.TestCase):
    def test_csv_roundtrip_boundaries_empty_reverse_and_stale(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / 'map_keyframes/V3C'
            folder.mkdir(parents=True)
            (folder/'00001.csv').write_text('keyframe_id,timestamp_sec\n000001,1\n000002,2\n000003,3\n')
            for source, field in [('asr','transcript'), ('captions','caption')]:
                target = root/source/'V3C'
                target.mkdir(parents=True)
                (target/'00001.json').write_text(json.dumps([
                    {'start':1,'end':2,field:'first'}, {'start':2,'end':4,field:'second'},
                    {'start':8,'end':9,field:'empty'}, {'start':0,'end':.5,field:'before first frame'}]))
            self.assertEqual(build_maps(root,'v3c'), {'asr':4,'caption':4})
            records = list(load_segments(root,'v3c','caption'))
            links = SegmentMaps(root,'v3c','caption',records)
            self.assertEqual(links.forward[records[0]['segment_id']], [('00001',1)])
            self.assertEqual(links.reverse[('00001',2)], [records[1]['segment_id']])
            self.assertEqual(links.forward[records[2]['segment_id']], [])
            asr_records = list(load_segments(root,'v3c','asr'))
            asr_links = SegmentMaps(root,'v3c','asr',asr_records)
            self.assertEqual(asr_links.forward[asr_records[2]['segment_id']], [('00001',3)])
            self.assertEqual(asr_links.forward[asr_records[3]['segment_id']], [('00001',1)])
            self.assertEqual(asr_links.relations[asr_records[2]['segment_id']][('00001',3)],
                             {'relation':'previous','offset_seconds':5.0})
            self.assertEqual(asr_links.reverse[('00001',3)], [asr_records[1]['segment_id'], asr_records[2]['segment_id']])
            with (root/'map_data/map_caption/V3C/00001.csv').open() as stream:
                self.assertEqual(next(csv.DictReader(stream))['keyframe_id'], '000001')
            records[0]['text'] = 'changed'
            with self.assertRaisesRegex(RuntimeError,'Stale'):
                SegmentMaps(root,'v3c','caption',records)

    def engine(self):
        engine = HierarchicalSearch.__new__(HierarchicalSearch)
        vectors = np.array([[1.,0.],[0.,1.]], dtype='float32')
        index = faiss.IndexFlatIP(2)
        index.add(vectors)
        records = [dict(segment_id=f's{i}',video_id=f'v{i}',source='caption',speed='normal',text=f't{i}') for i in range(2)]
        maps = SimpleNamespace(forward={'s0':[('v0',1)],'s1':[('v1',2)]},
            reverse={('v0',1):['s0'],('v1',2):['s1']})
        engine.sources = {'caption':SimpleNamespace(index=index,vectors=vectors,records=records,
            maps=maps,by_id={'s0':0,'s1':1})}
        engine.encode = lambda source, queries: np.array([[0.,1.] for _ in queries], dtype='float32')
        engine.frame = lambda key, model_key=None: dict(video_id=key[0],keyframe_id=str(key[1]))
        engine.visual_scores = lambda rows,*a: (rows,np.array([.95 if r['video_id']=='v0' else .6 for r in rows]))
        return engine

    def test_log_fallback_missing_sides_left_bias_and_far_left(self):
        def select(times, start=10., end=12.):
            return choose_asr_fallback(times, [(t,str(i)) for i,t in enumerate(times)], start, end)
        self.assertEqual(select([4.,16.])[2], 'previous')  # left 6s, right 4s
        self.assertEqual(select([16.])[2], 'next')  # missing left is infinity
        self.assertEqual(select([4.])[2], 'previous')
        self.assertIsNone(select([]))
        self.assertEqual(select([2171.88,2397.72],2394.03,2395.93)[2], 'next')

    def test_top_down_visual_rank_only_and_filter(self):
        engine = self.engine()
        result = engine.search(query='context', sequence_top_k=2)
        self.assertEqual(result.video_id.tolist(), ['v0','v1'])
        self.assertNotIn('matched_sequence', result.columns)
        self.assertEqual(result['rank'].tolist(), [1,2])
        filtered = with_video_filter(['v1'], engine.search, query='context')
        self.assertEqual(filtered.video_id.tolist(), ['v1'])

    def test_bottom_up_context_can_reverse_visual_order(self):
        engine = self.engine()
        rows = [engine.frame(('v0',1)),engine.frame(('v1',2))]
        result = engine.search(query='context', direction='bottom_up', initial_frames=rows, visual_weight=.1)
        self.assertEqual(result.video_id.tolist(), ['v1','v0'])
        visual = engine.search(query='context', direction='bottom_up', initial_frames=rows, visual_weight=1)
        self.assertEqual(visual.video_id.tolist(), ['v0','v1'])

    def test_and_and_invalid_weights(self):
        matrix = np.array([[1., .4],[-1., .4]])
        self.assertGreater(weighted_scores(matrix,operation='and')[1], weighted_scores(matrix,operation='and')[0])
        for weights in ([1],[-1,2],[0,0],[float('nan'),1]):
            with self.assertRaises(ValueError):
                weighted_scores(matrix,weights)


if __name__ == '__main__':
    unittest.main()
