"""Run with: python -m unittest src.retrieval.retriever.common.test_video_filter."""
import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, RLock

import numpy as np

from src.retrieval.retriever.common.video_filter import (
    filtered_faiss_search, video_ids_context, with_video_filter,
)


class RankedIndex:
    ntotal = 160

    def __init__(self):
        self.calls = []

    def search(self, embedding, k):
        self.calls.append(k)
        ids = np.arange(self.ntotal)
        if embedding[0, 0] < 0:
            ids = ids[::-1]
        return (1 - np.arange(k, dtype=np.float32) / self.ntotal)[None, :], ids[:k][None, :]


class VideoFilterTests(unittest.TestCase):
    def setUp(self):
        self.index = RankedIndex()
        self.metadata = [{"video_id": "selected" if i >= 140 else "other"} for i in range(160)]

    def search(self, ids, k=3):
        return with_video_filter(ids, filtered_faiss_search, self.index, RLock(), np.array([[1.], [-1.]], np.float32), k, self.metadata)

    def test_selected_videos_below_global_top_k_are_retrieved(self):
        _, rows = self.search(["selected"])
        np.testing.assert_array_equal(rows, [[140, 141, 142], [159, 158, 157]])
        self.assertIn(160, self.index.calls)

    def test_fewer_eligible_frames_than_top_k(self):
        _, rows = self.search(["selected"], 500)
        self.assertEqual(rows.shape, (2, 20))
        self.assertTrue(np.all(rows >= 140))

    def test_unknown_id_returns_empty_without_global_fallback(self):
        _, rows = self.search(["missing"])
        self.assertEqual(rows.shape, (2, 0))
        self.assertEqual(self.index.calls, [])

    def test_empty_cache_disables_filter(self):
        self.assertEqual(with_video_filter([], video_ids_context.get), frozenset())

    def test_scope_restored_after_failure(self):
        def fail():
            raise RuntimeError("failed query")
        with self.assertRaises(RuntimeError):
            with_video_filter(["selected"], fail)
        self.assertFalse(video_ids_context.get())

    def test_concurrent_filters_are_isolated(self):
        barrier = Barrier(2)
        def run(value):
            def read():
                barrier.wait(timeout=5)
                return video_ids_context.get()
            return with_video_filter([value], read)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run, value) for value in ["first", "second"]]
            self.assertEqual([f.result() for f in futures], [frozenset(["first"]), frozenset(["second"])])


if __name__ == "__main__":
    unittest.main()
