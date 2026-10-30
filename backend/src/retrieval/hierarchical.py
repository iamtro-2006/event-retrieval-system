"""Mapped text retrieval and two-direction keyframe ranking. No visual inspection."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.retrieval.segment_data import (
    SegmentMaps, build_maps, fingerprint, load_indexable_segments, load_segments,
)
from src.retrieval.retriever.common.video_filter import video_ids_context

log = logging.getLogger(__name__)
QWEN_MODEL_ID = 'Qwen/Qwen3-VL-Embedding-2B'


def weighted_scores(matrix, weights=None, operation='weighted'):
    values = np.clip((np.asarray(matrix, dtype=float) + 1) / 2, 0, 1)
    w = np.asarray(weights if weights is not None else np.ones(len(values)), dtype=float)
    if w.shape != (len(values),) or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError('Query weights must be finite, nonnegative, and match query count')
    w /= w.sum()
    if operation == 'and':
        return np.exp((np.log(np.maximum(values, 1e-8)) * w[:, None]).sum(axis=0))
    return w @ values


class SegmentIndex:
    def __init__(self, root, dataset, source):
        import faiss
        folder = (root / 'database' / 'faiss_qwen3_vl_embedding_2b_vllm'
                  / source / dataset.upper())
        meta = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
        self.records = meta['records']
        source_records = list(load_segments(root, dataset, source))
        indexable_records = load_indexable_segments(root, dataset, source)
        if fingerprint(indexable_records) != meta['source_hash']:
            raise RuntimeError(f'Stale {source} index; rebuild segment data')
        expected_by_id = {record['segment_id']: record for record in indexable_records}
        indexed_ids = [record.get('segment_id') for record in self.records]
        if (any(segment_id not in expected_by_id for segment_id in indexed_ids)
                or self.records != [expected_by_id[segment_id] for segment_id in indexed_ids]
                or len(indexed_ids) != len(set(indexed_ids))):
            raise RuntimeError(f'{source} metadata contains invalid or duplicate segment records')
        self.model_name = meta['model']
        if meta.get('engine') != 'vllm' or self.model_name != QWEN_MODEL_ID:
            raise RuntimeError(f'{source} index must use {QWEN_MODEL_ID} via vLLM; rebuild segment data')
        self.index = faiss.read_index(str(folder / f'{source}.faiss'))
        self.vectors = np.load(folder / 'vectors_fp32.npy', mmap_mode='r')
        if len(self.records) != self.index.ntotal or self.vectors.shape != (len(self.records), self.index.d):
            raise RuntimeError('Segment index/metadata/vector mismatch')
        self.maps = SegmentMaps(root, dataset, source, source_records)
        self.by_id = {r['segment_id']: i for i, r in enumerate(self.records)}


class HierarchicalSearch:
    def __init__(self, orchestrator, root, dataset):
        self.orchestrator = orchestrator
        self.root, self.dataset = Path(root), dataset
        self.sources, self.errors = {}, {}
        try:
            counts = build_maps(self.root, self.dataset)
            log.info('Built startup segment maps dataset=%s asr=%d caption=%d',
                     self.dataset, counts['asr'], counts['caption'])
        except Exception as exc:
            log.warning('Could not build startup segment maps for %s: %s', self.dataset, exc)
        for source in ('asr', 'caption'):
            try:
                self.sources[source] = SegmentIndex(self.root, dataset, source)
            except Exception as exc:
                self.errors[source] = str(exc)
                log.warning('%s/%s unavailable: %s', dataset, source, exc)

    def health(self):
        return {s: {'backend': 'embedding', 'configured': True, 'available': s in self.sources,
                    'documents': len(self.sources[s].records) if s in self.sources else 0,
                    'error': self.errors.get(s)} for s in ('asr', 'caption')}

    def encode(self, source, queries):
        if source not in self.sources:
            raise RuntimeError(self.errors.get(source, f'{source} unavailable'))
        # Segment documents and their search queries must share the same Qwen
        # vLLM encoder and embedding space as visual semantic retrieval.
        index = self.orchestrator.semantic_search._resolve()
        model = index.model
        if (index.model_name != QWEN_MODEL_ID or index.backend != 'qwen3_vl_embedding'
                or getattr(model, 'engine', None) != 'vllm'):
            raise RuntimeError('Caption/ASR search requires the Qwen3-VL vLLM visual index')
        encoded = model.encode_text(list(queries))
        vectors = np.asarray(encoded.detach().cpu().numpy(), dtype=np.float32)
        if vectors.shape != (len(queries), self.sources[source].index.d):
            raise RuntimeError(f'Qwen text embedding shape mismatch: {vectors.shape}')
        if not np.isfinite(vectors).all():
            raise RuntimeError('Qwen text embeddings contain NaN/Inf')
        return np.ascontiguousarray(vectors)

    def hits(self, source, queries, top_k, weights=None, operation='weighted'):
        vectors = self.encode(source, queries)
        store = self.sources[source]
        allowed = video_ids_context.get()
        # FAISS candidate union; filtered retrieval must search enough to avoid filter starvation.
        count = store.index.ntotal if allowed else min(store.index.ntotal, max(top_k * 5, 100))
        _, ids = store.index.search(vectors, count)
        candidates = sorted({int(i) for i in ids.flat if i >= 0 and
                             (not allowed or store.records[int(i)]['video_id'] in allowed)})
        if not candidates:
            return []
        scores = weighted_scores(vectors @ store.vectors[candidates].T, weights, operation)
        ranked = np.argsort(-scores, kind='stable')[:top_k]
        log.info('Segment retrieval source=%s candidates=%s returned=%s', source, len(candidates), len(ranked))
        return [(store.records[candidates[i]], float(scores[i])) for i in ranked]

    def frame(self, key, model_key=None):
        index = self.orchestrator.semantic_search._resolve(model_key)
        return index.metadata_row_for_ocr_hit(key[0], str(key[1]))

    def segment_search(self, source, query, top_k=10):
        rows = []
        for segment, score in self.hits(source, [query], top_k):
            maps = self.sources[source].maps
            frames = []
            for key in maps.forward[segment['segment_id']]:
                record = self.frame(key)
                if record is not None:
                    frames.append({**record, 'mapping': maps.relations[segment['segment_id']].get(key, {})})
            if not frames:
                continue
            row = frames[len(frames) // 2].copy()
            for frame in frames:
                frame.update(score=score, retrieval_score=score, sub_query=segment['text'])
                if source == 'asr':
                    frame['playback_start'] = segment['start_time']
            row.update(segment_id=segment['segment_id'], matched_sequence=frames,
                       matched_texts=[segment['text']], caption=segment['text'],
                       temporal_start_time=segment['start_time'], temporal_end_time=segment['end_time'],
                       search_mode=source, retrieval_score=score, score=score,
                       display_rank=len(rows) + 1, rank=len(rows) + 1)
            if source == 'asr':
                row['playback_start'] = segment['start_time']
            rows.append(row)
        return pd.DataFrame(rows)

    def visual_scores(self, rows, queries, weights=None, operation='weighted', model_key=None,
                      image_paths=None):
        from src.retrieval.retriever.semantic_search.pipeline.search import similarities_for_candidates
        index = self.orchestrator.semantic_search._resolve(model_key)
        ids, found = [], []
        for row in rows:
            identity = (str(row['video_id']), int(row.get('keyframe_id_int', row['keyframe_id'])))
            number = index._row_by_video_frame.get(identity)
            if number is not None:
                ids.append(number)
                found.append(row)
        embeddings = [index.encode_text(q) for q in queries]
        embeddings += [index.encode_image(p) for p in image_paths or []]
        if not embeddings:
            raise ValueError('A visual query or input image is required')
        if not ids:
            return [], np.array([])
        matrix = similarities_for_candidates(index.index, index.search_lock, np.concatenate(embeddings),
            np.asarray(ids, dtype=np.int64), index.index_vectors)
        return found, weighted_scores(matrix, weights, operation)

    def search(self, *, query, direction='top_down', source_queries=None, source_weights=None,
               visual_queries=None, query_weights=None, operation='weighted', visual_weight=0.7,
               sequence_top_k=30, pool_k=300, top_k=20, model_key=None, speed=None,
               ocr_query=None, ocr_match='phrase', initial_frames=None, image_paths=None,
               source_query_weights=None):
        if direction not in ('top_down', 'bottom_up'):
            raise ValueError('Invalid hierarchical direction')
        if not 0 <= visual_weight <= 1:
            raise ValueError('visual_weight must be between 0 and 1')
        source_queries = source_queries or {s: [query] for s in self.sources}
        if not source_queries:
            raise RuntimeError('No segment embedding sources available')
        weights = source_weights or {s: 1.0 for s in source_queries}
        if set(weights) != set(source_queries) or any(not np.isfinite(w) or w < 0 for w in weights.values()) or sum(weights.values()) <= 0:
            raise ValueError('source_weights must match active sources and have a positive sum')
        visual_queries = visual_queries if visual_queries is not None else [query]
        if direction == 'top_down':
            segments = []
            for source, queries in source_queries.items():
                if weights[source] == 0:
                    continue
                for rank, (segment, _) in enumerate(self.hits(source, queries, sequence_top_k,
                        (source_query_weights or {}).get(source)), 1):
                    # Weighted interleaving keeps a lower-weight modality represented
                    # instead of letting an RRF offset make one source occupy every slot.
                    segments.append((weights[source] / rank, source, segment))
            rows, seen = [], set()
            for _, source, segment in sorted(segments, key=lambda x: -x[0])[:sequence_top_k]:
                for key in self.sources[source].maps.forward[segment['segment_id']]:
                    if key not in seen:
                        seen.add(key)
                        row = self.frame(key, model_key)
                        if row is not None:
                            rows.append(row.copy())
            # Bound candidate work without privileging the first long segment.
            if len(rows) > pool_k:
                rows = [rows[i] for i in np.linspace(0, len(rows)-1, pool_k, dtype=int)]
        else:
            if initial_frames is not None:
                rows = [r.copy() for r in initial_frames]
            else:
                semantic = self.orchestrator.semantic_search
                rows = []
                for text in visual_queries:
                    rows += semantic.multi_query_search([text], pool_k, pool_k, model_key=model_key).to_dict('records')
                for path in image_paths or []:
                    index = semantic._resolve(model_key)
                    rows += semantic.multi_query_search(['image'], pool_k, pool_k,
                        query_embeddings=index.encode_image(path), model_key=model_key).to_dict('records')
                unique = {(str(r['video_id']), int(r['keyframe_id'])): r for r in rows}
                rows = list(unique.values())
        allowed = video_ids_context.get()
        rows = [r for r in rows if not allowed or str(r['video_id']) in allowed]
        log.info('Hierarchical direction=%s mapped_frame_pool=%s', direction, len(rows))
        if speed:
            if 'caption' not in self.sources:
                raise RuntimeError('Speed filter requires caption data')
            store = self.sources['caption']
            rows = [r for r in rows if any(store.records[store.by_id[s]]['speed'] == speed
                for s in store.maps.reverse.get((str(r['video_id']), int(r['keyframe_id'])), []))]
        if ocr_query:
            pipeline = self.orchestrator.ocr_search_pipeline
            if pipeline is None:
                raise RuntimeError('OCR is unavailable')
            es = pipeline.repository.es
            # Exact filtering is limited to candidate identities, never a global top-k cutoff.
            permitted = es.match_frame_keys(ocr_query, [(str(r['video_id']), str(r['keyframe_id'])) for r in rows], ocr_match)
            rows = [r for r in rows if (str(r['video_id']), int(r['keyframe_id'])) in permitted]
        rows, visual = self.visual_scores(rows, visual_queries, query_weights, operation, model_key, image_paths)
        log.info('Hierarchical direction=%s frames_after_filters=%s', direction, len(rows))
        if not rows:
            return pd.DataFrame()
        final = visual.copy()
        if direction == 'bottom_up':
            # Missing context is neutral; normalize over available context per candidate.
            numerator, denominator = np.zeros(len(rows)), np.zeros(len(rows))
            for source, queries in source_queries.items():
                embeddings = self.encode(source, queries)
                store = self.sources[source]
                for i, row in enumerate(rows):
                    sids = store.maps.reverse.get((str(row['video_id']), int(row['keyframe_id'])), [])
                    if sids:
                        ids = [store.by_id[s] for s in sids]
                        score = weighted_scores(embeddings @ store.vectors[ids].T,
                            (source_query_weights or {}).get(source)).max()
                        numerator[i] += weights[source] * score
                        denominator[i] += weights[source]
            has = denominator > 0
            context = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=has)
            final[has] = visual_weight * visual[has] + (1-visual_weight) * context[has]
        result = pd.DataFrame(rows)
        result['visual_score'] = visual
        result['retrieval_score'] = final
        result['score'] = final
        result['search_mode'] = 'hierarchical'
        result = result.sort_values('retrieval_score', ascending=False, kind='stable').head(top_k).copy()
        result['rank'] = result['display_rank'] = range(1, len(result)+1)
        # Never leak sequence-shaped cards into hierarchical results.
        return result.drop(columns=['matched_sequence'], errors='ignore')
