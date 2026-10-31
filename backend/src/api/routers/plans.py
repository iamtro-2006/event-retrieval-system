"""Real search plans, sequential agent sessions, and immutable result pools."""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import sqlite3
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Literal

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.retrieval.retriever.common.video_filter import with_video_filter

def get_dataset_resources(request, dataset):
    from src.api.legacy.deps import get_dataset_resources as resolve
    return resolve(request, dataset)


def _run_or_503(function, **kwargs):
    from src.api.routers.search import _run_or_503 as run
    return run(function, **kwargs)


router = APIRouter(tags=['search-plans'])
log = logging.getLogger(__name__)
ROOT = Path(os.environ.get('SEARCH_STATE_ROOT', Path(__file__).resolve().parents[4] / 'data' / 'search_state'))
LOCK = RLock()
TASKS = {}


class SearchPlan(BaseModel):
    model_config = ConfigDict(extra='forbid')
    label: str = Field(default='Search', max_length=200)
    query: str | None = Field(default=None, min_length=1, max_length=4000,
        description='General text query. Semantic visual plans can omit it when visual_queries is provided; hierarchical plans can omit it when source_queries and visual_queries are provided. Similarity uses reference_video/reference_frame; color uses color_cells.')
    mode: Literal['semantic', 'temporal', 'asr', 'caption', 'ocr', 'hierarchical', 'similarity', 'color'] = 'semantic'
    direction: Literal['top_down', 'bottom_up'] = 'top_down'
    visual_queries: list[str] | None = Field(default=None, max_length=8)
    image_event_indices: list[int] | None = Field(default=None, max_length=6,
        description='For temporal search, zero-based event assigned to each attached image; defaults to first event.')
    query_weights: list[float] | None = Field(default=None, max_length=14)
    operation: Literal['weighted', 'and'] = 'weighted'
    source_queries: dict[str, list[str]] | None = None
    source_weights: dict[str, float] | None = None
    source_query_weights: dict[str, list[float]] | None = None
    visual_weight: float = Field(default=.7, ge=0, le=1)
    sequence_top_k: int = Field(default=30, ge=1, le=1000)
    pool_k: int = Field(default=300, ge=1, le=2000)
    top_k: int = Field(default=200, ge=1, le=500,
        description='Requested result count. Semantic/visual plans usually target 200; segment-first plans target 50–100 based on query ambiguity.')
    model_key: str | None = None
    speed: Literal['normal', 'timelapse', 'slow_motion', 'unknown'] | None = None
    ocr_query: str | None = Field(default=None, max_length=1000)
    ocr_match: Literal['terms', 'phrase'] = 'phrase'
    duration_limit: float = Field(default=-1, ge=-1)
    parent_pool_id: str | None = None
    restrict_to_parent: bool = False
    reference_video: str | None = None
    reference_frame: int | None = Field(default=None, ge=0)
    color_cells: list[dict] = Field(default_factory=list, max_length=25)

    @model_validator(mode='after')
    def constraints(self):
        import math
        if not self.label.strip():
            raise ValueError('label must not be blank')
        if self.query is not None and not self.query.strip():
            raise ValueError('query must not be blank')
        if self.mode == 'semantic' and not self.query and self.visual_queries is None:
            raise ValueError('Semantic plans without query must provide the visual_queries field; use an empty list only with an attached image')
        if self.mode == 'hierarchical' and not self.query:
            has_sources = bool(self.source_queries and any(self.source_queries.values()))
            if not (has_sources and self.visual_queries is not None):
                raise ValueError('Hierarchical plans without query must provide source_queries and the visual_queries field')
        if self.mode not in ('semantic', 'hierarchical', 'similarity', 'color') and not self.query:
            raise ValueError('query is required for text-based plans')
        if self.source_queries is not None:
            if not self.source_queries or set(self.source_queries) - {'asr', 'caption'}:
                raise ValueError('Sources must be asr/caption')
            if any(not qs or len(qs)>8 or any(not q.strip() or len(q)>4000 for q in qs) for qs in self.source_queries.values()):
                raise ValueError('Each source needs 1..8 nonempty queries')
        for values in (self.source_weights.values() if self.source_weights else None, self.query_weights):
            if values is not None and (any(not math.isfinite(v) or v<0 for v in values) or sum(values)<=0):
                raise ValueError('Weights must be finite, nonnegative and have positive sum')
        if self.source_weights is not None and self.source_queries is not None and set(self.source_weights) != set(self.source_queries):
            raise ValueError('source_weights must have exactly the same keys as source_queries')
        if self.source_weights is not None and self.source_queries is None:
            raise ValueError('source_weights requires source_queries')
        if self.source_query_weights:
            if not self.source_queries or set(self.source_query_weights) - set(self.source_queries):
                raise ValueError('source_query_weights needs matching source_queries')
            for source, weights in self.source_query_weights.items():
                if len(weights) != len(self.source_queries[source]) or any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
                    raise ValueError('Invalid source query weights')
        if self.visual_queries is not None and any(not q.strip() for q in self.visual_queries):
            raise ValueError('Empty visual query')
        if self.mode != 'hierarchical' and any(value is not None for value in (
                self.source_queries, self.source_weights, self.source_query_weights)):
            raise ValueError('source_queries and source weights require hierarchical mode')
        if self.mode not in ('semantic', 'hierarchical') and any(value is not None for value in (
                self.visual_queries, self.query_weights)):
            raise ValueError('visual_queries and query_weights require semantic or hierarchical mode')
        if self.mode != 'hierarchical' and (self.direction != 'top_down' or self.speed is not None or self.ocr_query):
            raise ValueError('direction, speed, and ocr_query filters require hierarchical mode')
        if self.mode != 'temporal' and (self.duration_limit != -1 or self.image_event_indices is not None):
            raise ValueError('duration_limit and image_event_indices require temporal mode')
        if self.mode != 'hierarchical' and self.visual_weight != .7:
            raise ValueError('visual_weight requires hierarchical mode')
        if self.mode != 'hierarchical' and self.sequence_top_k != 30:
            raise ValueError('sequence_top_k requires hierarchical mode')
        if self.mode not in ('semantic', 'temporal', 'hierarchical') and self.pool_k != 300:
            raise ValueError('pool_k is only used by semantic visual, temporal image, or hierarchical plans')
        if self.mode not in ('semantic', 'hierarchical') and self.operation != 'weighted':
            raise ValueError('operation requires semantic visual or hierarchical mode')
        if self.mode != 'color' and self.color_cells:
            raise ValueError('color_cells require color mode')
        if self.mode != 'similarity' and (self.reference_video is not None or self.reference_frame is not None):
            raise ValueError('reference_video/reference_frame require similarity mode')
        if self.image_event_indices is not None and any(isinstance(index, bool) or index < 0 for index in self.image_event_indices):
            raise ValueError('image_event_indices must be nonnegative event indexes')
        if self.restrict_to_parent and (not self.parent_pool_id or self.direction != 'bottom_up' or self.mode != 'hierarchical'):
            raise ValueError('Parent pool restriction requires hierarchical bottom_up')
        if self.ocr_query is not None and not self.ocr_query.strip():
            raise ValueError('ocr_query must not be blank')
        if self.mode not in ('ocr', 'hierarchical') and self.ocr_match != 'phrase':
            raise ValueError('ocr_match requires OCR or hierarchical mode')
        if self.mode == 'similarity' and (not self.reference_video or not self.reference_video.strip() or self.reference_frame is None):
            raise ValueError('Similarity plans require reference_video and reference_frame')
        if self.mode == 'color':
            if not self.color_cells:
                raise ValueError('Color plans require at least one selected color cell')
            allowed_colors = {'black', 'blue', 'brown', 'grey', 'green', 'orange', 'pink', 'purple', 'red', 'white', 'yellow'}
            positions = set()
            for cell in self.color_cells:
                if (not isinstance(cell, dict) or isinstance(cell.get('row'), bool) or isinstance(cell.get('col'), bool)
                        or not isinstance(cell.get('row'), int) or not 0 <= cell['row'] < 5
                        or not isinstance(cell.get('col'), int) or not 0 <= cell['col'] < 5
                        or not isinstance(cell.get('color'), str) or cell.get('color') not in allowed_colors):
                    raise ValueError('Each color cell needs row/col in 0..4 and a supported color name')
                position = (cell['row'], cell['col'])
                if position in positions:
                    raise ValueError('Color plans may select each grid cell only once')
                positions.add(position)
        return self


class ExecuteRequest(SearchPlan):
    dataset: Literal['v3c', 'lhe']
    video_ids: list[str] | None = None


@contextmanager
def database():
    ROOT.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(ROOT / 'sessions.sqlite3')
    try:
        with con:
            con.execute('CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, data TEXT NOT NULL)')
            yield con
    finally:
        con.close()


def save(session):
    with LOCK, database() as con:
        con.execute('INSERT OR REPLACE INTO sessions VALUES (?,?)', (session['id'], json.dumps(session, ensure_ascii=False)))


def load(identifier):
    with LOCK, database() as con:
        row = con.execute('SELECT data FROM sessions WHERE id=?', (identifier,)).fetchone()
    if row is None:
        raise HTTPException(404, 'Unknown search session')
    return json.loads(row[0])


def pool_by_id(session, identifier):
    for plan in session['plans']:
        if plan['id'] == identifier and plan.get('status') == 'completed':
            return plan
    raise ValueError('Unknown completed pool in this session')


def execute(plan, system, paths, video_ids=None, session=None, image_paths=None):
    """Shared execution path for manual API requests and Codex tools."""
    from src.api.legacy.serializers import dict_to_result_FAST
    from src.api.routers.search import _records_from_df
    engine = system.orchestrator.mapped_search
    if plan.mode == 'hierarchical' and plan.direction == 'top_down':
        plan.sequence_top_k = max(plan.sequence_top_k, 2 * plan.top_k)
        plan.pool_k = max(plan.pool_k, 2 * plan.top_k)
    initial = None
    if plan.parent_pool_id:
        parent = pool_by_id(session, plan.parent_pool_id) if session else None
        if parent is None:
            raise ValueError('Parent pool requires a session')
        if not image_paths:
            image_paths = parent.get('image_paths', [])
        if plan.restrict_to_parent:
            initial = []
            for row in parent['records']:
                initial.extend(row.get('matched_sequence') or [row])
    def run():
        if plan.mode == 'hierarchical':
            kwargs = {k: getattr(plan, k) for k in ('query', 'direction', 'source_queries', 'source_weights',
                'visual_queries', 'query_weights', 'operation', 'visual_weight', 'sequence_top_k',
                'pool_k', 'top_k', 'model_key', 'speed', 'ocr_query', 'ocr_match', 'source_query_weights')}
            return system.search_hierarchical(**kwargs, initial_frames=initial, image_paths=image_paths)
        if plan.mode in ('asr', 'caption'):
            return engine.segment_search(plan.mode, plan.query, plan.top_k)
        if plan.mode == 'ocr':
            pipeline = system.orchestrator.ocr_search_pipeline
            if pipeline is None:
                raise RuntimeError('OCR is unavailable')
            hits = pipeline.search(plan.query, plan.top_k * 3, match=plan.ocr_match)
            return system.orchestrator._enrich_ocr_hits(hits, plan.top_k)
        if plan.mode == 'temporal' and image_paths:
            import numpy as np
            from src.retrieval.retriever.common.query_parser import parse_explicit_query
            from src.retrieval.retriever.temporal_search.pipeline.search import temporal_search_from_events
            events, _ = parse_explicit_query(plan.query)
            event_images = plan.image_event_indices or [0] * len(image_paths)
            if len(event_images) != len(image_paths) or any(i < 0 or i >= len(events) for i in event_images):
                raise ValueError('image_event_indices must assign each image to an existing temporal event')
            index = system.orchestrator.semantic_search._resolve(plan.model_key)
            labels, embeddings = [], []
            for event_number, texts in enumerate(events):
                current = list(texts)
                vectors = [index.encode_text(text) for text in texts]
                for number, path in enumerate(image_paths):
                    if event_images[number] == event_number:
                        current.append(f'Image {number+1}')
                        vectors.append(index.encode_image(path))
                labels.append(current)
                embeddings.extend(vectors)
            df = temporal_search_from_events(index.index, index.search_lock, index.metadata_records,
                index.index_vectors, index.allow_npy_fallback, labels, np.vstack(embeddings),
                plan.top_k, plan.pool_k, plan.duration_limit)
            if len(df):
                df['model_key'] = index.model_key
            return df
        if plan.mode == 'semantic' and (plan.visual_queries or image_paths):
            semantic = system.orchestrator.semantic_search
            index = semantic._resolve(plan.model_key)
            queries = plan.visual_queries or ([] if image_paths else [plan.query])
            rows = []
            for q in queries:
                rows += semantic.multi_query_search([q], plan.pool_k, plan.pool_k, model_key=plan.model_key).to_dict('records')
            for image in image_paths or []:
                rows += semantic.multi_query_search(['image'], plan.pool_k, plan.pool_k,
                    query_embeddings=index.encode_image(image), model_key=plan.model_key).to_dict('records')
            rows = list({(str(r['video_id']), int(r['keyframe_id'])): r for r in rows}.values())
            rows, scores = engine.visual_scores(rows, queries, plan.query_weights, plan.operation, plan.model_key, image_paths)
            df = pd.DataFrame(rows)
            if len(df):
                df['retrieval_score'] = scores
                df = df.sort_values('retrieval_score', ascending=False).head(plan.top_k).copy()
                df['rank'] = df['display_rank'] = range(1, len(df)+1)
            return df
        if plan.mode == 'similarity':
            if plan.reference_video is None or plan.reference_frame is None:
                raise ValueError('Similarity requires reference_video and reference_frame')
            return system.orchestrator.semantic_search.similarity_search_by_frame(
                plan.reference_video, plan.reference_frame, top_k=plan.top_k, model_key=plan.model_key)
        if plan.mode == 'color':
            from src.color_retrieval.search import get_color_index
            index = system.orchestrator.index
            config = session.get('config', {}) if session else {}
            color = config.get('color', {}).get('index_path')
            if not color:
                raise RuntimeError('Color index not configured')
            color = Path(color)
            if not color.is_absolute():
                color = paths.backend_dir / color
            return get_color_index(color, index.metadata_records).search(index.metadata_records, plan.color_cells, plan.top_k, video_ids)
        return system.orchestrator.run_search(plan.query, mode=plan.mode, top_k=plan.top_k,
            model_key=plan.model_key, duration_limit=plan.duration_limit)[0]
    df = with_video_filter(video_ids, run)
    records = _records_from_df(df)
    results = [dict_to_result_FAST(r, paths.keyframes_root, paths.backend_dir, paths.dataset_key) for r in records]
    return records, results


@router.post('/api/search/plan')
def execute_manual(payload: ExecuteRequest, request: Request):
    if payload.parent_pool_id:
        raise HTTPException(422, 'parent_pool_id is available only in an agent search session')
    if payload.mode == 'semantic' and not payload.query and not payload.visual_queries:
        raise HTTPException(422, 'Semantic plans without query require visual_queries')
    if payload.mode == 'hierarchical':
        visuals = payload.visual_queries if payload.visual_queries is not None else ([payload.query] if payload.query else [])
        if not visuals:
            raise HTTPException(422, 'Hierarchical plans require visual_queries or a text query')
    if payload.mode == 'temporal' and (payload.image_event_indices is not None or payload.pool_k != 300):
        raise HTTPException(422, 'image_event_indices and custom pool_k require attached images, which this endpoint does not accept')
    if payload.query_weights is not None:
        if payload.mode == 'hierarchical':
            visuals = payload.visual_queries if payload.visual_queries is not None else ([payload.query] if payload.query else [])
        elif payload.mode == 'semantic' and payload.visual_queries:
            visuals = payload.visual_queries
        else:
            raise HTTPException(422, 'query_weights requires semantic visual_queries or a hierarchical plan')
        if len(payload.query_weights) != len(visuals):
            raise HTTPException(422, 'query_weights must contain one value per visual query')
    system, config, paths = get_dataset_resources(request, payload.dataset)
    records, results = _run_or_503(lambda: execute(payload, system, paths, payload.video_ids,
        {'config': config}), feature_name=payload.mode)
    return {'results': results, 'count': len(results), 'plan': payload.model_dump()}


@router.post('/api/search/caption')
def caption(payload: ExecuteRequest, request: Request):
    return execute_manual(payload.model_copy(update={'mode': 'caption'}), request)


@router.post('/api/search/hierarchical')
def hierarchical(payload: ExecuteRequest, request: Request):
    return execute_manual(payload.model_copy(update={'mode': 'hierarchical'}), request)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra='ignore')  # Existing UI sends display preferences as well.
    dataset: Literal['v3c', 'lhe']
    message: str = Field(min_length=1, max_length=10000)
    session_id: str | None = None
    parent_pool_id: str | None = None
    video_ids: list[str] | None = None
    top_k_limit: int = Field(default=500, ge=1, le=500)
    model: str | None = None
    provider: str = 'codex'
    reasoning_effort: str | None = None
    images: list[str] = Field(default_factory=list, max_length=6)
    context_frame_ids: list[str] = Field(default_factory=list, max_length=100)
    search_mode: str = 'semantic'
    color_cells: list[dict] = Field(default_factory=list, max_length=25)


class PlanBatch(BaseModel):
    plans: list[SearchPlan] = Field(min_length=1, max_length=6)


class RunNext(BaseModel):
    pass


class RealTools:
    max_calls = 20
    turn_timeout = 1800
    model_result_limit = 12
    max_workers = 1
    known_ids = set()
    displayed = []
    input_images = []

    def __init__(self, session, system, paths, image_paths):
        self.session, self.system, self.paths = session, system, paths
        self.dataset, self.video_ids = session['dataset'], session['video_ids']
        self.image_paths, self.calls = image_paths, 0
        self.top_k_limit = 500

    @staticmethod
    def specs():
        return [{'type': 'function', 'name': name, 'description': description, 'inputSchema': model.model_json_schema()}
            for name, description, model in [
                ('list_capabilities', 'Available real retrieval sources and existing plans/pools.', RunNext),
                ('register_plans', 'Register up to six distinct strategies BEFORE running them. Keep exact constraints; attach parent_pool_id for feedback branches.', PlanBatch),
                ('run_next_plan', 'Execute next queued plan sequentially and publish its pool immediately. Failed plans do not block later ones.', RunNext)]]

    def execute(self, name, arguments):
        self.calls += 1
        if self.calls > self.max_calls:
            raise ValueError('Tool budget exceeded')
        if name == 'list_capabilities':
            sources = self.system.text_search_health()
            models = self.system.available_models()
            ocr = sources.get('ocr', {})
            segment_available = {
                name: bool(sources.get(name, {}).get('available')) and int(sources.get(name, {}).get('documents') or 0) > 0
                for name in ('asr', 'caption')
            }
            color_path = self.session.get('config', {}).get('color', {}).get('index_path')
            color_available = False
            if color_path:
                color_path = Path(color_path)
                if not color_path.is_absolute():
                    color_path = self.paths.backend_dir / color_path
                color_available = color_path.is_file()
            mode_availability = {
                'semantic': bool(models),
                'temporal': bool(models),
                'asr': segment_available['asr'] and bool(models),
                'caption': segment_available['caption'] and bool(models),
                'ocr': bool(models) and bool(ocr.get('available') and ocr.get('reachable')
                    and ocr.get('index_exists') and int(ocr.get('documents') or 0) > 0),
                'hierarchical': bool(models) and any(segment_available.values()),
                'similarity': bool(models),
                'color': color_available and bool(models),
            }
            return {'sources': sources, 'models': models,
                    'modes': [mode for mode, available in mode_availability.items() if available],
                    'mode_availability': mode_availability,
                    'images_attached': len(self.image_paths), 'selected_frame_references': getattr(self, 'frame_references', []),
                    'color_cells': getattr(self, 'color_cells', []), 'preferred_mode': getattr(self, 'preferred_search_mode', 'semantic'),
                    'top_k_limit': self.top_k_limit,
                    'plans': [{k: p.get(k) for k in ('id','plan','status','error')} for p in self.session['plans']]}, []
        if name == 'register_plans':
            batch = PlanBatch.model_validate(arguments)
            if len(self.session['plans']) + len(batch.plans) > 60:
                raise ValueError('Session plan limit reached; start a new session')
            prepared = []
            for plan in batch.plans:
                parent = pool_by_id(self.session, plan.parent_pool_id) if plan.parent_pool_id else None
                inherited_images = self.image_paths or (parent.get('image_paths', []) if parent else [])
                if plan.mode == 'semantic' and not plan.query and not (plan.visual_queries or inherited_images):
                    raise ValueError('Semantic plan without query requires visual_queries or an attached search image')
                if plan.mode == 'semantic' and not (plan.visual_queries or inherited_images):
                    if plan.query_weights is not None or plan.operation != 'weighted' or plan.pool_k != 300:
                        raise ValueError('query_weights, operation, and pool_k require semantic visual queries or an attached search image')
                if plan.mode == 'temporal' and not inherited_images and plan.pool_k != 300:
                    raise ValueError('pool_k is only used by temporal plans with attached images')
                if plan.image_event_indices is not None:
                    if plan.mode != 'temporal' or not inherited_images:
                        raise ValueError('image_event_indices require a temporal plan with attached images')
                    if len(plan.image_event_indices) != len(inherited_images):
                        raise ValueError('image_event_indices must contain one event index per attached image')
                if plan.mode == 'hierarchical':
                    visual_queries = plan.visual_queries if plan.visual_queries is not None else ([plan.query] if plan.query else [])
                    if not visual_queries and not inherited_images:
                        raise ValueError('Hierarchical plans require visual_queries, a text query, or an attached search image')
                if plan.query_weights is not None:
                    if plan.mode == 'hierarchical':
                        visual_queries = plan.visual_queries if plan.visual_queries is not None else ([plan.query] if plan.query else [])
                    elif plan.mode == 'semantic' and (plan.visual_queries or inherited_images):
                        visual_queries = plan.visual_queries or ([] if inherited_images else [plan.query])
                    else:
                        raise ValueError('query_weights is only used for semantic visual queries or hierarchical plans')
                    expected_weights = len(visual_queries) + len(inherited_images)
                    if len(plan.query_weights) != expected_weights:
                        raise ValueError(f'query_weights must contain one weight per visual query/image ({expected_weights} total)')
                segment_first = plan.mode in ('asr', 'caption', 'ocr') or (
                    plan.mode == 'hierarchical' and plan.direction == 'top_down')
                if segment_first:
                    upper = min(self.top_k_limit, 100)
                    preferred = 70 if 'top_k' not in plan.model_fields_set else plan.top_k
                    plan.top_k = min(upper, max(min(50, upper), preferred))
                else:
                    preferred = 200 if 'top_k' not in plan.model_fields_set else plan.top_k
                    plan.top_k = min(preferred, self.top_k_limit)
                if plan.mode == 'hierarchical' and plan.direction == 'top_down':
                    plan.sequence_top_k = max(plan.sequence_top_k, 2 * plan.top_k)
                    plan.pool_k = max(plan.pool_k, 2 * plan.top_k)
                prepared.append({'id': uuid.uuid4().hex, 'plan': plan.model_dump(), 'status': 'queued', 'image_paths': self.image_paths})
            if load(self.session['id']).get('stop_requested'):
                raise ValueError('User stopped this turn')
            self.session['plans'].extend(prepared)
            save(self.session)
            return {'plans': [{k:v for k,v in p.items() if k != 'image_paths'} for p in prepared]}, []
        if name != 'run_next_plan':
            raise ValueError('Unknown tool')
        RunNext.model_validate(arguments)
        if load(self.session['id']).get('stop_requested'):
            for entry in self.session['plans']:
                if entry['status'] == 'queued':
                    entry['status'] = 'cancelled'
            self.session['stop_requested'] = True
            save(self.session)
            return {'message': 'User stopped this turn. Do not run more plans.', 'results': [], 'count': 0}, []
        entry = next((p for p in self.session['plans'] if p['status'] == 'queued'), None)
        if entry is None:
            return {'message': 'No queued plans', 'results': [], 'count': 0}, []
        entry['status'] = 'running'
        save(self.session)
        log.info('Plan %s starting mode=%s direction=%s', entry['id'], entry['plan']['mode'], entry['plan']['direction'])
        started = time.monotonic()
        try:
            entry['records'], entry['results'] = execute(SearchPlan.model_validate(entry['plan']),
                self.system, self.paths, self.video_ids, self.session, entry.get('image_paths', self.image_paths))
            entry['status'] = 'completed'
        except Exception as exc:
            log.exception('Search plan %s failed', entry['id'])
            entry.update(status='failed', error=str(exc), records=[], results=[])
        entry['latency_ms'] = round((time.monotonic()-started)*1000)
        log.info('Plan %s status=%s results=%s latency_ms=%s', entry['id'], entry['status'], len(entry['results']), entry['latency_ms'])
        if load(self.session['id']).get('stop_requested'):
            self.session['stop_requested'] = True
        save(self.session)
        return {k: v for k,v in {**entry, 'pool_id': entry['id'], 'count': len(entry['results'])}.items() if k not in ('records', 'image_paths')}, []


@router.get('/api/agent/status')
async def agent_status():
    from src.mock_codex import codex_status
    ROOT.mkdir(parents=True, exist_ok=True)
    return {'codex': await codex_status(ROOT), 'mock': False}


@router.get('/api/agent/sessions/{identifier}')
def read_session(identifier: str):
    session = load(identifier)
    if identifier not in TASKS:
        changed = False
        for entry in session['plans']:
            if entry['status'] in ('queued', 'running'):
                entry.update(status='interrupted', error='Host restarted or previous turn ended')
                changed = True
        if changed:
            save(session)
    public = {k: v for k,v in session.items() if k != 'config'}
    public['running'] = identifier in TASKS
    public['plans'] = [{k:v for k,v in p.items() if k != 'image_paths'} for p in session['plans']]
    return public


@router.post('/api/agent/sessions/{identifier}/stop')
async def stop_session(identifier: str):
    load(identifier)
    session = load(identifier)
    session['stop_requested'] = True
    save(session)
    return {'stopping': identifier in TASKS}


@router.post('/api/agent/chat')
async def chat(payload: ChatRequest, request: Request):
    system, config, paths = get_dataset_resources(request, payload.dataset)
    if payload.provider != 'codex':
        raise HTTPException(422, 'Real search requires provider=codex; demo remains on mock_server')
    session = load(payload.session_id) if payload.session_id else {
        'id': uuid.uuid4().hex, 'dataset': payload.dataset, 'video_ids': payload.video_ids,
        'plans': [], 'history': [], 'config': config}
    if session['dataset'] != payload.dataset or session['video_ids'] != payload.video_ids:
        raise HTTPException(409, 'Start a new session after changing dataset/video filters')
    if session['id'] in TASKS:
        raise HTTPException(409, 'A search turn is still running in this session')
    # Completed snapshots survive a restart; an old in-flight plan cannot resume implicitly.
    for entry in session['plans']:
        if entry['status'] in ('running', 'queued'):
            entry.update(status='interrupted', error='Previous search turn ended before this plan completed')
    session['stop_requested'] = False
    session['parent_pool_id'] = payload.parent_pool_id
    if payload.parent_pool_id:
        try:
            pool_by_id(session, payload.parent_pool_id)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
    queue = asyncio.Queue()
    save(session)

    async def run():
        from src.mock_codex import stream_codex
        try:
            with tempfile.TemporaryDirectory(prefix='ers-images-') as temporary:
                images = []
                image_root = ROOT / 'images' / session['id']
                image_root.mkdir(parents=True, exist_ok=True)
                for number, value in enumerate(payload.images):
                    if not value.startswith(('data:image/png;base64,', 'data:image/jpeg;base64,', 'data:image/webp;base64,')) or len(value)>3_000_000:
                        raise ValueError('Images must be PNG/JPEG/WebP data URLs under 2MB')
                    image = Path(temporary) / f'{number}.img'
                    image.write_bytes(base64.b64decode(value.split(',',1)[1], validate=True))
                    from PIL import Image
                    with Image.open(image) as decoded:
                        decoded.verify()
                    stored = image_root / f'{uuid.uuid4().hex}.img'
                    stored.write_bytes(image.read_bytes())
                    images.append(str(stored))
                tools = RealTools(session, system, paths, images)
                server_top_k_limit = int(config.get('search', {}).get('max_top_k', 500))
                tools.top_k_limit = min(payload.top_k_limit, server_top_k_limit, 500)
                tools.color_cells = payload.color_cells
                tools.preferred_search_mode = payload.search_mode
                tools.frame_references = []
                for identifier in payload.context_frame_ids:
                    try:
                        video, frame = identifier.rsplit('_', 1)
                        if payload.video_ids and video not in payload.video_ids:
                            continue
                        row = system.orchestrator.index.metadata_row_for_ocr_hit(video, frame)
                        if row is not None:
                            tools.frame_references.append({'reference_video': video, 'reference_frame': int(frame)})
                    except (ValueError, AttributeError):
                        continue
                tools.reasoning_effort = payload.reasoning_effort
                tools.instructions = (Path(__file__).resolve().parents[4] / '.agents/skills/hierarchical-video-search/SKILL.md').read_text(encoding='utf-8')
                # Do not ask Codex to inspect input pixels: the image is passed to the visual encoder.
                text = (payload.message + f'\nSelected pool for possible feedback: {payload.parent_pool_id or "none"}. '
                    'Use it as parent only if the user asks to refine, continue, or give feedback on that pool; '
                    'otherwise create independent search plans. '
                    f'Attached search images: {len(images)}.')
                async for event in stream_codex(ROOT, tools, text, session['history'], payload.model):
                    await queue.put(event)
                # Finish queued strategies even if the model ends after publishing the first pool.
                while any(p['status']=='queued' for p in session['plans']):
                    await queue.put({'type': 'tool_start', 'name': 'run_next_plan', 'arguments': {}})
                    try:
                        data, _ = await asyncio.to_thread(tools.execute, 'run_next_plan', {})
                    except Exception as exc:
                        await queue.put({'type': 'tool_end', 'name': 'run_next_plan', 'success': False, 'error': str(exc)})
                        raise
                    await queue.put({'type': 'tool_end', 'name': 'run_next_plan', 'count': data.get('count'),
                        'success': data.get('status') == 'completed', 'error': data.get('error')})
                    await queue.put({'type': 'pool', **data})
                session['history'] = (session['history'] + [{'role': 'user', 'text': payload.message}])[-16:]
                save(session)
        except asyncio.CancelledError:
            for p in session['plans']:
                if p['status'] == 'queued':
                    p['status'] = 'cancelled'
            save(session)
            await queue.put({'type': 'status', 'message': 'Search stopped; an active inference may finish before resources are released.'})
        except Exception as exc:
            log.exception('Agent search failed')
            await queue.put({'type': 'error', 'message': str(exc)})
        finally:
            await queue.put({'type': 'done'})
            TASKS.pop(session['id'], None)

    TASKS[session['id']] = asyncio.create_task(run())
    async def events():
        yield json.dumps({'type': 'session', 'session_id': session['id']}) + '\n'
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=15)
            except asyncio.TimeoutError:
                yield json.dumps({'type': 'status', 'message': 'Search running; pools remain available in this session'}) + '\n'
                continue
            yield json.dumps(event, ensure_ascii=False) + '\n'
            if event['type'] == 'done':
                break
    return StreamingResponse(events(), media_type='application/x-ndjson', headers={'X-Accel-Buffering': 'no'})
