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
    query: str = Field(min_length=1, max_length=4000)
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
    duration_limit: float = -1
    parent_pool_id: str | None = None
    restrict_to_parent: bool = False
    reference_video: str | None = None
    reference_frame: int | None = Field(default=None, ge=0)
    color_cells: list[dict] = Field(default_factory=list, max_length=25)

    @model_validator(mode='after')
    def constraints(self):
        import math
        if self.source_queries is not None:
            if not self.source_queries or set(self.source_queries) - {'asr', 'caption'}:
                raise ValueError('Sources must be asr/caption')
            if any(not qs or len(qs)>8 or any(not q.strip() or len(q)>4000 for q in qs) for qs in self.source_queries.values()):
                raise ValueError('Each source needs 1..8 nonempty queries')
        for values in (self.source_weights.values() if self.source_weights else None, self.query_weights):
            if values is not None and (any(not math.isfinite(v) or v<0 for v in values) or sum(values)<=0):
                raise ValueError('Weights must be finite, nonnegative and have positive sum')
        if self.source_query_weights:
            if not self.source_queries or set(self.source_query_weights) - set(self.source_queries):
                raise ValueError('source_query_weights needs matching source_queries')
            for source, weights in self.source_query_weights.items():
                if len(weights) != len(self.source_queries[source]) or any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
                    raise ValueError('Invalid source query weights')
        if self.visual_queries is not None and any(not q.strip() for q in self.visual_queries):
            raise ValueError('Empty visual query')
        if self.restrict_to_parent and (not self.parent_pool_id or self.direction != 'bottom_up' or self.mode != 'hierarchical'):
            raise ValueError('Parent pool restriction requires hierarchical bottom_up')
        if self.mode != 'hierarchical' and (self.speed is not None or self.ocr_query):
            raise ValueError('speed/ocr_query filters currently require hierarchical mode')
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
            return {'sources': self.system.text_search_health(), 'models': self.system.available_models(), 'modes': ['semantic', 'temporal', 'asr', 'caption', 'ocr', 'hierarchical', 'similarity', 'color'],
                    'images_attached': len(self.image_paths), 'selected_frame_references': getattr(self, 'frame_references', []),
                    'color_cells': getattr(self, 'color_cells', []), 'preferred_mode': getattr(self, 'preferred_search_mode', 'semantic'),
                    'top_k_limit': self.top_k_limit,
                    'plans': [{k: p.get(k) for k in ('id','plan','status','error')} for p in self.session['plans']]}, []
        if name == 'register_plans':
            # Models occasionally emit an empty query for an unused modality.
            # Drop that source before validation so valid sibling sources still run.
            if isinstance(arguments, dict) and isinstance(arguments.get('plans'), list):
                for raw_plan in arguments['plans']:
                    if not isinstance(raw_plan, dict) or not isinstance(raw_plan.get('source_queries'), dict):
                        continue
                    original = raw_plan['source_queries']
                    cleaned = {
                        source: [query.strip() for query in queries if isinstance(query, str) and query.strip()]
                        for source, queries in original.items()
                        if isinstance(queries, list)
                    }
                    cleaned = {source: queries for source, queries in cleaned.items() if queries}
                    if cleaned:
                        raw_plan['source_queries'] = cleaned
                    else:
                        raw_plan.pop('source_queries', None)
                    active_sources = set(cleaned)
                    if isinstance(raw_plan.get('source_weights'), dict):
                        weights = {key: value for key, value in raw_plan['source_weights'].items() if key in active_sources}
                        if weights:
                            raw_plan['source_weights'] = weights
                        else:
                            raw_plan.pop('source_weights', None)
                    if isinstance(raw_plan.get('source_query_weights'), dict):
                        focused_weights = {
                            key: value for key, value in raw_plan['source_query_weights'].items()
                            if key in active_sources and len(value) == len(cleaned[key])
                        }
                        if focused_weights:
                            raw_plan['source_query_weights'] = focused_weights
                        else:
                            raw_plan.pop('source_query_weights', None)
            batch = PlanBatch.model_validate(arguments)
            if len(self.session['plans']) + len(batch.plans) > 60:
                raise ValueError('Session plan limit reached; start a new session')
            for plan in batch.plans:
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
                if not plan.parent_pool_id:
                    plan.parent_pool_id = self.session.get('parent_pool_id')
                if plan.parent_pool_id:
                    pool_by_id(self.session, plan.parent_pool_id)
                self.session['plans'].append({'id': uuid.uuid4().hex, 'plan': plan.model_dump(), 'status': 'queued', 'image_paths': self.image_paths})
            if load(self.session['id']).get('stop_requested'):
                raise ValueError('User stopped this turn')
            save(self.session)
            return {'plans': [{k:v for k,v in p.items() if k != 'image_paths'} for p in self.session['plans'][-len(batch.plans):]]}, []
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
                if not payload.images and payload.parent_pool_id:
                    images = pool_by_id(session, payload.parent_pool_id).get('image_paths', [])
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
                text = payload.message + f'\nFeedback parent pool: {payload.parent_pool_id}. Attached search images: {len(images)}.'
                async for event in stream_codex(ROOT, tools, text, session['history'], payload.model):
                    await queue.put(event)
                # Finish queued strategies even if the model ends after publishing the first pool.
                while any(p['status']=='queued' for p in session['plans']):
                    data, _ = await asyncio.to_thread(tools.execute, 'run_next_plan', {})
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
