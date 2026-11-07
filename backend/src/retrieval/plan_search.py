"""Execute retrieval plans within resolved scope; no HTTP or UI serialization."""
from pathlib import Path
import pandas as pd
from src.retrieval.retriever.common.candidate_filter import with_video_filter, with_frame_filter


def search_plan(plan, system, paths, scope, session=None, image_paths=None):
    if scope.empty:
        return pd.DataFrame()
    video_ids = sorted(scope.videos) if scope.videos is not None else None
    engine = system.orchestrator.mapped_search
    if plan.mode == 'hierarchical' and plan.direction == 'top_down':
        plan.sequence_top_k = max(plan.sequence_top_k, 2 * plan.top_k)
        plan.pool_k = max(plan.pool_k, 2 * plan.top_k)
    def run():
        if plan.mode == 'hierarchical':
            kwargs = {k: getattr(plan, k) for k in ('query', 'direction', 'source_queries', 'source_weights',
                'visual_queries', 'query_weights', 'operation', 'visual_weight', 'sequence_top_k',
                'pool_k', 'top_k', 'model_key', 'speed', 'ocr_query', 'ocr_match', 'source_query_weights')}
            return system.search_hierarchical(**kwargs, image_paths=image_paths)
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
    return with_video_filter(video_ids, lambda: with_frame_filter(scope.frames, run))
