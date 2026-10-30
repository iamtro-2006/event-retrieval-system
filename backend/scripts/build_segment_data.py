"""Build Qwen vLLM segment indexes: python -m scripts.build_segment_data --dataset v3c."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.retrieval.segment_data import (SegmentMaps, build_maps, fingerprint,
                                        load_segment_files, load_segments)


def main():
    from src.embedding_extraction.models.backends.qwen_vllm import MODEL_ID

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['v3c', 'lhe'], required=True)
    parser.add_argument('--data-root', type=Path, default=Path('../data'))
    parser.add_argument('--model', default=MODEL_ID)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--sources', nargs='+', choices=['asr', 'caption'], default=['asr', 'caption'])
    parser.add_argument('--embeddings-root', type=Path,
                        help='Optional Qwen root containing <source>/<DATASET>/<video_id>/<segment_number>.npy vectors')
    parser.add_argument('--maps-only', action='store_true')
    parser.add_argument('--fallback-tau', type=float, default=5.)
    parser.add_argument('--fallback-right-weight', type=float, default=2.)
    args = parser.parse_args()
    if args.model != MODEL_ID:
        parser.error(f'Caption/ASR indexes must use {MODEL_ID} through vLLM')
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')

    root = args.data_root.resolve()
    print('Mapping:', build_maps(root, args.dataset, args.fallback_tau, args.fallback_right_weight), flush=True)
    if args.maps_only:
        return

    import faiss
    import numpy as np
    from src.embedding_extraction.models.backends.qwen_vllm import QwenVllmEncoder
    from src.retrieval.indexer.faiss.vector_cache import build_vector_cache

    embeddings_root = args.embeddings_root
    if embeddings_root is not None and not embeddings_root.is_absolute():
        embeddings_root = root / embeddings_root
    encoder = QwenVllmEncoder() if embeddings_root is None else None
    try:
        for source in args.sources:
            source_files = list(load_segment_files(root, args.dataset, source))
            if source == 'caption':
                all_records = [record for _, file_records in source_files for record in file_records]
                maps = SegmentMaps(root, args.dataset, 'caption', all_records)
                source_files = [(video_id, [record for record in file_records
                                            if maps.forward.get(record['segment_id'])])
                                for video_id, file_records in source_files]
            source_files = [(video_id, file_records) for video_id, file_records in source_files
                            if file_records]
            if not any(file_records for _, file_records in source_files):
                print(f'{source}: no indexable documents; skipped', flush=True)
                continue

            all_indexable_records = [record for _, file_records in source_files for record in file_records]
            records = []
            index = faiss.IndexFlatIP(2048)
            if embeddings_root is not None:
                artifact_root = embeddings_root / source / args.dataset.upper()
                for video_id, file_records in source_files:
                    if not file_records:
                        continue
                    video_vector_dir = artifact_root / video_id
                    valid_records, batch_vectors = [], []
                    for record in file_records:
                        segment_number = int(record['segment_id'].rsplit(':', 1)[-1])
                        vector_path = video_vector_dir / f'{segment_number:06d}.npy'
                        try:
                            vector = np.load(vector_path, allow_pickle=False)
                            if vector.shape != (2048,) or not np.isfinite(vector).all():
                                raise ValueError(f'invalid vector shape/content: {vector.shape}')
                            if not np.isclose(np.linalg.norm(vector), 1.0, atol=1e-3, rtol=0):
                                raise ValueError('vector is not normalized')
                        except (OSError, ValueError, EOFError) as exc:
                            print(f'{source}: skip segment {record["segment_id"]}: {exc}', flush=True)
                            continue
                        valid_records.append(record)
                        batch_vectors.append(vector.astype(np.float32, copy=False))
                        if len(batch_vectors) >= args.batch_size:
                            index.add(np.ascontiguousarray(np.asarray(batch_vectors, dtype=np.float32)))
                            records.extend(valid_records)
                            valid_records, batch_vectors = [], []
                    if batch_vectors:
                        index.add(np.ascontiguousarray(np.asarray(batch_vectors, dtype=np.float32)))
                        records.extend(valid_records)
                    print(f'{source}: added {video_id} ({sum(r["video_id"] == video_id for r in records)} valid segments)', flush=True)
            else:
                completed = 0
                for video_id, file_records in source_files:
                    if not file_records:
                        continue
                    for offset in range(0, len(file_records), args.batch_size):
                        batch = file_records[offset:offset + args.batch_size]
                        encoded = encoder.encode_text([record['text'] for record in batch])
                        vectors = np.asarray(encoded.detach().cpu().numpy(), dtype=np.float32)
                        index.add(np.ascontiguousarray(vectors))
                        records.extend(batch)
                        completed += len(batch)
                        print(f'{source}: embedded/indexed {completed}', flush=True)
            if index.ntotal != len(records):
                raise RuntimeError(f'{source} FAISS/metadata row mismatch: {index.ntotal} != {len(records)}')
            folder = (root / 'database' / 'faiss_qwen3_vl_embedding_2b_vllm'
                      / source / args.dataset.upper())
            folder.mkdir(parents=True, exist_ok=True)
            index_path = folder / f'{source}.faiss'
            temporary_index_path = folder / f'{source}.faiss.tmp'
            faiss.write_index(index, str(temporary_index_path))
            temporary_index_path.replace(index_path)
            (folder / 'metadata.json.tmp').write_text(json.dumps({
                'model': MODEL_ID, 'engine': 'vllm', 'source_hash': fingerprint(all_indexable_records),
                'records': records,
            }, ensure_ascii=False), encoding='utf-8')
            (folder / 'metadata.json.tmp').replace(folder / 'metadata.json')

            cache_path = folder / 'vectors_fp32.npy'
            temporary_cache_path = folder / 'vectors_fp32.npy.tmp'
            build_vector_cache(index_path, temporary_cache_path, np.dtype(np.float32),
                               batch_size=50_000, normalize=False)
            temporary_cache_path.replace(cache_path)
            print(f'{source}: {len(records)} Qwen vLLM documents → {folder}', flush=True)
    finally:
        if encoder is not None:
            encoder.close()


if __name__ == '__main__':
    main()
