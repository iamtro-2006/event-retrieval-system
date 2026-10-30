"""Pre-embed one ASR or caption collection with the Qwen vLLM endpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.embedding_extraction.models.backends.qwen_vllm import MODEL_ID, QwenVllmEncoder
from src.retrieval.segment_data import build_maps, load_segment_files, SegmentMaps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['v3c', 'lhe'], required=True)
    parser.add_argument('--source', choices=['asr', 'caption'], required=True)
    parser.add_argument('--data-root', type=Path, default=Path('../data'))
    parser.add_argument('--output-root', type=Path,
                        help='Defaults to <data-root>/embeddings/embeddings_qwen3_vl_embedding_2b_vllm')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--video-ids', nargs='+', help='Optional exact video IDs to embed, e.g. 00011 00012 or LHE11 LHE12')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')

    data_root = args.data_root.resolve()
    output_root = args.output_root or (
        data_root / 'embeddings' / 'embeddings_qwen3_vl_embedding_2b_vllm')
    if not output_root.is_absolute():
        output_root = (Path.cwd() / output_root).resolve()

    counts = build_maps(data_root, args.dataset)
    print(f"Maps ready: ASR={counts['asr']} caption={counts['caption']}", flush=True)
    source_files = list(load_segment_files(data_root, args.dataset, args.source))
    if args.video_ids:
        requested = set(args.video_ids)
        available = {video_id for video_id, _ in source_files}
        missing = sorted(requested - available)
        if missing:
            raise FileNotFoundError(f'Missing {args.source} JSON files for requested videos: {missing}')
        source_files = [(video_id, records) for video_id, records in source_files
                        if video_id in requested]
    if args.source == 'caption':
        all_records = [record for _, records in source_files for record in records]
        maps = SegmentMaps(data_root, args.dataset, 'caption', all_records)
        source_files = [(video_id, [record for record in records
                                    if maps.forward.get(record['segment_id'])])
                       for video_id, records in source_files]
    if not source_files:
        raise RuntimeError(f'No indexable {args.source} segments found for {args.dataset}')

    encoder = QwenVllmEncoder()
    try:
        target_root = output_root / args.source / args.dataset.upper()
        total = sum(len(records) for _, records in source_files)
        completed = 0
        for video_id, records in source_files:
            vectors = np.empty((len(records), encoder.embedding_dim), dtype=np.float32)
            for offset in range(0, len(records), args.batch_size):
                batch = records[offset:offset + args.batch_size]
                encoded = encoder.encode_text([record['text'] for record in batch])
                batch_vectors = encoded.detach().cpu().numpy().astype(np.float32, copy=False)
                if batch_vectors.shape != (len(batch), encoder.embedding_dim):
                    raise RuntimeError(f'Invalid embedding batch for {args.source}/{video_id}: {batch_vectors.shape}')
                vectors[offset:offset + len(batch)] = batch_vectors
                completed += len(batch)
                print(f'{args.source}/{video_id}: embedded {completed}/{total}', flush=True)
            target = target_root / f'{video_id}.npy'
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary_vectors = target.with_suffix('.npy.tmp')
            with temporary_vectors.open('wb') as stream:
                np.save(stream, vectors, allow_pickle=False)
            temporary_vectors.replace(target)
    finally:
        encoder.close()

    print(f'Saved per-video Qwen vLLM matrices under: {target_root}', flush=True)


if __name__ == '__main__':
    main()
