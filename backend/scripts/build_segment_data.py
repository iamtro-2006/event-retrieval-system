"""python -m scripts.build_segment_data --dataset v3c [--maps-only]."""
import argparse
import json
from pathlib import Path

from src.retrieval.segment_data import build_maps, fingerprint, load_segments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['v3c', 'lhe'], required=True)
    parser.add_argument('--data-root', type=Path, default=Path('../data'))
    parser.add_argument('--model', default='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    parser.add_argument('--maps-only', action='store_true')
    parser.add_argument('--fallback-tau', type=float, default=5.)
    parser.add_argument('--fallback-right-weight', type=float, default=2.)
    args = parser.parse_args()
    root = args.data_root.resolve()
    print('Mapping:', build_maps(root, args.dataset, args.fallback_tau, args.fallback_right_weight), flush=True)
    if args.maps_only:
        return
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)
    for source in ('asr', 'caption'):
        records = list(load_segments(root, args.dataset, source))
        if not records:
            print(f'{source}: no documents; skipped', flush=True)
            continue
        # Long captions must not silently lose the end of their event description.
        # Pool token-bounded chunks back to one vector per original segment.
        limit = max(16, int(model.max_seq_length) - 8)
        chunks, owners = [], []
        for owner, record in enumerate(records):
            tokens = model.tokenizer.encode(record['text'], add_special_tokens=False)
            for start in range(0, max(1, len(tokens)), limit):
                chunks.append(model.tokenizer.decode(tokens[start:start+limit], skip_special_tokens=True))
                owners.append(owner)
        encoded = np.asarray(model.encode(chunks, normalize_embeddings=True, show_progress_bar=True), dtype='float32')
        vectors = np.zeros((len(records), encoded.shape[1]), dtype='float32')
        np.add.at(vectors, owners, encoded)
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        # Versioned directory + atomic pointer prevents a half-written index being loaded.
        version = fingerprint(records + [{'model': args.model, 'pooling': 'chunk_mean_v1'}])[:20]
        base = root / 'database' / f'{source}_{args.dataset}'
        folder = base / version
        folder.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(folder / 'segments.faiss'))
        np.save(folder / 'vectors.npy', vectors)
        (folder / 'metadata.json').write_text(json.dumps({'model': args.model,
            'source_hash': fingerprint(records), 'records': records}, ensure_ascii=False), encoding='utf-8')
        temporary = base / 'CURRENT.tmp'
        temporary.write_text(version)
        temporary.replace(base / 'CURRENT')
        print(f'{source}: {len(records)} documents → {folder}', flush=True)


if __name__ == '__main__':
    main()
