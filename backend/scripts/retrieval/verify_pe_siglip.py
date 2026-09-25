"""Verify FAISS, metadata, and vector cache against source embeddings."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import faiss
import numpy as np


REPO = Path(__file__).resolve().parents[3]
MODELS = {"pe": "pe_core_l14_336", "siglip2": "siglip2_so400m_384"}


def verify(dataset: str, model: str) -> None:
    root = REPO / "data" / dataset.upper()
    tag = MODELS[model]
    source = root / "embeddings" / f"embeddings_{tag}"
    output = root / "database" / f"faiss_hnsw_{tag}"
    source_paths = sorted(source.rglob("*.npy"))
    index = faiss.read_index(str(output / "keyframes.faiss"))
    cache = np.load(output / "vectors_fp32.npy", mmap_mode="r")
    expected = len(source_paths)
    assert index.ntotal == expected, (index.ntotal, expected)
    assert cache.shape == (expected, index.d), (cache.shape, expected, index.d)
    assert cache.dtype == np.float32, cache.dtype
    assert index.metric_type == faiss.METRIC_INNER_PRODUCT, index.metric_type

    video_counts: Counter[str] = Counter()
    metadata_count = 0
    sample_positions = set(np.linspace(0, expected - 1, min(50, expected), dtype=int))
    with (output / "metadata.csv").open(newline="", encoding="utf-8-sig") as stream:
        for position, row in enumerate(csv.DictReader(stream)):
            metadata_count += 1
            video_counts[f"{row['dataset']}/{row['video_id']}"] += 1
            assert Path(row["embedding_path"]) == source_paths[position], (position, row["embedding_path"])
            assert Path(row["keyframe_path"]).is_file(), (position, row["keyframe_path"])
            assert row["collection"] == dataset, (position, row["collection"])
            if position in sample_positions:
                raw = np.load(source_paths[position]).astype(np.float32).reshape(-1)
                raw /= max(np.linalg.norm(raw), 1e-12)
                reconstructed = index.reconstruct(position)
                assert np.allclose(cache[position], raw, atol=1e-5), (position, "cache/source")
                assert np.allclose(reconstructed, raw, atol=1e-5), (position, "index/source")
    assert metadata_count == expected, (metadata_count, expected)
    source_videos = Counter(
        f"{path.relative_to(source).parts[0]}/{path.relative_to(source).parts[1]}"
        for path in source_paths
    )
    assert video_counts == source_videos, "Per-video metadata counts differ from source"
    print(f"PASS {dataset}/{model}: {expected} vectors, {len(source_videos)} videos, "
          f"dimension={index.d}, {len(sample_positions)} sampled vectors match", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["aic", "cam"], required=True)
    parser.add_argument("--model", choices=list(MODELS), required=True)
    args = parser.parse_args()
    verify(args.dataset, args.model)


if __name__ == "__main__":
    main()
