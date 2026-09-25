"""Rebuild PE-Core and SigLIP2 FAISS indexes and aligned vector caches.

Usage (from repository root):
    env/Scripts/python.exe backend/scripts/retrieval/rebuild_pe_siglip.py

Builds one dataset/model at a time with bounded batches and stages output
files before replacing the current index, metadata, and cache.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import faiss
import numpy as np
from numpy.lib.format import open_memmap


REPO = Path(__file__).resolve().parents[3]
BACKEND = REPO / "backend"
sys.path.insert(0, str(BACKEND))

from src.retrieval.indexer.faiss.index_builder import (  # noqa: E402
    find_keyframe_path,
    parse_embedding_path,
)

MODELS = {
    "pe": "pe_core_l14_336",
    "siglip2": "siglip2_so400m_384",
}
FIELDS = [
    "collection", "dataset", "video_id", "keyframe_id", "keyframe_id_int",
    "source_name", "frame_idx", "timestamp_sec", "fps", "keyframe_path",
    "embedding_path", "map_path",
]


def map_rows(path: Path, video_id: str) -> dict[int, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = csv.DictReader(stream)
        result = {}
        for row in rows:
            if row.get("video_id", video_id) != video_id:
                continue
            raw_id = row.get("keyframe_id") or row.get("idx")
            if raw_id is not None and raw_id != "":
                result[int(float(raw_id))] = row
        return result


def build(dataset: str, model: str, batch_size: int, threads: int) -> None:
    tag = MODELS[model]
    collection = dataset.upper()
    data_root = REPO / "data" / collection
    embedding_root = data_root / "embeddings" / f"embeddings_{tag}"
    keyframes_root = data_root / "keyframes"
    map_root = data_root / ("map_keyframes" if dataset == "aic" else "map-keyframes")
    output_dir = data_root / "database" / f"faiss_hnsw_{tag}"
    paths = sorted(embedding_root.rglob("*.npy"))
    if not paths:
        raise RuntimeError(f"No embeddings: {embedding_root}")
    first = np.load(paths[0], mmap_mode="r")
    dimension = int(first.size)
    count = len(paths)
    print(f"[{dataset}/{model}] {count} embeddings, dimension={dimension}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    staged_index = output_dir / "keyframes.faiss.building"
    staged_meta = output_dir / "metadata.csv.building"
    staged_cache = output_dir / "vectors_fp32.npy.building"

    faiss.omp_set_num_threads(threads)
    index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 820
    index.hnsw.efSearch = 512
    vectors = open_memmap(staged_cache, mode="w+", dtype=np.float32, shape=(count, dimension))
    current_video = None
    current_map: dict[int, dict[str, str]] = {}
    batch = np.empty((batch_size, dimension), dtype=np.float32)
    started = time.monotonic()
    written = 0

    with staged_meta.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for position, path in enumerate(paths):
            group, video_id, parsed = parse_embedding_path(path, embedding_root)
            video = (group, video_id)
            if video != current_video:
                current_video = video
                current_map = map_rows(map_root / group / f"{video_id}.csv", video_id)
            vector = np.load(path).astype(np.float32).reshape(-1)
            if vector.size != dimension or not np.isfinite(vector).all():
                raise ValueError(f"Invalid embedding shape/value: {path}")
            batch[position - written] = vector
            map_row = current_map.get(parsed["keyframe_id_int"], {})
            keyframe_path = find_keyframe_path(keyframes_root, group, video_id, parsed)
            if not keyframe_path.is_file():
                raise FileNotFoundError(f"Keyframe missing for {path}: {keyframe_path}")
            writer.writerow({
                "collection": dataset,
                "dataset": group,
                "video_id": video_id,
                "keyframe_id": parsed["keyframe_id_str"],
                "keyframe_id_int": parsed["keyframe_id_int"],
                "source_name": parsed["source_name"],
                "frame_idx": map_row.get("frame_idx", parsed.get("frame_idx_from_name")),
                "timestamp_sec": map_row.get("timestamp_sec", map_row.get("timestamp")),
                "fps": map_row.get("fps"),
                "keyframe_path": str(keyframe_path),
                "embedding_path": str(path),
                "map_path": str(map_root / group / f"{video_id}.csv"),
            })
            if (position + 1 - written) == batch_size or position + 1 == count:
                size = position + 1 - written
                chunk = batch[:size]
                faiss.normalize_L2(chunk)
                index.add(chunk)
                vectors[written:position + 1] = chunk
                written = position + 1
                if written % (batch_size * 10) == 0 or written == count:
                    vectors.flush()
                    stream.flush()
                    elapsed = time.monotonic() - started
                    print(f"[{dataset}/{model}] {written}/{count} indexed in {elapsed:.0f}s", flush=True)

    vectors.flush()
    del vectors
    if index.ntotal != count:
        raise RuntimeError(f"Index count {index.ntotal} != source count {count}")
    faiss.write_index(index, str(staged_index))
    del index
    check = np.load(staged_cache, mmap_mode="r")
    if check.shape != (count, dimension):
        raise RuntimeError(f"Cache shape {check.shape} != {(count, dimension)}")
    del check
    for staged, final in (
        (staged_index, output_dir / "keyframes.faiss"),
        (staged_meta, output_dir / "metadata.csv"),
        (staged_cache, output_dir / "vectors_fp32.npy"),
    ):
        os.replace(staged, final)
    print(f"[{dataset}/{model}] complete: {output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["aic", "cam"], required=True)
    parser.add_argument("--model", choices=list(MODELS), required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    build(args.dataset, args.model, args.batch_size, args.threads)


if __name__ == "__main__":
    main()
