from __future__ import annotations

from pathlib import Path
import re

import faiss
import numpy as np
import pandas as pd


IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]


def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
	norms = np.linalg.norm(matrix, axis=1, keepdims=True)
	return matrix / np.clip(norms, 1e-12, None)


def collect_embedding_files(embeddings_root: str | Path) -> list[Path]:
	embeddings_root = Path(embeddings_root)

	if not embeddings_root.exists():
		raise FileNotFoundError(f"Embeddings root not found: {embeddings_root}")

	return sorted(embeddings_root.rglob("*.npy"))


def parse_keyframe_stem(stem: str) -> dict:
	if stem.isdigit():
		return {
			"source_name": stem,
			"keyframe_id_int": int(stem),
			"keyframe_id_str": stem.zfill(6),
			"frame_idx_from_name": None,
		}

	match = re.search(r"debug_(\d+)_frame_(\d+)", stem)

	if match:
		keyframe_id = int(match.group(1))
		frame_idx = int(match.group(2))

		return {
			"source_name": stem,
			"keyframe_id_int": keyframe_id,
			"keyframe_id_str": f"{keyframe_id:06d}",
			"frame_idx_from_name": frame_idx,
		}

	numbers = re.findall(r"\d+", stem)

	if numbers:
		keyframe_id = int(numbers[0])
		frame_idx = int(numbers[-1]) if len(numbers) > 1 else None

		return {
			"source_name": stem,
			"keyframe_id_int": keyframe_id,
			"keyframe_id_str": f"{keyframe_id:06d}",
			"frame_idx_from_name": frame_idx,
		}

	raise ValueError(f"Cannot parse keyframe stem: {stem}")


def parse_embedding_path(embedding_path: Path, embeddings_root: Path):
	rel = embedding_path.relative_to(embeddings_root)

	if len(rel.parts) < 3:
		raise ValueError(f"Invalid embedding path: {embedding_path}")

	dataset = rel.parts[0]
	video_id = rel.parts[1]
	parsed = parse_keyframe_stem(embedding_path.stem)

	return dataset, video_id, parsed


def find_map_path(map_root: Path, dataset: str, video_id: str) -> Path:
	return map_root / dataset / f"{video_id}.csv"


def find_keyframe_path(
	keyframes_root: Path,
	dataset: str,
	video_id: str,
	parsed: dict,
) -> Path:
	video_dir = keyframes_root / dataset / video_id

	candidate_names = [
		parsed["source_name"],
		parsed["keyframe_id_str"],
		str(parsed["keyframe_id_int"]),
	]

	for name in candidate_names:
		for ext in IMAGE_EXTENSIONS:
			path = video_dir / f"{name}{ext}"
			if path.exists():
				return path

	return video_dir / f"{parsed['keyframe_id_str']}.jpg"


def normalize_map_df(df: pd.DataFrame, video_id: str) -> pd.DataFrame:
	df = df.copy()

	if "keyframe_id" not in df.columns and "idx" in df.columns:
		df["keyframe_id"] = df["idx"]

	if "timestamp_sec" not in df.columns and "timestamp" in df.columns:
		df["timestamp_sec"] = df["timestamp"]

	if "video_id" not in df.columns:
		df["video_id"] = video_id

	return df


def load_map_row(
	map_path: Path,
	video_id: str,
	parsed: dict,
) -> dict:
	if not map_path.exists():
		return {}

	df = pd.read_csv(map_path)

	if df.empty:
		return {}

	df = normalize_map_df(df, video_id)

	df = df[df["video_id"].astype(str) == str(video_id)]

	if df.empty:
		return {}

	keyframe_id = parsed["keyframe_id_int"]
	frame_idx = parsed.get("frame_idx_from_name")

	if "keyframe_id" in df.columns:
		rows = df[df["keyframe_id"].astype(int) == int(keyframe_id)]

		if not rows.empty:
			return rows.iloc[0].to_dict()

	if frame_idx is not None and "frame_idx" in df.columns:
		rows = df[df["frame_idx"].astype(int) == int(frame_idx)]

		if not rows.empty:
			return rows.iloc[0].to_dict()

	return {}


def build_matrix_and_metadata(
	embeddings_root: str | Path,
	keyframes_root: str | Path,
	map_keyframes_root: str | Path,
):
	embeddings_root = Path(embeddings_root)
	keyframes_root = Path(keyframes_root)
	map_root = Path(map_keyframes_root)

	embedding_paths = collect_embedding_files(embeddings_root)

	if not embedding_paths:
		raise RuntimeError(f"No embedding files found in: {embeddings_root}")

	vectors = []
	records = []

	for embedding_path in embedding_paths:
		dataset, video_id, parsed = parse_embedding_path(
			embedding_path=embedding_path,
			embeddings_root=embeddings_root,
		)

		embedding = np.load(embedding_path).astype(np.float32).reshape(-1)
		vectors.append(embedding)

		map_path = find_map_path(
			map_root=map_root,
			dataset=dataset,
			video_id=video_id,
		)

		keyframe_path = find_keyframe_path(
			keyframes_root=keyframes_root,
			dataset=dataset,
			video_id=video_id,
			parsed=parsed,
		)

		map_row = load_map_row(
			map_path=map_path,
			video_id=video_id,
			parsed=parsed,
		)

		records.append({
			"dataset": dataset,
			"video_id": video_id,
			"keyframe_id": parsed["keyframe_id_str"],
			"keyframe_id_int": parsed["keyframe_id_int"],
			"source_name": parsed["source_name"],
			"frame_idx": map_row.get("frame_idx", parsed.get("frame_idx_from_name")),
			"timestamp_sec": map_row.get("timestamp_sec", None),
			"fps": map_row.get("fps", None),
			"keyframe_path": str(keyframe_path),
			"embedding_path": str(embedding_path),
			"map_path": str(map_path),
		})

	matrix = np.vstack(vectors).astype(np.float32)
	metadata = pd.DataFrame(records)

	return matrix, metadata


def create_faiss_index(
	matrix: np.ndarray,
	metric: str = "cosine",
	index_type: str = "hnsw",
	hnsw_m: int = 32,
	ef_construction: int = 200,
	ef_search: int = 64,
):
	"""Build Flat or HNSW index. Cosine uses normalized vectors + inner product."""
	if matrix.ndim != 2:
		raise ValueError(f"Matrix must be 2D, got shape={matrix.shape}")
	matrix = np.ascontiguousarray(matrix, dtype=np.float32)
	dim = matrix.shape[1]
	metric_name = str(metric).lower()
	metric_type = faiss.METRIC_INNER_PRODUCT if metric_name == "cosine" else faiss.METRIC_L2
	if metric_name == "cosine":
		faiss.normalize_L2(matrix)
	elif metric_name != "l2":
		raise ValueError(f"Unsupported metric: {metric}")

	if str(index_type).lower() == "hnsw":
		index = faiss.IndexHNSWFlat(dim, int(hnsw_m), metric_type)
		index.hnsw.efConstruction = int(ef_construction)
		index.hnsw.efSearch = int(ef_search)
	elif str(index_type).lower() == "flat":
		index = faiss.IndexFlatIP(dim) if metric_type == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(dim)
	else:
		raise ValueError(f"Unsupported index_type: {index_type}")

	index.add(matrix)
	return index


def save_faiss_index(
	index,
	metadata: pd.DataFrame,
	output_dir: str | Path,
	index_name: str,
	metadata_name: str,
):
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	index_path = output_dir / index_name
	metadata_path = output_dir / metadata_name

	faiss.write_index(index, str(index_path))
	metadata.to_csv(metadata_path, index=False, encoding="utf-8-sig")

	return index_path, metadata_path

