from __future__ import annotations

from pathlib import Path
import gc
import logging

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.embedding_extraction.models.backends.base import LoadedModel
from src.embedding_extraction.models.registry import load_model as _load_model


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def load_embedding_model(
	model_name: str,
	pretrained: str | None,
	precision: str,
	device_name: str,
	backend: str | None = None,
	logger: logging.Logger | None = None,
	**extra,
) -> tuple:
	"""Load any supported embedding model (open_clip, hf_clip, blip2, beit3, ...).

	Kept as `load_clip_model` / returning the same 4-tuple shape for backward
	compatibility with existing callers - dispatch now goes through
	`src.embedding_extraction.models.registry.load_model`, which picks the
	right backend based on `backend` (or a preset keyed by `model_name`).
	"""
	loaded: LoadedModel = _load_model(
		model_name=model_name,
		backend=backend,
		pretrained=pretrained,
		precision=precision,
		device_name=device_name,
		logger=logger,
		**extra,
	)
	return loaded.model, loaded.preprocess, loaded.device, loaded.precision


def list_image_paths(image_dir: Path) -> list[Path]:
	return sorted(
		p for p in image_dir.iterdir()
		if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
	)


def _load_image_tensor(
	image_path: Path,
	preprocess,
	logger: logging.Logger | None = None,
):
	image = cv2.imread(str(image_path))

	if image is None:
		if logger:
			logger.warning("Cannot read image: %s", image_path)
		return None

	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = Image.fromarray(image)

	return preprocess(image)


def _build_image_batch(
	image_paths: list[Path],
	preprocess,
	logger: logging.Logger | None = None,
):
	tensors = []
	valid_paths = []

	for path in image_paths:
		tensor = _load_image_tensor(path, preprocess, logger=logger)

		if tensor is None:
			continue

		tensors.append(tensor)
		valid_paths.append(path)

	if not tensors:
		return None, []

	return torch.stack(tensors), valid_paths


@torch.inference_mode()
def encode_keyframe_images(
	model,
	preprocess,
	device: torch.device,
	image_paths: list[Path],
	batch_size: int,
	precision: str,
	normalize: bool = True,
	logger: logging.Logger | None = None,
) -> tuple[np.ndarray, list[Path]]:
	features: list[np.ndarray] = []
	valid_all_paths: list[Path] = []

	use_amp = device.type == "cuda" and precision == "amp"

	for start in tqdm(
		range(0, len(image_paths), batch_size),
		desc="Embed keyframes",
		unit="batch",
	):
		batch_paths = image_paths[start:start + batch_size]

		batch, valid_paths = _build_image_batch(
			image_paths=batch_paths,
			preprocess=preprocess,
			logger=logger,
		)

		if batch is None:
			continue

		batch = batch.to(device, non_blocking=True)

		with torch.autocast(
			device_type="cuda",
			dtype=torch.float16,
			enabled=use_amp,
		):
			emb = model.encode_image(batch)

			if normalize:
				emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)

		features.append(emb.float().cpu().numpy())
		valid_all_paths.extend(valid_paths)

	if not features:
		return np.empty((0, 0), dtype=np.float32), []

	arr = np.vstack(features).astype(np.float32)

	gc.collect()
	if device.type == "cuda":
		torch.cuda.empty_cache()

	return arr, valid_all_paths


def load_embedding(path: Path) -> np.ndarray:
	return np.load(path).astype(np.float32)

