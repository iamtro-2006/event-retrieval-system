"""Encoder wrapper forwarding to existing src.models.encoder"""

from __future__ import annotations
from pathlib import Path
import gc
import logging
import pickle

import numpy as np
import torch
from tqdm import tqdm
from src.utils.config import FrameLoaderConfig
from src.embedding_extraction.models.frame_loader import build_frame_loader, iter_frame_batches
from src.embedding_extraction.models.registry import load_model as _load_model

def load_clip_model(model_name: str, pretrained: str | None, precision: str = "fp32", device_name: str = "auto", backend: str | None = None, logger: logging.Logger | None = None, **extra):
	"""Same dispatch as embedder.py's load_clip_model, kept here too so the
	video-frame encoding path (encode_video_frames below) can also load any
	registered backend (open_clip / hf_clip / blip2 / beit3), not just open_clip.
	"""
	loaded = _load_model(
		model_name=model_name,
		backend=backend,
		pretrained=pretrained,
		precision=precision,
		device_name=device_name,
		logger=logger,
		**extra,
	)
	return loaded.model, loaded.preprocess, loaded.device

def encode_video_frames(
	model,
	preprocess,
	device: torch.device,
	video_path: Path,
	output_path: Path,
	batch_size: int,
	loader_cfg: FrameLoaderConfig,
	logger: logging.Logger | None = None,
) -> np.ndarray:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	loader = build_frame_loader(
		video_path=video_path,
	transform=preprocess,
	batch_size=batch_size,
	backend=loader_cfg.backend,
	num_workers=loader_cfg.num_workers,
	pin_memory=loader_cfg.pin_memory and device.type == "cuda",
	persistent_workers=loader_cfg.persistent_workers,
	)

	if logger:
		logger.info("Encoding %s: frames=%d batch_size=%d workers=%d", video_path.name, len(loader.dataset), batch_size, loader_cfg.num_workers)
	features: list[np.ndarray] = []
	use_amp = device.type == "cuda"

	with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
		for batch in tqdm(iter_frame_batches(loader), total=len(loader), desc=f"Embed {video_path.name}", unit="batch"):
			frames = batch.frames.to(device, non_blocking=True)
			if use_amp:
				frames = frames.half()
			emb = model.encode_image(frames)
			emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
			features.append(emb.float().cpu().numpy())

	if not features:
		raise RuntimeError(f"No frames encoded from {video_path}")
	arr = np.vstack(features).astype(np.float32)
	with output_path.open("wb") as f:
		pickle.dump(arr, f, protocol=pickle.HIGHEST_PROTOCOL)
	if logger:
		logger.info("Saved embeddings: %s shape=%s", output_path, arr.shape)

	gc.collect()
	if device.type == "cuda":
		torch.cuda.empty_cache()
	return arr

def load_embeddings(path: Path) -> np.ndarray:
	with path.open("rb") as f:
		return np.asarray(pickle.load(f), dtype=np.float32)
