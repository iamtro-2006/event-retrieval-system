"""Encoder wrapper forwarding to existing src.models.encoder"""

from __future__ import annotations
from pathlib import Path
import gc
import logging
import pickle

import numpy as np
import open_clip
import torch
from tqdm import tqdm
from src.utils.config import FrameLoaderConfig
from src.embeddings.models.frame_loader import build_frame_loader, iter_frame_batches
from src.utils.device import resolve_device

def load_clip_model(model_name: str, pretrained: str, precision: str, device_name: str, logger: logging.Logger | None = None):
	device = resolve_device(device_name)
	if logger:
		logger.info("Loading embedding model: %s / %s on %s", model_name, pretrained, device)
	model, _, preprocess = open_clip.create_model_and_transforms(
		model_name,
		pretrained=pretrained,
	)
	model = model.to(device).eval()
	return model, preprocess, device

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
