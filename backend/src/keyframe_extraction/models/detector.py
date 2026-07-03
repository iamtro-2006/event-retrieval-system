from __future__ import annotations

from pathlib import Path
import logging
import sys

import numpy as np
import torch
from tqdm import tqdm

from src.utils.device import resolve_device
from src.utils.video_io import decode_for_transnet


def load_transnet(repo_dir: Path, weights_path: Path, device_name: str = "auto", logger: logging.Logger | None = None):
	sys.path.insert(0, str(repo_dir.parent.resolve()))
	sys.path.insert(0, str(repo_dir.resolve()))
	from external.TransNetV2.inference_pytorch.transnetv2_pytorch import TransNetV2

	if not weights_path.exists():
		raise FileNotFoundError(f"TransNetV2 weights not found: {weights_path}")

	device = resolve_device(device_name)
	if logger:
		logger.info("Loading TransNetV2 from %s on %s", weights_path, device)
	model = TransNetV2()
	state = torch.load(str(weights_path), map_location=device)
	model.load_state_dict(state)
	return model.to(device).eval(), device


def predictions_to_scenes(predictions: np.ndarray, threshold: float) -> np.ndarray:
	preds = predictions.reshape(-1)
	cuts = np.where(preds > threshold)[0]
	scenes: list[tuple[int, int]] = []
	start = 0
	for cut in cuts:
		cut = int(cut)
		if cut > start:
			scenes.append((start, cut))
			start = cut + 1
	if start < len(preds):
		scenes.append((start, len(preds) - 1))
	if not scenes and len(preds) > 0:
		scenes = [(0, len(preds) - 1)]
	return np.asarray(scenes, dtype=np.int32)


def detect_scenes(
	model,
	device: torch.device,
	video_path: Path,
	batch_size: int,
	threshold: float,
	output_path: Path,
	logger: logging.Logger | None = None,
) -> np.ndarray:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	frames_np = decode_for_transnet(video_path)
	frames = torch.from_numpy(frames_np)
	predictions: list[np.ndarray] = []

	if logger:
		logger.info("Detecting scenes: %s frames=%d threshold=%.3f", video_path.name, len(frames), threshold)

	with torch.inference_mode():
		for start in tqdm(range(0, len(frames), batch_size), desc=f"TransNetV2 {video_path.name}", unit="batch"):
			batch = frames[start:start + batch_size].unsqueeze(0).to(device, non_blocking=True)
			single, _ = model(batch)
			predictions.append(torch.sigmoid(single)[0].detach().cpu().numpy())

	scenes = predictions_to_scenes(np.concatenate(predictions, axis=0), threshold)
	np.savetxt(output_path, scenes, fmt="%d")

	if logger:
		logger.info("Saved scenes: %s scenes=%d", output_path, len(scenes))
	return scenes

