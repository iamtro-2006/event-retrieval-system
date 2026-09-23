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
	if len(preds) == 0:
		return np.empty((0, 2), dtype=np.int32)

	# A gradual transition commonly stays above the threshold for several
	# consecutive frames. Treat that run as one transition and place the cut at
	# its midpoint instead of creating a tiny scene for every positive frame.
	mask = preds > threshold
	padded = np.pad(mask.astype(np.int8), (1, 1))
	run_starts = np.where(np.diff(padded) == 1)[0]
	run_ends = np.where(np.diff(padded) == -1)[0] - 1
	cuts = np.rint((run_starts + run_ends) / 2.0).astype(np.int32)
	scenes: list[tuple[int, int]] = []
	start = 0
	for cut in cuts:
		cut = int(cut)
		if cut >= start:
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
	context_frames: int = 25,
) -> np.ndarray:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	frames_np = decode_for_transnet(video_path)
	if context_frames < 0:
		raise ValueError("context_frames must be non-negative")
	if batch_size <= 2 * context_frames:
		raise ValueError("batch_size must be greater than 2 * context_frames")

	core_size = batch_size - 2 * context_frames
	if context_frames:
		padded_np = np.concatenate(
			[
				np.repeat(frames_np[:1], context_frames, axis=0),
				frames_np,
				np.repeat(frames_np[-1:], context_frames, axis=0),
			],
			axis=0,
		)
	else:
		padded_np = frames_np
	frames = torch.from_numpy(padded_np)
	predictions: list[np.ndarray] = []

	if logger:
		logger.info(
			"Detecting scenes: %s frames=%d threshold=%.3f window=%d context=%d",
			video_path.name,
			len(frames_np),
			threshold,
			batch_size,
			context_frames,
		)

	with torch.inference_mode():
		for start in tqdm(range(0, len(frames_np), core_size), desc=f"TransNetV2 {video_path.name}", unit="batch"):
			core_length = min(core_size, len(frames_np) - start)
			window_length = core_length + 2 * context_frames
			batch = frames[start:start + window_length].unsqueeze(0).to(device, non_blocking=True)
			single, _ = model(batch)
			core = torch.sigmoid(single)[0, context_frames:context_frames + core_length, 0]
			predictions.append(core.detach().cpu().numpy())

	scenes = predictions_to_scenes(np.concatenate(predictions, axis=0), threshold)
	np.savetxt(output_path, scenes, fmt="%d")

	if logger:
		logger.info("Saved scenes: %s scenes=%d", output_path, len(scenes))
	return scenes

