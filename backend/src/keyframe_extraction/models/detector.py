from __future__ import annotations

from pathlib import Path
import logging

import numpy as np
import torch
from tqdm import tqdm

from src.utils.device import resolve_device
from src.utils.video_io import decode_for_transnet
from src.keyframe_extraction.models.transnetv2 import TransNetV2


def load_transnet(repo_dir: Path, weights_path: Path, device_name: str = "auto", logger: logging.Logger | None = None):
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


def predictions_to_scenes_p3(predictions: np.ndarray, threshold: float) -> np.ndarray:
	"""Convert transition runs to cuts without changing legacy LMSKE behavior."""
	preds = predictions.reshape(-1)
	mask = preds > threshold
	padded = np.pad(mask.astype(np.int8), (1, 1))
	starts = np.where(np.diff(padded) == 1)[0]
	ends = np.where(np.diff(padded) == -1)[0] - 1
	cuts = [int(round((start + end) / 2)) for start, end in zip(starts, ends)]
	scenes: list[tuple[int, int]] = []
	start = 0
	for cut in cuts:
		if cut >= start:
			scenes.append((start, cut))
			start = cut + 1
	if start < len(preds):
		scenes.append((start, len(preds) - 1))
	if not scenes and len(preds) > 0:
		scenes = [(0, len(preds) - 1)]
	return np.asarray(scenes, dtype=np.int32)


def repair_and_split_scenes(scenes: np.ndarray, fps: float, max_duration_sec: float) -> np.ndarray:
	"""Normalize inclusive scene ranges and split shots longer than the configured duration."""
	if fps <= 0:
		raise ValueError("fps must be positive")
	max_frames = max(1, int(round(max_duration_sec * fps)))
	result: list[tuple[int, int]] = []
	for raw_start, raw_end in np.asarray(scenes, dtype=np.int32).reshape(-1, 2):
		start, end = int(raw_start), int(raw_end)
		if end < start:
			start, end = end, start
		while end - start + 1 > max_frames:
			result.append((start, start + max_frames - 1))
			start += max_frames
		result.append((start, end))
	return np.asarray(result, dtype=np.int32).reshape(-1, 2)


def detect_scenes(
	model,
	device: torch.device,
	video_path: Path,
	batch_size: int,
	threshold: float,
	output_path: Path,
	logger: logging.Logger | None = None,
	merge_transition_runs: bool = False,
) -> np.ndarray:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	frames_np = decode_for_transnet(video_path)
	predictions: list[np.ndarray] = []

	if logger:
		logger.info("Detecting scenes: %s frames=%d threshold=%.3f", video_path.name, len(frames_np), threshold)

	with torch.inference_mode():
		if merge_transition_runs:
			context = 25
			if batch_size <= 2 * context:
				raise ValueError("TransNet batch_size must be greater than 50 for P3 context windows")
			core_size = batch_size - 2 * context
			padded = np.concatenate(
				[
					np.repeat(frames_np[:1], context, axis=0),
					frames_np,
					np.repeat(frames_np[-1:], context, axis=0),
				],
				axis=0,
			)
			frames = torch.from_numpy(padded)
			for start in tqdm(
				range(0, len(frames_np), core_size), desc=f"TransNetV2 {video_path.name}", unit="batch"
			):
				core_length = min(core_size, len(frames_np) - start)
				batch = frames[start : start + core_length + 2 * context].unsqueeze(0).to(
					device, non_blocking=True
				)
				single, _ = model(batch)
				predictions.append(
					torch.sigmoid(single)[0, context : context + core_length].detach().cpu().numpy()
				)
		else:
			frames = torch.from_numpy(frames_np)
			for start in tqdm(
				range(0, len(frames), batch_size), desc=f"TransNetV2 {video_path.name}", unit="batch"
			):
				batch = frames[start : start + batch_size].unsqueeze(0).to(device, non_blocking=True)
				single, _ = model(batch)
				predictions.append(torch.sigmoid(single)[0].detach().cpu().numpy())

	converter = predictions_to_scenes_p3 if merge_transition_runs else predictions_to_scenes
	scenes = converter(np.concatenate(predictions, axis=0), threshold)
	np.savetxt(output_path, scenes, fmt="%d")

	if logger:
		logger.info("Saved scenes: %s scenes=%d", output_path, len(scenes))
	return scenes
