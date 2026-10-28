"""Original SpeedPredictor sampling/three-pass estimator, with NVDEC/CUDA I/O.

Reference: Seeing-Fast-and-Slow/speed_perception/{dataset,utils,
video_speed_estimation}.py. No temporal position interpolation. The only
short-input extension is to return unknown instead of failing on zero windows.
"""
from __future__ import annotations

import gc
import json
import logging
import math
from pathlib import Path

import torch
from torch import nn
from torchvision.transforms.functional import resize

from src.utils.gpu_video import CudaVideoReader
from .speed_architecture import VisionTransformer


def original_window_slices(frame_count: int, fps: float, duration: float):
    """Preserve upstream floor-division window count and end-anchored last clip."""
    video_length = frame_count / fps
    clip_length = int(duration * fps)
    if video_length < duration or clip_length < 1:
        return []
    count = math.ceil((video_length - duration) // duration) + 1
    return [(int(duration * i * fps), int(duration * i * fps) + clip_length)
            if i < count - 1 else (frame_count - clip_length, frame_count)
            for i in range(count)]


def speed_label(score: float | None, cfg: dict) -> str:
    if score is None:
        return "unknown"
    if score > cfg["timelapse_threshold"]:
        return "timelapse"
    if score < cfg["slow_motion_threshold"]:
        return "slow-motion"
    return "normal"


class SpeedModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.videomaev2 = nn.Module()
        self.videomaev2.model = VisionTransformer(**config["model_config"])
        self.fc_norm = nn.LayerNorm(config["hidden_size"])
        self.classifier = nn.Linear(config["hidden_size"], 1)
        self.beta = nn.Parameter(torch.zeros(()))

    def forward(self, frames):
        return self.classifier(self.fc_norm(self.videomaev2.model(frames)))


class ShotSpeedPredictor:
    def __init__(self, cfg: dict, logger=None):
        if not torch.cuda.is_available():
            raise RuntimeError("SpeedPredictor requires CUDA; CPU fallback is disabled")
        torch.cuda.set_device(0)
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)
        self.sample_fps = 30 / 4  # Upstream reference cadence, NOT the source FPS.
        self.recursion = int(cfg["recursion"])
        if self.recursion != 3:
            raise ValueError("Original inference configuration requires three passes")
        if not 0 < cfg["slow_motion_threshold"] < cfg["timelapse_threshold"]:
            raise ValueError("Invalid speed thresholds")
        self.logger.info("[SPEED] Loading %s revision=%s on CUDA FP32", cfg["model"], cfg["revision"])
        def download(name):
            return hf_hub_download(cfg["model"], name, revision=cfg["revision"])
        config = json.loads(Path(download("config.json")).read_text())
        pre = json.loads(Path(download("preprocessor_config.json")).read_text())
        self.count = int(config["model_config"]["num_frames"])
        self.span = self.count / self.sample_fps
        self.model = SpeedModel(config)
        self.model.load_state_dict(load_file(download("model.safetensors")), strict=True)
        self.model.eval().to("cuda:0")
        self.model.videomaev2.model.pos_embed = self.model.videomaev2.model.pos_embed.to("cuda:0")
        self.size = int(config["model_config"]["img_size"])
        self.mean = torch.tensor(pre["image_mean"], device="cuda:0").view(1, 3, 1, 1)
        self.std = torch.tensor(pre["image_std"], device="cuda:0").view(1, 3, 1, 1)
        self.logger.info("[SPEED] Ready: frames=%d reference_fps=%.2f base_window=%.6fs passes=%d; no position interpolation",
                         self.count, self.sample_fps, self.span, self.recursion)

    def _estimate(self, frames, fps: float, context: str) -> float | None:
        """frames is GPU TCHW; resampling, model and preprocessing stay on CUDA."""
        length = len(frames) / fps
        if length < self.span:
            self.logger.warning("[SPEED] %s duration=%.3fs < %.3fs: unknown (no original-size window)",
                                context, length, self.span)
            return None
        factor = 1.0
        for iteration in range(self.recursion):
            duration = self.span * factor
            slices = original_window_slices(len(frames), fps, duration)
            if not slices:
                self.logger.warning("[SPEED] %s pass=%d: unknown (no valid window after cadence clamp)", context, iteration + 1)
                return None
            self.logger.info("[SPEED] %s pass=%d/%d window=%.6fs factor=%.6f clips=%d",
                             context, iteration + 1, self.recursion, duration, factor, len(slices))
            logs = []
            for clip_index, (left, right) in enumerate(slices, 1):
                # UniformTemporalSubsample uses linspace, clamp and long (floor).
                indices = torch.linspace(0, right - left - 1, self.count, device=frames.device).clamp(0, right - left - 1).long() + left
                clip = frames.index_select(0, indices).permute(1, 0, 2, 3).unsqueeze(0)
                log_speed = float(self.model(clip).item())
                if not math.isfinite(log_speed):
                    raise RuntimeError(f"Nonfinite speed prediction: {context}")
                logs.append(log_speed)
                self.logger.info("[SPEED] %s pass=%d clip=%d/%d local=[%.3f,%.3f)s log_score=%.6f",
                                 context, iteration + 1, clip_index, len(slices), left / fps, right / fps, log_speed)
                del clip
            predicted = math.exp(sum(logs) / len(logs)) / factor
            if not math.isfinite(predicted) or predicted <= 0:
                raise RuntimeError(f"Invalid speed score: {context}")
            max_factor = length / self.span - 1e-9
            min_factor = self.sample_fps / fps + 1e-9
            factor = float(max(min_factor, min(max_factor, 1 / predicted)))
            self.logger.info("[SPEED] %s pass=%d corrected_score=%.6f next_factor=%.6f", context, iteration + 1, predicted, factor)
        return predicted

    def predict(self, video: Path, windows: list[dict], fps: float) -> list[str]:
        """Decode once, buffering at most one resized caption segment on GPU."""
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Source FPS must be finite and positive")
        if not windows:
            return []
        for i, w in enumerate(windows):
            if not 0 <= w["start"] < w["end"] or (i and w["start"] < windows[i-1]["end"]):
                raise ValueError("Speed segments must be nonoverlapping and sorted")
        self.logger.info("[SPEED] %s source_fps=%.8f segments=%d NVDEC start", video.name, fps, len(windows))
        labels, chunks = [], []
        current = 0
        next_decode_log = 0.0
        with torch.inference_mode(), CudaVideoReader(video, batch_size=8) as reader:
            bounds = [(math.ceil(w["start"] * fps), min(reader.total, math.ceil(w["end"] * fps))) for w in windows]
            for offset, batch in reader:
                batch_end = offset + len(batch)
                decoded_seconds = batch_end / fps
                if decoded_seconds >= next_decode_log:
                    self.logger.info("[SPEED-DECODE] %s frame=%d/%d time=%.2fs", video.name, batch_end, reader.total, decoded_seconds)
                    next_decode_log = decoded_seconds + 5
                while current < len(windows) and bounds[current][0] < batch_end:
                    first, last = bounds[current]
                    left, right = max(first, offset), min(last, batch_end)
                    if left < right:
                        rgb = batch[left-offset:right-offset].float().div_(255)
                        rgb = (rgb - self.mean) / self.std
                        chunks.append(resize(rgb, [self.size, self.size], antialias=True))
                        del rgb
                    if last > batch_end:
                        break
                    shot = windows[current]
                    context = f"{video.name} shot={current+1}/{len(windows)} [{shot['start']:.3f},{shot['end']:.3f})"
                    if not chunks:
                        raise RuntimeError(f"No decoded frames for {context}")
                    frames = torch.cat(chunks)
                    chunks.clear()
                    if len(frames) != last - first:
                        raise RuntimeError(f"Incomplete GPU frames for {context}")
                    self.logger.info("[SPEED] %s START source_frames=%d duration=%.3fs", context, len(frames), len(frames)/fps)
                    score = self._estimate(frames, fps, context)
                    label = speed_label(score, self.cfg)
                    labels.append(label)
                    self.logger.info("[SPEED] %s DONE final_score=%s label=%s", context,
                                     "unknown" if score is None else f"{score:.6f}", label)
                    del frames
                    current += 1
        if current != len(windows):
            raise RuntimeError(f"Incomplete GPU speed predictions: {video}")
        return labels

    def close(self):
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()
