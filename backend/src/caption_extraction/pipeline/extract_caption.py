from __future__ import annotations

import gc
import json
import logging
import math
import re
import subprocess
from difflib import SequenceMatcher
from fractions import Fraction
from pathlib import Path

import numpy as np

from src.utils.frame_sampling import merge_consecutive_short_scenes

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm"}


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _probe_video(video: Path) -> tuple[float, float]:
    data = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,duration:format=duration",
        "-of", "json", str(video),
    ], text=True))
    stream = data["streams"][0]
    fps = float(Fraction(stream["avg_frame_rate"]))
    duration = float(data["format"].get("duration") or stream["duration"])
    if not math.isfinite(fps) or not math.isfinite(duration) or fps <= 0 or duration <= 0:
        raise ValueError(f"Invalid video FPS/duration: {video}")
    return fps, duration


def _normalized(text: str) -> str:
    return re.sub(r"[^\w]+", " ", text.casefold()).strip()


def _speed_prefix(caption: str) -> str:
    match = re.match(r"^The\s+(timelapse|slow-motion)\s+video\s+segment\b", caption, re.I)
    return match.group(1).lower() if match else "normal"


def apply_speed_prefix(caption: str, label: str) -> str:
    """Apply the classifier decision exactly once; repair a missing VLM prefix."""
    text = re.sub(r"^The\s+(?:(?:timelapse|slow-motion)\s+)?video\s+segment\b\s*",
                  "", caption.strip(), count=1, flags=re.I)
    prefix = f"The {label} video segment" if label in {"timelapse", "slow-motion"} else "The video segment"
    return f"{prefix} {text}".strip()


def merge_duplicate_captions(records: list[dict], max_gap: float, similarity: float) -> list[dict]:
    """Merge near-identical, consecutive captions; preserve distinct actions."""
    merged: list[dict] = []
    for record in sorted(records, key=lambda item: (item["start"], item["end"])):
        current = {key: record[key] for key in ("video_id", "start", "end", "caption")}
        if merged:
            previous = merged[-1]
            gap = current["start"] - previous["end"]
            score = SequenceMatcher(
                None, _normalized(previous["caption"]), _normalized(current["caption"])
            ).ratio()
            if (-0.001 <= gap <= max_gap and score >= similarity
                    and _speed_prefix(previous["caption"]) == _speed_prefix(current["caption"])):
                previous["end"] = max(previous["end"], current["end"])
                continue
        merged.append(current)
    return merged


class ExtractCaptionPipeline:
    """Write one general-shot TXT and one ASR-shaped caption JSON per video."""

    def __init__(self, cfg: dict, input_dir: Path, scenes_dir: Path,
                 output_dir: Path, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.input_dir, self.scenes_dir, self.output_dir = input_dir, scenes_dir, output_dir
        self.cfg = cfg["extraction"]
        self.shot_model = self.cfg["shot_model"]
        self.shot_revision = self.cfg["shot_revision"]
        self.caption_model_name = self.cfg["caption_model"]
        self.max_seconds = float(self.cfg["max_shot_seconds"])
        self.min_seconds = float(self.cfg["min_caption_seconds"])
        self.sample_fps = float(self.cfg["frame_sample_fps"])
        self.profiles = self.cfg["caption_profiles"]
        self.max_new_tokens = int(self.cfg["max_new_tokens"])
        self.duplicate_gap = float(self.cfg["duplicate_max_gap"])
        self.duplicate_similarity = float(self.cfg["duplicate_similarity"])
        self.prompt = self.cfg["prompt"].format(language=self.cfg["language"])
        self.speed_config = self.cfg["speed_predictor"]
        if not (0 < self.min_seconds <= self.max_seconds <= 10 and self.sample_fps > 0):
            raise ValueError("Caption duration/FPS config is invalid")
        if not self.profiles or any(p["max_frames"] < 1 or p["short_side"] < 1 for p in self.profiles):
            raise ValueError("Caption profiles must have positive frame counts and sizes")
        if not (self.duplicate_gap >= 0 and 0 < self.duplicate_similarity <= 1):
            raise ValueError("Duplicate merge config is invalid")
        self.model = None
        self.processor = None

    def video_files(self, video_ids: list[str] | None = None) -> list[Path]:
        if not self.input_dir.is_dir():
            raise FileNotFoundError(self.input_dir)
        files = sorted(p for p in self.input_dir.iterdir()
                       if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
        by_id = {p.stem: p for p in files}
        if len(by_id) != len(files):
            raise ValueError("Several video files have the same video ID")
        if video_ids:
            missing = set(video_ids) - by_id.keys()
            if missing:
                raise FileNotFoundError(f"Missing videos: {sorted(missing)}")
            return [by_id[vid] for vid in dict.fromkeys(video_ids)]
        return files

    def scene_path(self, video: Path) -> Path:
        return self.scenes_dir / f"{video.stem}.scenes.txt"

    def _detect_scenes(self, video: Path, detector) -> np.ndarray:
        # The public clean_shot mode returns the ranges we need directly.
        # Default-mode intra labels vary in representation across wrappers.
        from src.utils.omni_progress import inference_with_progress
        fps, duration = _probe_video(video)
        self.logger.info("[OMNI] %s START source_fps=%.8f duration=%.3fs", video.name, fps, duration)
        ranges = inference_with_progress(detector, video, mode="clean_shot", fps=fps, logger=self.logger)
        scenes = np.asarray([
            [int(start), int(end) - 1] for start, end in ranges
            if int(end) > int(start)
        ], dtype=np.int64).reshape(-1, 2)
        if len(scenes) == 0:
            raise RuntimeError(f"OmniShotCut clean_shot returned no shots: {video}")
        for index, (first_frame, last_frame) in enumerate(scenes, 1):
            self.logger.info("[OMNI-SHOT] %s shot=%d/%d frames=[%d,%d] time=[%.3f,%.3f)s",
                             video.name, index, len(scenes), first_frame, last_frame,
                             first_frame / fps, (last_frame + 1) / fps)
        timeline = np.arange(int(scenes[-1, 1]) + 1, dtype=np.float64) / fps
        # Reuse the keyframe short-shot merger only inside touching general runs.
        # Bridging a gap here would put an excluded transition back into a caption.
        runs: list[np.ndarray] = []
        first = 0
        for index in range(1, len(scenes) + 1):
            if index == len(scenes) or scenes[index, 0] != scenes[index - 1, 1] + 1:
                runs.append(merge_consecutive_short_scenes(
                    scenes[first:index], timeline, fps, threshold_seconds=self.min_seconds,
                ))
                first = index
        return np.vstack(runs)

    def prepare_shots(self, videos: list[Path], overwrite: bool = False) -> None:
        """Detect the whole batch before the caption model occupies GPU memory."""
        detector = None
        try:
            for video_index, video in enumerate(videos, 1):
                self.logger.info("[OMNI] Batch video=%d/%d %s", video_index, len(videos), video.name)
                scene_path = self.scene_path(video)
                if scene_path.is_file() and not overwrite:
                    self.logger.info("Reuse shots: %s", scene_path)
                    continue
                if detector is None:
                    import omnishotcut
                    from huggingface_hub import hf_hub_download
                    weights = hf_hub_download(
                        repo_id=self.shot_model, filename="OmniShotCut_ckpt.pth",
                        revision=self.shot_revision,
                    )
                    detector = omnishotcut.load(weights)
                scenes = self._detect_scenes(video, detector)
                scene_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = scene_path.with_suffix(scene_path.suffix + ".tmp")
                try:
                    np.savetxt(temporary, scenes, fmt="%d")
                    temporary.replace(scene_path)
                finally:
                    temporary.unlink(missing_ok=True)
                self.logger.info("Saved %d shots -> %s", len(scenes), scene_path)
        finally:
            del detector
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _windows(self, video: Path) -> list[dict]:
        scenes = np.loadtxt(self.scene_path(video), dtype=np.int64).reshape(-1, 2)
        fps, duration = _probe_video(video)
        windows = []
        for first, last in scenes:
            if first < 0 or last < first:
                raise ValueError(f"Invalid scene range in {self.scene_path(video)}")
            start = min(duration, first / fps)
            end = min(duration, (last + 1) / fps)
            if end - start < self.min_seconds:
                continue
            parts = math.ceil((end - start) / self.max_seconds)
            for part in range(parts):
                left = start + (end - start) * part / parts
                right = start + (end - start) * (part + 1) / parts
                windows.append({"start": round(left, 3), "end": round(right, 3)})
        if not windows:
            raise RuntimeError(f"No captionable shots: {video}")
        return windows

    def _load_caption_model(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
        if not torch.cuda.is_available():
            raise RuntimeError("VideoLLaMA3 requires CUDA for this pipeline")
        self.logger.info("[CAPTION] Loading %s on CUDA", self.caption_model_name)
        self.caption_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            self.caption_model_name, trust_remote_code=True,
            device_map={"": "cuda:0"}, torch_dtype=self.caption_dtype,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.caption_model_name, trust_remote_code=True)
        self.logger.info("[CAPTION] Model ready")

    def _sample_frames(self, reader, start: float, end: float, profile: dict):
        import cv2
        fps = reader.get_avg_fps()
        first = max(0, math.ceil(start * fps))
        last = min(len(reader) - 1, math.ceil(end * fps) - 1)
        if first > last:
            raise ValueError(f"No decoded video frame in [{start}, {end}]")
        count = min(int(profile["max_frames"]), max(1, math.ceil((end - start) * self.sample_fps)))
        indices = np.linspace(first, last, min(count, last - first + 1), dtype=int).tolist()
        decoded = reader.get_batch(indices).asnumpy()
        frames = []
        for frame in decoded:
            height, width = frame.shape[:2]
            scale = int(profile["short_side"]) / min(height, width)
            resized = cv2.resize(frame, (max(1, round(width * scale)),
                                         max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
            frames.append(np.ascontiguousarray(resized.transpose(2, 0, 1)))
        if not frames:
            raise ValueError(f"No sampled video frames in [{start}, {end}]")
        return frames, [index / fps for index in indices]

    def _caption(self, reader, start: float, end: float) -> str:
        import torch
        for attempt, profile in enumerate(self.profiles, 1):
            inputs = ids = frames = None
            try:
                frames, timestamps = self._sample_frames(reader, start, end, profile)
                self.logger.info("[CAPTION] [%.3f,%.3f)s profile=%d/%d frames=%d short_side=%d",
                                 start, end, attempt, len(self.profiles), len(frames), profile["short_side"])
                conversation = [
                    {"role": "system", "content": "You are a precise video captioner."},
                    {"role": "user", "content": [
                        {"type": "video", "video": frames,
                         "num_frames": len(frames), "timestamps": timestamps},
                        {"type": "text", "text": self.prompt},
                    ]},
                ]
                inputs = self.processor(
                    conversation=conversation, images=[("video", frames)],
                    add_system_prompt=True, add_generation_prompt=True, return_tensors="pt",
                )
                inputs = {key: value.to("cuda:0") if isinstance(value, torch.Tensor) else value
                          for key, value in inputs.items()}
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(self.caption_dtype)
                with torch.inference_mode():
                    ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                              do_sample=False)
                text = self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
                if not text:
                    raise RuntimeError(f"Empty caption at [{start}, {end}]")
                return text
            except torch.cuda.OutOfMemoryError:
                self.logger.warning("CUDA OOM at [%.2f, %.2f], profile %s", start, end, profile)
                if attempt == len(self.profiles):
                    raise
            finally:
                del inputs, ids, frames
                gc.collect()
                torch.cuda.empty_cache()
        raise RuntimeError("No caption profile succeeded")

    def caption_videos(self, videos: list[Path], overwrite: bool = False) -> None:
        from decord import VideoReader, cpu
        import torch
        from .speed import ShotSpeedPredictor

        # Finish labeling the whole pending batch before VideoLLaMA3 occupies VRAM.
        pending = [v for v in videos if overwrite or not (self.output_dir / f"{v.stem}.json").is_file()]
        labels_by_video = {}
        self.model = self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        predictor = None
        try:
            if pending:
                predictor = ShotSpeedPredictor(self.speed_config, self.logger)
            for video in pending:
                windows = self._windows(video)
                labels_by_video[video] = predictor.predict(video, windows, _probe_video(video)[0])
                self.logger.info("GPU speed labeling finished: %s (%d segments)", video.name, len(windows))
        finally:
            if predictor is not None:
                predictor.close()
            del predictor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for video in videos:
            output_path = self.output_dir / f"{video.stem}.json"
            if output_path.is_file() and not overwrite:
                self.logger.info("Skip existing caption JSON: %s", output_path)
                continue
            windows = self._windows(video)
            speed_labels = labels_by_video[video]
            partial_path = output_path.with_suffix(".partial.json")
            signature = {"video_size": video.stat().st_size,
                         "scene_ranges": self.scene_path(video).read_text(encoding="utf-8"),
                         "caption_model": self.caption_model_name, "prompt": self.prompt,
                         "profiles": self.profiles, "sample_fps": self.sample_fps,
                         "max_new_tokens": self.max_new_tokens, "windows": windows,
                         "speed_predictor": self.speed_config, "speed_prefix_version": 2}
            if partial_path.is_file() and not overwrite:
                state = json.loads(partial_path.read_text(encoding="utf-8"))
                if state.get("signature") != signature:
                    raise RuntimeError(f"Stale caption checkpoint: {partial_path}")
            else:
                state = {"signature": signature, "records": []}
            done = len(state["records"])
            if done > len(windows) or any(
                not isinstance(item, dict) or item.get("video_id") != video.stem
                or item.get("start") != window["start"] or item.get("end") != window["end"]
                or not isinstance(item.get("caption"), str) or not item["caption"].strip()
                for item, window in zip(state["records"], windows)
            ):
                raise RuntimeError(f"Invalid caption checkpoint: {partial_path}")
            self._load_caption_model()
            self.logger.info("[CAPTION] %s START segments=%d resumed=%d remaining=%d",
                             video.name, len(windows), done, len(windows) - done)
            reader = VideoReader(str(video), ctx=cpu(0), num_threads=2)
            try:
                for index, window in enumerate(windows[done:], done + 1):
                    self.logger.info("[CAPTION] %s shot=%d/%d [%.3f,%.3f)s START label=%s",
                                     video.name, index, len(windows), window["start"], window["end"], speed_labels[index-1])
                    caption = self._caption(reader, window["start"], window["end"])
                    caption = apply_speed_prefix(caption, speed_labels[index - 1])
                    state["records"].append({"video_id": video.stem, **window, "caption": caption})
                    _save_json(partial_path, state)
                    self.logger.info("[CAPTION-FINAL] %s shot %d/%d [%.2f-%.2f]: %s", video.name,
                                     index, len(windows), window["start"], window["end"], caption)
            finally:
                del reader
                gc.collect()
            records = merge_duplicate_captions(
                state["records"], self.duplicate_gap, self.duplicate_similarity,
            )
            for index, record in enumerate(records, 1):
                self.logger.info("[JSON-CAPTION] %s segment=%d/%d [%.3f,%.3f)s %s", video.name,
                                 index, len(records), record["start"], record["end"], record["caption"])
            _save_json(output_path, records)
            partial_path.unlink(missing_ok=True)
            self.logger.info("Saved %d caption segments -> %s", len(records), output_path)
