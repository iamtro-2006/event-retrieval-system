from __future__ import annotations

import gc
import json
import logging
import math
import re
import subprocess
import sys
from contextlib import closing
from difflib import SequenceMatcher
from fractions import Fraction
from pathlib import Path
from time import perf_counter

import numpy as np

from src.utils.frame_sampling import merge_consecutive_short_scenes
from .internvl import (CAPTION_FRAME_FILTER_VERSION, InternVLCaptioner,
                       occupied_color_bins, prepared_batches, validate_settings)

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


def predict_speed_in_worker(config: dict, videos: list[dict], logger) -> dict:
    """Isolate native codec dependencies; stream diagnostics, transfer labels in RAM."""
    import os
    prefix = "__SPEED_LABELS__="
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONFAULTHANDLER"] = "1"
    backend = str(Path(__file__).resolve().parents[3])
    env["PYTHONPATH"] = backend + os.pathsep + env.get("PYTHONPATH", "")
    result = None
    with subprocess.Popen(
        [sys.executable, "-u", "-X", "faulthandler", "-m",
         "src.caption_extraction.pipeline.speed_worker"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1, env=env,
    ) as process:
        logger.info("[SPEED-WORKER] pid=%d START isolated NVDEC process", process.pid)
        try:
            process.stdin.write(json.dumps({"config": config, "videos": videos}))
            process.stdin.close()
            for line in process.stdout:
                if line.startswith(prefix):
                    result = json.loads(line[len(prefix):])
                else:
                    print(line, end="", flush=True)
            code = process.wait()
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
        logger.info("[SPEED-WORKER] pid=%d END exit_code=%d", process.pid, code)
    if code:
        hint = " Native SIGSEGV in the codec/CUDA stack." if code == -11 else ""
        raise RuntimeError(
            f"Isolated GPU speed worker exited with code {code}.{hint} "
            "See live worker output above; no CPU fallback was used."
        )
    if not isinstance(result, dict) or set(result) != {v["path"] for v in videos}:
        raise RuntimeError("Speed worker returned incomplete labels")
    for video in videos:
        labels = result[video["path"]]
        if (not isinstance(labels, list) or len(labels) != len(video["windows"])
                or any(label not in {"normal", "timelapse", "slow-motion", "unknown"} for label in labels)):
            raise RuntimeError(f"Invalid labels for {video['path']}")
    return result


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
        if "caption_profiles" in self.cfg or "internvl" in self.cfg:
            raise ValueError("Legacy caption config: rerun notebook cell 3; use caption_max_frames and vllm")
        self.max_frames = self.cfg.get("caption_max_frames", 16)
        self.batch_size = self.cfg.get("caption_batch_size", 8)
        self.min_color_bins = self.cfg.get("caption_min_color_bins", 5)
        if type(self.min_color_bins) is not int or not 1 <= self.min_color_bins <= 512:
            raise ValueError("caption_min_color_bins must be an integer in [1, 512]")
        self.vllm_config = validate_settings(self.caption_model_name, self.max_frames,
                                            self.batch_size, self.cfg.get("vllm", {}))
        self.max_seconds = float(self.cfg["max_shot_seconds"])
        self.min_seconds = float(self.cfg["min_caption_seconds"])
        self.sample_fps = float(self.cfg["frame_sample_fps"])
        self.max_new_tokens = int(self.cfg["max_new_tokens"])
        self.duplicate_gap = float(self.cfg["duplicate_max_gap"])
        self.duplicate_similarity = float(self.cfg["duplicate_similarity"])
        self.prompt = self.cfg["prompt"].format(language=self.cfg["language"])
        self.speed_config = self.cfg["speed_predictor"]
        if not (0 < self.min_seconds <= self.max_seconds <= 10
                and math.isfinite(self.sample_fps) and self.sample_fps > 0):
            raise ValueError("Caption duration/FPS config is invalid")
        if self.max_new_tokens < 1 or not self.prompt.strip():
            raise ValueError("Caption prompt and max_new_tokens must be nonempty/positive")
        if not (self.duplicate_gap >= 0 and 0 < self.duplicate_similarity <= 1):
            raise ValueError("Duplicate merge config is invalid")
        self.captioner = None

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
        previous_last = -1
        for scene_index, (first, last) in enumerate(scenes, 1):
            if first < 0 or last < first:
                raise ValueError(f"Invalid scene range in {self.scene_path(video)}")
            if first <= previous_last:
                raise ValueError(f"Scenes overlap or are unsorted in {self.scene_path(video)}: "
                                 f"shot={scene_index} first_frame={first} previous_last_frame={previous_last}")
            previous_last = last
            start = min(duration, first / fps)
            end = min(duration, (last + 1) / fps)
            if end - start < self.min_seconds:
                continue
            parts = math.ceil((end - start) / self.max_seconds)
            edges = np.linspace(start, end, parts + 1).tolist()
            for part in range(parts):
                left, right = edges[part], edges[part + 1]
                # Keep frame-boundary precision for sampling; log formatting rounds display only.
                windows.append({"start": left, "end": right})
        if not windows:
            raise RuntimeError(f"No captionable shots: {video}")
        return windows

    def _load_caption_model(self) -> None:
        if self.captioner is None:
            self.captioner = InternVLCaptioner(self.caption_model_name, self.max_frames,
                                              self.max_new_tokens, self.vllm_config, self.logger)

    def _sample_frames(self, reader, start: float, end: float):
        from PIL import Image
        fps = reader.get_avg_fps()
        first = max(0, math.ceil(start * fps - 1e-7))
        last = min(len(reader) - 1, math.ceil(end * fps - 1e-7) - 1)
        if first > last:
            raise ValueError(f"No decoded video frame in [{start}, {end}]")
        count = min(self.max_frames, max(1, math.ceil((end - start) * self.sample_fps)))
        indices = np.linspace(first, last, min(count, last - first + 1), dtype=int).tolist()
        decoded = reader.get_batch(indices).asnumpy()
        size = self.captioner.image_size
        # One tile/frame matches the official video default. Resize before queueing
        # so prefetch/IPC retain 448px images rather than native 4K arrays.
        frames, timestamps = [], []
        for index, frame in zip(indices, decoded):
            occupied = occupied_color_bins(frame)
            if occupied < self.min_color_bins:
                self.logger.info("[CAPTION-FILTER] [%.3f,%.3f)s frame=%d time=%.3fs bins=%d < %d DROP",
                                 start, end, index, index / fps, occupied, self.min_color_bins)
                continue
            frames.append(Image.fromarray(frame).convert("RGB").resize((size, size), Image.Resampling.BICUBIC))
            timestamps.append(index / fps)
        self.logger.info("[CAPTION-FILTER] [%.3f,%.3f)s sampled=%d kept=%d dropped=%d min_bins=%d",
                         start, end, len(indices), len(frames), len(indices) - len(frames), self.min_color_bins)
        return frames, timestamps

    def _prepare_batch(self, reader, video, total, done, offset, windows):
        requests = []
        for index, window in enumerate(windows, done + offset + 1):
            frames, timestamps = self._sample_frames(reader, window["start"], window["end"])
            if not frames:
                requests.append(None)
                self.logger.info("[CAPTION] %s shot=%d/%d [%.3f,%.3f)s SKIP no informative sampled frames",
                                 video.name, index, total, window["start"], window["end"])
                continue
            requests.append(self.captioner.request(frames, timestamps, self.prompt))
            self.logger.info("[CAPTION] %s shot=%d/%d [%.3f,%.3f)s READY frames=%d tile=%d",
                             video.name, index, total, window["start"], window["end"],
                             len(frames), self.captioner.image_size)
        return requests

    def caption_videos(self, videos: list[Path], overwrite: bool = False) -> None:
        import torch

        # Finish labeling the whole pending batch before the caption model occupies VRAM.
        pending = [v for v in videos if overwrite or not (self.output_dir / f"{v.stem}.json").is_file()]
        labels_by_video = {}
        self.captioner = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if pending:
            requests = [{"path": str(v), "windows": self._windows(v), "fps": _probe_video(v)[0]}
                        for v in pending]
            results = predict_speed_in_worker(self.speed_config, requests, self.logger)
            labels_by_video = {v: results[str(v)] for v in pending}

        # Delay Decord until the NVDEC worker has fully exited.
        from decord import VideoReader, cpu

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
                         "backend": "vllm-internvl-v1", "max_frames": self.max_frames,
                         "sample_fps": self.sample_fps,
                         "frame_filter": {"version": CAPTION_FRAME_FILTER_VERSION,
                                          "min_color_bins": self.min_color_bins},
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
                or not ((isinstance(item.get("caption"), str) and item["caption"].strip())
                        or (item.get("caption") is None and item.get("skip_reason") == "uniform_frames"))
                for item, window in zip(state["records"], windows)
            ):
                raise RuntimeError(f"Invalid caption checkpoint: {partial_path}")
            self._load_caption_model()
            self.logger.info("[CAPTION] %s START segments=%d resumed=%d remaining=%d",
                             video.name, len(windows), done, len(windows) - done)
            reader = VideoReader(str(video), ctx=cpu(0), num_threads=2)
            started = perf_counter()
            generated_tokens = 0
            inferred_shots = 0
            try:
                prepare = lambda offset, batch: self._prepare_batch(
                    reader, video, len(windows), done, offset, batch)
                with closing(prepared_batches(windows[done:], self.batch_size, prepare)) as batches:
                    for offset, batch, items, prepare_seconds in batches:
                        first = done + offset + 1
                        requests = [item for item in items if item is not None]
                        self.logger.info("[CAPTION-BATCH] %s shots=%d-%d/%d START submitted=%d skipped=%d concurrent=%d prepare=%.2fs",
                                         video.name, first, first + len(batch) - 1, len(windows), len(requests), len(items) - len(requests),
                                         self.vllm_config["max_num_seqs"], prepare_seconds)
                        inference_start = perf_counter()
                        try:
                            captions, tokens = self.captioner.generate(requests) if requests else ([], 0)
                        except RuntimeError as error:
                            raise RuntimeError("vLLM caption batch failed; see engine logs above. On CUDA OOM, "
                                               "reduce max_num_seqs/max_num_batched_tokens, or use enforce_eager=True "
                                               "and restart the CLI process. Completed batches remain resumable.") from error
                        elapsed = max(perf_counter() - inference_start, 1e-9)
                        generated_tokens += tokens
                        inferred_shots += len(requests)
                        caption_iter = iter(captions)
                        for index, window, item in zip(range(first, first + len(batch)), batch, items):
                            if item is None:
                                state["records"].append({"video_id": video.stem, **window, "caption": None,
                                                         "skip_reason": "uniform_frames"})
                                self.logger.info("[CAPTION-FINAL] %s shot %d/%d [%.2f-%.2f]: SKIP uniform sampled frames; no JSON segment",
                                                 video.name, index, len(windows), window["start"], window["end"])
                                continue
                            label = speed_labels[index - 1]
                            caption = apply_speed_prefix(next(caption_iter), label)
                            state["records"].append({"video_id": video.stem, **window, "caption": caption})
                            self.logger.info("[CAPTION-FINAL] %s shot %d/%d [%.2f-%.2f] label=%s: %s", video.name,
                                             index, len(windows), window["start"], window["end"], label, caption)
                        _save_json(partial_path, state)
                        self.logger.info("[CAPTION-BATCH] %s DONE inference=%.2fs shots/s=%.3f output_tokens/s=%.1f checkpoint=%d/%d",
                                         video.name, elapsed, len(requests) / elapsed, tokens / elapsed,
                                         len(state["records"]), len(windows))
                        del items, requests, captions
            finally:
                del reader
            elapsed = max(perf_counter() - started, 1e-9)
            self.logger.info("[CAPTION-PERF] %s caption_wall=%.2fs inferred=%d skipped=%d shots/s=%.3f output_tokens/s=%.1f (excludes model load/speed/OMNI)",
                             video.name, elapsed, inferred_shots, len(windows) - done - inferred_shots, inferred_shots / elapsed,
                             generated_tokens / elapsed)
            records = merge_duplicate_captions(
                [record for record in state["records"] if record["caption"] is not None],
                self.duplicate_gap, self.duplicate_similarity,
            )
            for index, record in enumerate(records, 1):
                self.logger.info("[JSON-CAPTION] %s segment=%d/%d [%.3f,%.3f)s %s", video.name,
                                 index, len(records), record["start"], record["end"], record["caption"])
            _save_json(output_path, records)
            partial_path.unlink(missing_ok=True)
            self.logger.info("Saved %d caption segments -> %s", len(records), output_path)
