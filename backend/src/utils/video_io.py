from __future__ import annotations

from pathlib import Path
import json
import logging
import subprocess

import cv2
import ffmpeg
import numpy as np

VIDEO_EXTENSIONS = (".mp4", ".mkv", ".avi", ".mov", ".webm")


def list_videos(input_dir: Path) -> list[Path]:
    return sorted(p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)


def decode_for_transnet(video_path: Path, width: int = 48, height: int = 27) -> np.ndarray:
    stream, _ = (
        ffmpeg.input(str(video_path))
        .output("pipe:", format="rawvideo", pix_fmt="rgb24", s=f"{width}x{height}")
        .run(capture_stdout=True, capture_stderr=True)
    )
    arr = np.frombuffer(stream, np.uint8)
    if arr.size == 0:
        raise RuntimeError(f"FFmpeg decoded no frames from {video_path}")
    return arr.reshape([-1, height, width, 3])


def get_video_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return fps if fps > 0 else 0.0


def _video_codec(video_path: Path) -> str:
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "json",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    streams = json.loads(probe.stdout).get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found: {video_path}")
    return streams[0].get("codec_name", "")


def ensure_h264(video_path: Path, logger: logging.Logger | None = None) -> Path:
    codec = _video_codec(video_path)
    if codec != "av1":
        return video_path

    tmp = video_path.with_name(video_path.stem + "_h264" + video_path.suffix)
    if logger:
        logger.info("Converting AV1 to H.264: %s", video_path)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-c:a",
            "copy",
            str(tmp),
        ],
        check=True,
    )
    tmp.replace(video_path)
    return video_path


def read_frame(video_path: Path, frame_idx: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()

    if ok:
        print(
            f"[READ_FRAME] idx={frame_idx} "
            f"shape={frame.shape} dtype={frame.dtype} "
            f"min={frame.min()} max={frame.max()}",
            flush=True,
        )
    else:
        print(f"[READ_FRAME_FAIL] idx={frame_idx}", flush=True)

    return frame if ok else None
