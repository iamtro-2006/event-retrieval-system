from __future__ import annotations

import argparse
from collections import deque
import json
import math
from pathlib import Path
import shutil
import time

import av
import cv2
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Online Gaussian scene detection using HSV and motion dissimilarity."
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=REPO_ROOT / "data/raw/videos/N021/N021-V001.mov",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data/raw/baselines/statistical_hsv_motion",
    )
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--analysis-width", type=int, default=320)
    parser.add_argument("--analysis-height", type=int, default=180)
    parser.add_argument("--reference-seconds", type=float, default=10.0)
    parser.add_argument("--current-seconds", type=float, default=1.0)
    parser.add_argument("--p-value", type=float, default=0.05)
    parser.add_argument("--consecutive", type=int, default=3)
    parser.add_argument("--min-scene-seconds", type=float, default=5.0)
    parser.add_argument("--max-scene-seconds", type=float, default=30.0)
    parser.add_argument("--hsv-weight", type=float, default=0.6)
    parser.add_argument("--motion-weight", type=float, default=0.4)
    parser.add_argument("--min-score-std", type=float, default=0.002)
    parser.add_argument("--image-quality", type=int, default=90)
    parser.add_argument("--output-width", type=int, default=640)
    parser.add_argument("--output-height", type=int, default=360)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _hsv_histogram(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    parts = [
        cv2.calcHist([hsv], [0], None, [16], [0, 180]).reshape(-1),
        cv2.calcHist([hsv], [1], None, [8], [0, 256]).reshape(-1),
        cv2.calcHist([hsv], [2], None, [8], [0, 256]).reshape(-1),
    ]
    histogram = np.concatenate(parts).astype(np.float64)
    histogram /= max(float(histogram.sum()), 1e-12)
    return histogram


def _bhattacharyya(left: np.ndarray, right: np.ndarray) -> float:
    coefficient = float(np.sqrt(np.clip(left, 0, None) * np.clip(right, 0, None)).sum())
    return math.sqrt(max(0.0, 1.0 - min(coefficient, 1.0)))


def _upper_gaussian_pvalue(value: float, mean: float, std: float) -> float:
    z_score = (value - mean) / std
    return 0.5 * math.erfc(z_score / math.sqrt(2.0))


def _foreground_centroid(mask: np.ndarray) -> tuple[float, float]:
    moments = cv2.moments(mask, binaryImage=True)
    if moments["m00"] <= 0:
        return 0.5, 0.5
    return (
        float(moments["m10"] / moments["m00"] / mask.shape[1]),
        float(moments["m01"] / moments["m00"] / mask.shape[0]),
    )


def _reference_distances(
    observations: list[tuple[np.ndarray, np.ndarray]],
    hsv_weight: float,
    motion_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    histograms = np.stack([item[0] for item in observations])
    motions = np.stack([item[1] for item in observations])
    histogram_mean = histograms.mean(axis=0)
    histogram_mean /= max(float(histogram_mean.sum()), 1e-12)
    motion_mean = motions.mean(axis=0)
    motion_std = np.maximum(motions.std(axis=0, ddof=1), 0.01)
    distances = np.asarray(
        [
            hsv_weight * _bhattacharyya(histogram, histogram_mean)
            + motion_weight * float(np.mean(np.abs((motion - motion_mean) / motion_std)))
            for histogram, motion in zip(histograms, motions)
        ],
        dtype=np.float64,
    )
    return histogram_mean, motion_mean, motion_std, distances


def _current_dissimilarity(
    observations: list[tuple[np.ndarray, np.ndarray]],
    histogram_mean: np.ndarray,
    motion_mean: np.ndarray,
    motion_std: np.ndarray,
    hsv_weight: float,
    motion_weight: float,
) -> float:
    histogram = np.stack([item[0] for item in observations]).mean(axis=0)
    histogram /= max(float(histogram.sum()), 1e-12)
    motion = np.stack([item[1] for item in observations]).mean(axis=0)
    return (
        hsv_weight * _bhattacharyya(histogram, histogram_mean)
        + motion_weight * float(np.mean(np.abs((motion - motion_mean) / motion_std)))
    )


def _scene_rows(boundaries: list[tuple[int, str, float, float]], frame_count: int, fps: float) -> list[dict[str, object]]:
    starts = [0] + [item[0] for item in boundaries if 0 < item[0] < frame_count]
    starts = sorted(set(starts))
    boundary_meta = {item[0]: item[1:] for item in boundaries}
    rows: list[dict[str, object]] = []
    for shot_id, start in enumerate(starts):
        end = starts[shot_id + 1] - 1 if shot_id + 1 < len(starts) else frame_count - 1
        reason, p_value, score = boundary_meta.get(start, ("video_start", 1.0, 0.0))
        rows.append(
            {
                "shot_id": shot_id,
                "start_frame": start,
                "end_frame": end,
                "start_sec": start / fps,
                "end_sec": end / fps,
                "duration_sec": (end - start + 1) / fps,
                "boundary_reason": reason,
                "boundary_p_value": p_value,
                "boundary_score": score,
            }
        )
    return rows


def _keyframe_rows(scene_rows: list[dict[str, object]], video_id: str, fps: float) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for scene in scene_rows:
        start, end = int(scene["start_frame"]), int(scene["end_frame"])
        span = end - start
        seen: set[int] = set()
        for position, frame_idx in (
            ("early", start + round(span / 6)),
            ("middle", start + round(span / 2)),
            ("late", start + round(5 * span / 6)),
        ):
            if frame_idx in seen:
                continue
            seen.add(frame_idx)
            candidates.append(
                {
                    "video_id": video_id,
                    "frame_idx": frame_idx,
                    "timestamp_sec": frame_idx / fps,
                    "fps": fps,
                    "method": "statistical_hsv_motion",
                    "shot_id": scene["shot_id"],
                    "shot_start_frame": start,
                    "shot_end_frame": end,
                    "position": position,
                }
            )

    # Preserve exact video endpoints for temporal navigation while keeping the
    # per-scene representatives away from adjacent scene boundaries.
    if scene_rows:
        first_scene = scene_rows[0]
        last_scene = scene_rows[-1]
        candidates.extend(
            [
                {
                    "video_id": video_id,
                    "frame_idx": int(first_scene["start_frame"]),
                    "timestamp_sec": int(first_scene["start_frame"]) / fps,
                    "fps": fps,
                    "method": "statistical_hsv_motion",
                    "shot_id": first_scene["shot_id"],
                    "shot_start_frame": int(first_scene["start_frame"]),
                    "shot_end_frame": int(first_scene["end_frame"]),
                    "position": "video_start",
                },
                {
                    "video_id": video_id,
                    "frame_idx": int(last_scene["end_frame"]),
                    "timestamp_sec": int(last_scene["end_frame"]) / fps,
                    "fps": fps,
                    "method": "statistical_hsv_motion",
                    "shot_id": last_scene["shot_id"],
                    "shot_start_frame": int(last_scene["start_frame"]),
                    "shot_end_frame": int(last_scene["end_frame"]),
                    "position": "video_end",
                },
            ]
        )

    unique = {int(record["frame_idx"]): record for record in candidates}
    records = [unique[index] for index in sorted(unique)]
    for keyframe_id, record in enumerate(records):
        record["keyframe_id"] = keyframe_id
    return records


def main() -> None:
    args = parse_args()
    video = args.video.resolve()
    output_root = args.output_dir.resolve()
    image_dir = output_root / "keyframes" / video.parent.name / video.stem
    map_path = output_root / "map_keyframes" / video.parent.name / f"{video.stem}.csv"
    scene_path = output_root / "scenes" / video.parent.name / f"{video.stem}.csv"
    trace_path = output_root / "traces" / video.parent.name / f"{video.stem}.csv"
    if image_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists; use --overwrite: {image_dir}")
        shutil.rmtree(image_dir)

    started = time.perf_counter()
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or stream.guessed_rate or 0.0)
        sample_stride = max(1, int(round(fps / args.sample_fps)))
        effective_sample_fps = fps / sample_stride
        reference_size = max(4, int(round(args.reference_seconds * effective_sample_fps)))
        current_size = max(1, int(round(args.current_seconds * effective_sample_fps)))
        min_scene_frames = max(1, int(round(args.min_scene_seconds * fps)))
        max_scene_frames = max(min_scene_frames, int(round(args.max_scene_seconds * fps)))
        subtractor = cv2.createBackgroundSubtractorMOG2(
            history=reference_size * 2, varThreshold=16, detectShadows=False
        )
        reference: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=reference_size)
        current: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=current_size)
        previous_gray: np.ndarray | None = None
        candidate_start: int | None = None
        anomaly_count = 0
        scene_start = 0
        boundaries: list[tuple[int, str, float, float]] = []
        traces: list[dict[str, object]] = []
        frame_count = 0

        stream.thread_type = "AUTO"
        for frame_count, frame in enumerate(container.decode(stream), start=1):
            frame_idx = frame_count - 1
            if frame_idx % sample_stride != 0:
                continue
            image = frame.to_ndarray(format="bgr24")
            image = cv2.resize(
                image,
                (args.analysis_width, args.analysis_height),
                interpolation=cv2.INTER_AREA,
            )
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            foreground = subtractor.apply(image)
            occupancy = float(np.count_nonzero(foreground)) / foreground.size
            centroid_x, centroid_y = _foreground_centroid(foreground)
            frame_difference = (
                0.0
                if previous_gray is None
                else float(cv2.absdiff(gray, previous_gray).mean() / 255.0)
            )
            previous_gray = gray
            observation = (
                _hsv_histogram(image),
                np.asarray([occupancy, centroid_x, centroid_y, frame_difference], dtype=np.float64),
            )
            current.append(observation)

            p_value, score, score_mean, score_std = 1.0, 0.0, 0.0, 0.0
            state = "warmup"
            if len(reference) < reference_size:
                reference.append(observation)
            elif len(current) == current_size:
                histogram_mean, motion_mean, motion_std, distances = _reference_distances(
                    list(reference), args.hsv_weight, args.motion_weight
                )
                score = _current_dissimilarity(
                    list(current),
                    histogram_mean,
                    motion_mean,
                    motion_std,
                    args.hsv_weight,
                    args.motion_weight,
                )
                score_mean = float(distances.mean())
                score_std = max(float(distances.std(ddof=1)), args.min_score_std)
                p_value = _upper_gaussian_pvalue(score, score_mean, score_std)
                state = "stable"
                if p_value < args.p_value:
                    state = "candidate"
                    if anomaly_count == 0:
                        candidate_start = frame_idx
                    anomaly_count += 1
                else:
                    anomaly_count = 0
                    candidate_start = None
                    reference.append(observation)

                if (
                    anomaly_count >= args.consecutive
                    and candidate_start is not None
                    and candidate_start - scene_start >= min_scene_frames
                ):
                    boundaries.append((candidate_start, "p_value", p_value, score))
                    scene_start = candidate_start
                    reference.clear()
                    reference.extend(current)
                    current.clear()
                    anomaly_count = 0
                    candidate_start = None
                    state = "confirmed"

            if frame_idx - scene_start >= max_scene_frames:
                boundary = frame_idx
                boundaries.append((boundary, "max_duration", p_value, score))
                scene_start = boundary
                reference.clear()
                reference.extend(current)
                current.clear()
                anomaly_count = 0
                candidate_start = None
                state = "forced"

            traces.append(
                {
                    "frame_idx": frame_idx,
                    "timestamp_sec": frame_idx / fps,
                    "score": score,
                    "score_mean": score_mean,
                    "score_std": score_std,
                    "p_value": p_value,
                    "state": state,
                    "occupancy": occupancy,
                    "frame_difference": frame_difference,
                }
            )

    scenes = _scene_rows(boundaries, frame_count, fps)
    keyframes = _keyframe_rows(scenes, video.stem, fps)
    scene_path.parent.mkdir(parents=True, exist_ok=True)
    map_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(scenes).to_csv(scene_path, index=False)
    pd.DataFrame(keyframes).to_csv(map_path, index=False)
    pd.DataFrame(traces).to_csv(trace_path, index=False)

    wanted = {int(row["frame_idx"]): int(row["keyframe_id"]) for row in keyframes}
    image_dir.mkdir(parents=True, exist_ok=True)
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame_idx, frame in enumerate(container.decode(stream)):
            if frame_idx not in wanted:
                continue
            image = frame.to_ndarray(format="bgr24")
            image = cv2.resize(
                image, (args.output_width, args.output_height), interpolation=cv2.INTER_AREA
            )
            path = image_dir / f"{wanted[frame_idx]:06d}.jpg"
            if not cv2.imwrite(
                str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), args.image_quality]
            ):
                raise OSError(f"Could not write {path}")

    reasons = pd.Series([scene["boundary_reason"] for scene in scenes]).value_counts().to_dict()
    result = {
        "video": str(video),
        "frames": frame_count,
        "fps": fps,
        "effective_sample_fps": effective_sample_fps,
        "scenes": len(scenes),
        "keyframes": len(keyframes),
        "boundary_reasons": reasons,
        "median_scene_seconds": float(pd.Series([row["duration_sec"] for row in scenes]).median()),
        "elapsed_sec": time.perf_counter() - started,
        "scene_csv": str(scene_path),
        "map_csv": str(map_path),
        "trace_csv": str(trace_path),
    }
    (output_root / f"{video.stem}.summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
