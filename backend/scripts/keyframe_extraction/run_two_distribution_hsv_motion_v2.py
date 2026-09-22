from __future__ import annotations

import argparse, itertools, json, shutil, sys, time
from collections import deque
from pathlib import Path

import av, cv2, numpy as np, pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from scripts.keyframe_extraction.run_statistical_hsv_motion import (
    REPO_ROOT, _bhattacharyya, _foreground_centroid, _hsv_histogram,
    _keyframe_rows, _scene_rows,
)


def parse_args():
    p = argparse.ArgumentParser(description="Theory-aligned two-distribution HSV+motion detector")
    p.add_argument("--video", type=Path, default=REPO_ROOT / "data/raw/videos/N021/N021-V002.mov")
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data")
    p.add_argument("--sample-fps", type=float, default=5.0)
    p.add_argument("--baseline-seconds", type=float, default=8.0)
    p.add_argument("--reference-test-seconds", type=float, default=2.0)
    p.add_argument("--current-seconds", type=float, default=2.0)
    p.add_argument("--window-step", type=float, default=2.0)
    p.add_argument("--p-value", type=float, default=0.05)
    p.add_argument("--consecutive-windows", type=int, default=2)
    p.add_argument("--local-consecutive", type=int, default=3)
    p.add_argument("--block-size", type=int, default=2)
    p.add_argument("--min-scene-seconds", type=float, default=5.0)
    p.add_argument("--max-scene-seconds", type=float, default=30.0)
    p.add_argument("--hsv-weight", type=float, default=0.6)
    p.add_argument("--motion-weight", type=float, default=0.4)
    p.add_argument("--difference-threshold", type=int, default=20)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def robust_center_scale(values, floor):
    center = np.median(values, axis=0)
    mad = np.median(np.abs(values - center), axis=0)
    return center, np.maximum(1.4826 * mad, floor)


def fit_scoring_model(baseline, hw, mw):
    histograms = np.stack([x[1] for x in baseline])
    motions = np.stack([x[2] for x in baseline])
    histogram_center = histograms.mean(axis=0)
    histogram_center /= histogram_center.sum()
    motion_center, motion_scale = robust_center_scale(motions, 0.01)
    hsv_dist = np.asarray([_bhattacharyya(h, histogram_center) for h in histograms])
    motion_dist = np.mean(np.abs((motions - motion_center) / motion_scale), axis=1)
    hsv_center, hsv_scale = robust_center_scale(hsv_dist, 0.002)
    motion_dist_center, motion_dist_scale = robust_center_scale(motion_dist, 0.05)
    return {
        "histogram_center": histogram_center, "motion_center": motion_center,
        "motion_scale": motion_scale, "hsv_center": float(hsv_center),
        "hsv_scale": float(hsv_scale), "motion_dist_center": float(motion_dist_center),
        "motion_dist_scale": float(motion_dist_scale), "hsv_weight": hw, "motion_weight": mw,
    }


def score_observations(observations, model):
    scores = []
    for _, histogram, motion in observations:
        dh = _bhattacharyya(histogram, model["histogram_center"])
        dm = float(np.mean(np.abs((motion - model["motion_center"]) / model["motion_scale"])))
        zh = (dh - model["hsv_center"]) / model["hsv_scale"]
        zm = (dm - model["motion_dist_center"]) / model["motion_dist_scale"]
        scores.append(model["hsv_weight"] * zh + model["motion_weight"] * zm)
    return np.asarray(scores, dtype=np.float64)


def block_permutation_pvalue(reference, current, block_size):
    if len(reference) % block_size or len(current) % block_size:
        raise ValueError("reference/current sample counts must be divisible by block-size")
    combined = np.concatenate([reference, current])
    blocks = combined.reshape(-1, block_size)
    current_blocks = len(current) // block_size
    observed = float(np.median(current) - np.median(reference))
    exceed, total = 0, 0
    all_ids = set(range(len(blocks)))
    for current_ids_tuple in itertools.combinations(range(len(blocks)), current_blocks):
        current_ids = set(current_ids_tuple)
        ref_ids = sorted(all_ids - current_ids)
        cur_ids = sorted(current_ids)
        statistic = float(np.median(blocks[cur_ids]) - np.median(blocks[ref_ids]))
        exceed += statistic >= observed - 1e-12
        total += 1
    # All block assignments are enumerated, so this is the exact permutation
    # tail probability (the +1 correction is only needed for Monte Carlo draws).
    return exceed / total, observed


def localize(current, current_scores, reference_scores, required):
    threshold = float(np.quantile(reference_scores, 0.95))
    run = 0
    for i, value in enumerate(current_scores):
        run = run + 1 if value > threshold else 0
        if run >= required:
            return current[i - required + 1][0]
    return current[0][0]


def main():
    args, started = parse_args(), time.perf_counter()
    video, root = args.video.resolve(), args.output_dir.resolve()
    image_dir = root / "keyframes" / video.parent.name / video.stem
    scene_path = root / "scenes" / video.parent.name / f"{video.stem}.csv"
    map_path = root / "map_keyframes" / video.parent.name / f"{video.stem}.csv"
    trace_path = root / "traces" / video.parent.name / f"{video.stem}.csv"
    if image_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists; use --overwrite: {image_dir}")
        shutil.rmtree(image_dir)

    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or stream.guessed_rate)
        stride = max(1, round(fps / args.sample_fps))
        sample_fps = fps / stride
        nb = round(args.baseline_seconds * sample_fps)
        if args.block_size < 1:
            raise ValueError("block-size must be at least 1")
        # Variable-FPS/container metadata can turn 2 seconds at 5 FPS into
        # 9 or 11 samples. The exact block permutation test requires both
        # distributions to contain complete blocks, so round down safely.
        nr = max(args.block_size, round(args.reference_test_seconds * sample_fps))
        nc = max(args.block_size, round(args.current_seconds * sample_fps))
        nr -= nr % args.block_size
        nc -= nc % args.block_size
        step = max(1, round(args.window_step * sample_fps))
        if nr % args.block_size or nc % args.block_size:
            raise ValueError("reference-test/current sample counts must be divisible by block-size")
        min_frames, max_frames = round(args.min_scene_seconds * fps), round(args.max_scene_seconds * fps)
        buffer = deque(maxlen=nb + nr + nc)
        previous_gray = None
        sampled_count = frame_count = 0
        scene_start = anomaly_count = 0
        candidate_boundary = None
        frozen_model = frozen_reference_scores = None
        boundaries, traces = [], []
        stream.thread_type = "AUTO"
        for frame_count, frame in enumerate(container.decode(stream), start=1):
            frame_idx = frame_count - 1
            if frame_idx % stride:
                continue
            sampled_count += 1
            image = cv2.resize(frame.to_ndarray(format="bgr24"), (320, 180), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if previous_gray is None:
                delta = np.zeros_like(gray)
            else:
                delta = cv2.absdiff(gray, previous_gray)
            previous_gray = gray
            mask = (delta >= args.difference_threshold).astype(np.uint8)
            occupancy = float(mask.mean())
            cx, cy = _foreground_centroid(mask)
            motion = np.asarray([occupancy, cx, cy, float(delta.mean() / 255.0)])
            buffer.append((frame_idx, _hsv_histogram(image), motion))
            p_value, effect, state = 1.0, 0.0, "warmup"
            should_test = len(buffer) == buffer.maxlen and (sampled_count - buffer.maxlen) % step == 0
            if should_test:
                data = list(buffer)
                current = data[-nc:]
                if frozen_model is None:
                    baseline = data[:nb]
                    reference_test = data[nb:nb + nr]
                    model = fit_scoring_model(baseline, args.hsv_weight, args.motion_weight)
                    reference_scores = score_observations(reference_test, model)
                else:
                    model, reference_scores = frozen_model, frozen_reference_scores
                current_scores = score_observations(current, model)
                p_value, effect = block_permutation_pvalue(reference_scores, current_scores, args.block_size)
                state = "stable"
                if p_value < args.p_value and effect > 0:
                    if anomaly_count == 0:
                        candidate_boundary = localize(current, current_scores, reference_scores, args.local_consecutive)
                        frozen_model, frozen_reference_scores = model, reference_scores.copy()
                    anomaly_count += 1
                    state = "candidate"
                else:
                    anomaly_count, candidate_boundary = 0, None
                    frozen_model = frozen_reference_scores = None
                if (anomaly_count >= args.consecutive_windows and candidate_boundary is not None
                        and candidate_boundary - scene_start >= min_frames):
                    boundaries.append((candidate_boundary, "block_permutation", p_value, effect))
                    scene_start = candidate_boundary
                    buffer.clear()
                    anomaly_count, candidate_boundary = 0, None
                    frozen_model = frozen_reference_scores = None
                    state = "confirmed"
            if frame_idx - scene_start >= max_frames:
                boundaries.append((frame_idx, "max_duration", p_value, effect))
                scene_start = frame_idx
                buffer.clear()
                anomaly_count, candidate_boundary = 0, None
                frozen_model = frozen_reference_scores = None
                state = "forced"
            traces.append({"frame_idx": frame_idx, "timestamp_sec": frame_idx / fps,
                           "p_value": p_value, "median_effect": effect, "state": state,
                           "buffer_size": len(buffer), "occupancy": occupancy,
                           "frame_difference": motion[-1]})

    scenes = _scene_rows(boundaries, frame_count, fps)
    keyframes = _keyframe_rows(scenes, video.stem, fps)
    for path in (scene_path, map_path, trace_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(scenes).to_csv(scene_path, index=False)
    pd.DataFrame(keyframes).to_csv(map_path, index=False)
    pd.DataFrame(traces).to_csv(trace_path, index=False)
    wanted = {int(x["frame_idx"]): int(x["keyframe_id"]) for x in keyframes}
    image_dir.mkdir(parents=True, exist_ok=True)
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for idx, frame in enumerate(container.decode(stream)):
            if idx in wanted:
                image = cv2.resize(frame.to_ndarray(format="bgr24"), (640, 360), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(image_dir / f"{wanted[idx]:06d}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    reasons = pd.Series([x["boundary_reason"] for x in scenes]).value_counts().to_dict()
    summary = {"video": str(video), "test": "exact block permutation of median shift",
               "effective_sample_fps": sample_fps, "baseline_samples": nb,
               "reference_test_samples": nr, "current_samples": nc, "block_size": args.block_size,
               "window_step_seconds": args.window_step, "consecutive_windows": args.consecutive_windows,
               "feature_scaling": "per-modality median/MAD", "motion": "non-adaptive frame difference",
               "scenes": len(scenes), "keyframes": len(keyframes), "boundary_reasons": reasons,
               "median_scene_seconds": float(pd.Series([x["duration_sec"] for x in scenes]).median()),
               "elapsed_sec": time.perf_counter() - started, "scene_csv": str(scene_path),
               "map_csv": str(map_path), "trace_csv": str(trace_path)}
    (root / f"{video.stem}.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
