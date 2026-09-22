from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import time

import av
import cv2
import numpy as np
import pandas as pd


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}


@dataclass(frozen=True)
class SelectedFrame:
    frame_idx: int
    timestamp_sec: float
    score: float | None = None


def list_input_videos(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def _video_metadata(video_path: Path) -> tuple[float, int]:
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or stream.guessed_rate or 0.0)
        frame_count = int(stream.frames or 0)
    return fps, frame_count


def _frame_timestamp(frame: av.VideoFrame, fps: float, fallback_idx: int) -> float:
    if frame.pts is not None and frame.time_base is not None:
        return float(frame.pts * frame.time_base)
    return fallback_idx / fps if fps > 0 else 0.0


def _timestamp_to_index(timestamp_sec: float, fps: float, fallback_idx: int) -> int:
    return max(0, int(round(timestamp_sec * fps))) if fps > 0 else fallback_idx


def _write_jpeg(
    frame: av.VideoFrame,
    output_path: Path,
    quality: int,
    output_width: int | None = None,
    output_height: int | None = None,
) -> None:
    image = frame.to_ndarray(format="bgr24")
    if output_width and output_height:
        image = cv2.resize(
            image, (output_width, output_height), interpolation=cv2.INTER_AREA
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(
        str(output_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ):
        raise OSError(f"Could not write keyframe image: {output_path}")


def _write_map(
    selected: list[SelectedFrame],
    video_path: Path,
    csv_path: Path,
    fps: float,
    method: str,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = {
        "keyframe_id": list(range(len(selected))),
        "video_id": [video_path.stem] * len(selected),
        "frame_idx": [item.frame_idx for item in selected],
        "timestamp_sec": [item.timestamp_sec for item in selected],
        "fps": [fps] * len(selected),
        "method": [method] * len(selected),
        "selection_score": [item.score for item in selected],
    }
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def extract_ffmpeg_iframes(
    video_path: Path,
    image_dir: Path,
    map_path: Path,
    image_quality: int = 95,
    output_width: int | None = None,
    output_height: int | None = None,
) -> dict[str, float | int | str]:
    """Extract codec keyframes using FFmpeg's NONKEY discard mode via PyAV."""
    started = time.perf_counter()
    fps, _ = _video_metadata(video_path)
    selected: list[SelectedFrame] = []
    first_timestamp: float | None = None

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.skip_frame = "NONKEY"
        stream.thread_type = "AUTO"
        for decoded_idx, frame in enumerate(container.decode(stream)):
            # Some FFmpeg builds accept NONKEY but still return decoded
            # non-key pictures. Keep the explicit flag check for correctness.
            if not frame.key_frame:
                continue
            raw_timestamp = _frame_timestamp(frame, fps, decoded_idx)
            if first_timestamp is None:
                first_timestamp = raw_timestamp
            timestamp = max(0.0, raw_timestamp - first_timestamp)
            frame_idx = _timestamp_to_index(timestamp, fps, decoded_idx)
            keyframe_id = len(selected)
            _write_jpeg(
                frame,
                image_dir / f"{keyframe_id:06d}.jpg",
                image_quality,
                output_width,
                output_height,
            )
            selected.append(SelectedFrame(frame_idx, timestamp))

    _write_map(selected, video_path, map_path, fps, "ffmpeg_iframe")
    return {
        "method": "ffmpeg_iframe",
        "video": str(video_path),
        "keyframes": len(selected),
        "elapsed_sec": time.perf_counter() - started,
    }


def extract_ffmpeg_scene_frames(
    video_path: Path,
    image_dir: Path,
    map_path: Path,
    *,
    scene_threshold: float = 0.05,
    image_quality: int = 90,
    output_width: int | None = None,
    output_height: int | None = None,
) -> dict[str, float | int | str]:
    """Run FFmpeg's native ``select=gt(scene,threshold)`` video filter.

    The first frame is included explicitly so a video with no detected scene
    change still has one representative keyframe.
    """
    started = time.perf_counter()
    fps, _ = _video_metadata(video_path)
    selected: list[SelectedFrame] = []
    first_input_timestamp: float | None = None

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        graph = av.filter.Graph()
        source = graph.add_buffer(template=stream)
        scene_filter = graph.add(
            "select", f"eq(n,0)+gt(scene,{float(scene_threshold):.8f})"
        )
        sink = graph.add("buffersink")
        source.link_to(scene_filter)
        scene_filter.link_to(sink)
        graph.configure()

        def drain() -> None:
            while True:
                try:
                    output_frame = graph.pull()
                except (BlockingIOError, av.error.EOFError):
                    return
                raw_timestamp = _frame_timestamp(output_frame, fps, len(selected))
                timestamp = max(0.0, raw_timestamp - (first_input_timestamp or 0.0))
                frame_idx = _timestamp_to_index(timestamp, fps, len(selected))
                _write_jpeg(
                    output_frame,
                    image_dir / f"{len(selected):06d}.jpg",
                    image_quality,
                    output_width,
                    output_height,
                )
                selected.append(SelectedFrame(frame_idx, timestamp))

        for frame in container.decode(stream):
            if first_input_timestamp is None:
                first_input_timestamp = _frame_timestamp(frame, fps, 0)
            graph.push(frame)
            drain()
        graph.push(None)
        drain()

    _write_map(selected, video_path, map_path, fps, "ffmpeg_scene")
    return {
        "method": "ffmpeg_scene",
        "video": str(video_path),
        "keyframes": len(selected),
        "scene_threshold": scene_threshold,
        "elapsed_sec": time.perf_counter() - started,
    }


def _beta_region_features(
    gray: np.ndarray,
    grid_rows: int,
    grid_cols: int,
    curve_points: int = 16,
    epsilon: float = 1e-4,
) -> np.ndarray:
    """Represent each image region by a normalized method-of-moments Beta PDF.

    KFEAVI describes one Beta distribution per non-overlapping region but does
    not specify how the continuous distributions are reduced to scalar values.
    Sampling and normalizing each PDF gives a deterministic, numerically stable
    representation whose inter-frame L2 distance implements its Eq. 7 idea.
    """
    image = gray.astype(np.float32) / 255.0
    height, width = image.shape
    ys = np.linspace(0, height, grid_rows + 1, dtype=np.int32)
    xs = np.linspace(0, width, grid_cols + 1, dtype=np.int32)
    samples = np.linspace(0.02, 0.98, curve_points, dtype=np.float64)
    features: list[np.ndarray] = []

    for row in range(grid_rows):
        for col in range(grid_cols):
            region = image[ys[row] : ys[row + 1], xs[col] : xs[col + 1]]
            mean = float(np.clip(region.mean(), epsilon, 1.0 - epsilon))
            variance = float(max(region.var(), epsilon))
            max_variance = max(mean * (1.0 - mean) - epsilon, epsilon)
            variance = min(variance, max_variance)
            common = max(mean * (1.0 - mean) / variance - 1.0, epsilon)
            alpha = max(mean * common, epsilon)
            beta = max((1.0 - mean) * common, epsilon)

            log_norm = math.lgamma(alpha + beta) - math.lgamma(alpha) - math.lgamma(beta)
            log_pdf = (
                log_norm
                + (alpha - 1.0) * np.log(samples)
                + (beta - 1.0) * np.log1p(-samples)
            )
            pdf = np.exp(log_pdf - np.max(log_pdf))
            pdf /= max(float(np.linalg.norm(pdf)), epsilon)
            features.append(pdf.astype(np.float32))

    return np.stack(features)


def _kfeavi_scores(
    video_path: Path,
    analysis_width: int,
    analysis_height: int,
    grid_rows: int,
    grid_cols: int,
    sample_stride: int,
) -> tuple[float, int, list[SelectedFrame]]:
    fps, frame_count = _video_metadata(video_path)
    scored: list[SelectedFrame] = []
    previous: np.ndarray | None = None

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        decoded_idx = -1
        for decoded_idx, frame in enumerate(container.decode(stream)):
            if decoded_idx % sample_stride != 0:
                continue
            timestamp = _frame_timestamp(frame, fps, decoded_idx)
            # This pass decodes every picture in presentation order, so the
            # enumeration is the exact index used again during image export.
            frame_idx = decoded_idx
            gray = frame.to_ndarray(format="gray")
            gray = cv2.resize(
                gray,
                (analysis_width, analysis_height),
                interpolation=cv2.INTER_AREA,
            )
            current = _beta_region_features(gray, grid_rows, grid_cols)
            if previous is not None:
                region_distances = np.linalg.norm(current - previous, axis=1)
                scored.append(
                    SelectedFrame(frame_idx, timestamp, float(region_distances.sum()))
                )
            previous = current

    if frame_count <= 0:
        frame_count = decoded_idx + 1
    return fps, frame_count, scored


def _robust_threshold(scores: np.ndarray, mad_scale: float, percentile: float) -> float:
    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median)))
    robust = median + mad_scale * 1.4826 * mad
    percentile_threshold = float(np.percentile(scores, percentile))
    threshold = max(robust, percentile_threshold)
    # A static or quantized sequence can have MAD=0 and a percentile equal to
    # the median. Move by one floating-point step so ordinary frames are not
    # all selected merely because comparison uses >=.
    return float(np.nextafter(median, np.inf)) if threshold <= median else threshold


def _select_with_temporal_nms(
    scored: list[SelectedFrame],
    threshold: float,
    min_gap_frames: int,
) -> list[SelectedFrame]:
    candidates = [item for item in scored if item.score is not None and item.score >= threshold]
    candidates.sort(key=lambda item: (-float(item.score), item.frame_idx))
    kept: list[SelectedFrame] = []
    for candidate in candidates:
        if all(abs(candidate.frame_idx - item.frame_idx) >= min_gap_frames for item in kept):
            kept.append(candidate)
    return sorted(kept, key=lambda item: item.frame_idx)


def _extract_selected_frames(
    video_path: Path,
    selected: list[SelectedFrame],
    image_dir: Path,
    image_quality: int,
    output_width: int | None,
    output_height: int | None,
) -> list[SelectedFrame]:
    wanted = {item.frame_idx: item for item in selected}
    written: list[SelectedFrame] = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for decoded_idx, frame in enumerate(container.decode(stream)):
            if decoded_idx not in wanted:
                continue
            item = wanted[decoded_idx]
            _write_jpeg(
                frame,
                image_dir / f"{len(written):06d}.jpg",
                image_quality,
                output_width,
                output_height,
            )
            written.append(item)
            if len(written) == len(wanted):
                break
    return written


def extract_kfeavi(
    video_path: Path,
    image_dir: Path,
    map_path: Path,
    *,
    image_quality: int = 95,
    output_width: int | None = None,
    output_height: int | None = None,
    analysis_width: int = 320,
    analysis_height: int = 180,
    grid_rows: int = 4,
    grid_cols: int = 4,
    sample_stride: int = 1,
    threshold: float | None = None,
    mad_scale: float = 3.5,
    percentile: float = 98.0,
    min_gap_seconds: float = 1.0,
    include_endpoints: bool = True,
) -> dict[str, float | int | str]:
    """Extract KFEAVI-style statistical Beta-distribution keyframes."""
    started = time.perf_counter()
    fps, frame_count, scored = _kfeavi_scores(
        video_path,
        analysis_width,
        analysis_height,
        grid_rows,
        grid_cols,
        sample_stride,
    )
    if not scored:
        raise RuntimeError(f"No frames decoded from {video_path}")

    score_values = np.asarray([float(item.score) for item in scored], dtype=np.float64)
    used_threshold = (
        float(threshold)
        if threshold is not None
        else _robust_threshold(score_values, mad_scale, percentile)
    )
    min_gap_frames = max(1, int(round(min_gap_seconds * fps)))
    selected = _select_with_temporal_nms(scored, used_threshold, min_gap_frames)

    if include_endpoints:
        endpoint_indexes = {0, max(0, frame_count - 1)}
        present = {item.frame_idx for item in selected}
        for frame_idx in sorted(endpoint_indexes - present):
            selected.append(
                SelectedFrame(
                    frame_idx,
                    frame_idx / fps if fps > 0 else 0.0,
                    None,
                )
            )
        selected.sort(key=lambda item: item.frame_idx)

    written = _extract_selected_frames(
        video_path,
        selected,
        image_dir,
        image_quality,
        output_width,
        output_height,
    )
    _write_map(written, video_path, map_path, fps, "kfeavi_beta")
    return {
        "method": "kfeavi_beta",
        "video": str(video_path),
        "keyframes": len(written),
        "threshold": used_threshold,
        "elapsed_sec": time.perf_counter() - started,
    }


def _hsv_marginal_histogram(image_bgr: np.ndarray) -> np.ndarray:
    """Return the paper's compact 8H + 4S + 4V histogram representation."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    parts = [
        cv2.calcHist([hsv], [0], None, [8], [0, 180]).reshape(-1),
        cv2.calcHist([hsv], [1], None, [4], [0, 256]).reshape(-1),
        cv2.calcHist([hsv], [2], None, [4], [0, 256]).reshape(-1),
    ]
    histogram = np.concatenate(parts).astype(np.float32)
    histogram /= max(float(histogram.sum()), 1e-8)
    return histogram


def _histogram_dissimilarity(left: np.ndarray, right: np.ndarray) -> float:
    return float(1.0 - np.minimum(left, right).sum())


def _motion_trend_boundaries(
    dissimilarities: np.ndarray,
    window: int,
    min_gap: int,
    trend_percentile: float,
) -> list[int]:
    """Find sign changes in past-vs-future motion trend (paper Section II-C)."""
    count = len(dissimilarities)
    if count < 3:
        return []
    window = max(1, min(window, (count - 1) // 2))
    kernel = np.ones(window, dtype=np.float64) / window
    smooth = np.convolve(dissimilarities, kernel, mode="same")
    trend = np.zeros(count, dtype=np.float64)
    for idx in range(window, count - window):
        trend[idx] = smooth[idx + 1 : idx + window + 1].mean() - smooth[
            idx - window : idx
        ].mean()

    strength = np.abs(trend)
    threshold = float(np.percentile(strength[window : count - window], trend_percentile))
    raw = [
        idx
        for idx in range(window + 1, count - window)
        if np.signbit(trend[idx - 1]) != np.signbit(trend[idx])
        and max(strength[idx - 1], strength[idx]) >= threshold
    ]

    kept: list[int] = []
    for idx in sorted(raw, key=lambda item: -strength[item]):
        if all(abs(idx - existing) >= min_gap for existing in kept):
            kept.append(idx)
    return sorted(kept)


def _global_kmeans_medoids(features: np.ndarray, clusters: int) -> list[int]:
    """Farthest-first global initialization followed by Lloyd medoid updates."""
    count = len(features)
    if count == 0:
        return []
    clusters = max(1, min(int(clusters), count))
    global_mean = features.mean(axis=0)
    centers = [int(np.argmin(np.linalg.norm(features - global_mean, axis=1)))]

    while len(centers) < clusters:
        distances = np.linalg.norm(
            features[:, None, :] - features[np.asarray(centers)][None, :, :], axis=2
        )
        nearest = distances.min(axis=1)
        nearest[np.asarray(centers)] = -1.0
        centers.append(int(np.argmax(nearest)))

    centers_array = np.asarray(centers, dtype=np.int32)
    for _ in range(30):
        distances = np.linalg.norm(
            features[:, None, :] - features[centers_array][None, :, :], axis=2
        )
        labels = distances.argmin(axis=1)
        updated: list[int] = []
        for cluster_id in range(clusters):
            members = np.flatnonzero(labels == cluster_id)
            if len(members) == 0:
                updated.append(int(centers_array[cluster_id]))
                continue
            mean = features[members].mean(axis=0)
            updated.append(int(members[np.argmin(np.linalg.norm(features[members] - mean, axis=1))]))
        new_centers = np.asarray(updated, dtype=np.int32)
        if np.array_equal(new_centers, centers_array):
            break
        centers_array = new_centers
    return sorted(set(centers_array.tolist()))


def extract_motion_kmeans_2010(
    video_path: Path,
    image_dir: Path,
    map_path: Path,
    *,
    image_quality: int = 90,
    output_width: int | None = 640,
    output_height: int | None = 360,
    analysis_width: int = 320,
    analysis_height: int = 180,
    sample_stride: int = 5,
    background_frames: int = 50,
    trend_window_seconds: float = 1.0,
    min_subshot_seconds: float = 2.0,
    trend_percentile: float = 50.0,
) -> dict[str, float | int | str]:
    """Traffic-2010 motion-trend segmentation plus global K-means medoids.

    This implements the parts of the paper that do not require camera-specific
    virtual-line calibration. The whole fixed-camera video is treated as one
    shot; motion-trend sign changes estimate its sub-shots, and their count is
    used as K, matching the paper's cluster-count rule.
    """
    started = time.perf_counter()
    fps, _ = _video_metadata(video_path)
    frame_indexes: list[int] = []
    timestamps: list[float] = []
    histograms: list[np.ndarray] = []
    background_images: list[np.ndarray] = []
    first_timestamp: float | None = None

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for decoded_idx, frame in enumerate(container.decode(stream)):
            if decoded_idx % sample_stride != 0:
                continue
            raw_timestamp = _frame_timestamp(frame, fps, decoded_idx)
            if first_timestamp is None:
                first_timestamp = raw_timestamp
            image = frame.to_ndarray(format="bgr24")
            image = cv2.resize(
                image, (analysis_width, analysis_height), interpolation=cv2.INTER_AREA
            )
            if len(background_images) < background_frames:
                background_images.append(image.astype(np.float32))
            frame_indexes.append(decoded_idx)
            timestamps.append(max(0.0, raw_timestamp - first_timestamp))
            histograms.append(_hsv_marginal_histogram(image))

    if not histograms:
        raise RuntimeError(f"No frames decoded from {video_path}")
    background = np.mean(background_images, axis=0).astype(np.uint8)
    background_hist = _hsv_marginal_histogram(background)
    features = np.stack(histograms)
    dissimilarities = np.asarray(
        [_histogram_dissimilarity(hist, background_hist) for hist in features],
        dtype=np.float64,
    )

    sampled_fps = fps / sample_stride if fps > 0 else 1.0
    trend_window = max(1, int(round(trend_window_seconds * sampled_fps)))
    min_gap = max(1, int(round(min_subshot_seconds * sampled_fps)))
    boundaries = _motion_trend_boundaries(
        dissimilarities, trend_window, min_gap, trend_percentile
    )
    cluster_count = max(1, len(boundaries) + 1)
    medoid_sample_indexes = _global_kmeans_medoids(features, cluster_count)

    selected = [
        SelectedFrame(
            frame_indexes[idx], timestamps[idx], float(dissimilarities[idx])
        )
        for idx in medoid_sample_indexes
    ]
    written = _extract_selected_frames(
        video_path,
        selected,
        image_dir,
        image_quality,
        output_width,
        output_height,
    )
    _write_map(written, video_path, map_path, fps, "motion_kmeans_2010")
    return {
        "method": "motion_kmeans_2010",
        "video": str(video_path),
        "keyframes": len(written),
        "subshots": cluster_count,
        "sample_stride": sample_stride,
        "elapsed_sec": time.perf_counter() - started,
    }
