"""Timestamp sampling with explicit source-frame indexes; no video frame decoding."""
from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
import pickle
import subprocess
import numpy as np

CACHE_FORMAT = 'vlm-sampled-v1'


def merge_consecutive_short_scenes(scenes: np.ndarray, timestamps: np.ndarray,
                                   source_fps: float, threshold_seconds: float = 1.) -> np.ndarray:
    """Absorb every sub-threshold shot into an adjacent shot until none remains."""
    shots = np.asarray(scenes, dtype=np.int64).reshape(-1, 2)
    times = np.asarray(timestamps, dtype=np.float64)
    if (not len(shots) or not len(times) or source_fps <= 0 or threshold_seconds <= 0
            or np.any(shots < 0) or np.any(shots[:, 1] >= len(times))
            or np.any(shots[:, 0] > shots[:, 1])
            or np.any(shots[1:, 0] <= shots[:-1, 1])):
        raise ValueError('Invalid scene ranges or timeline for short-shot merging')
    merged = [row.tolist() for row in shots]

    def duration(first, last):
        next_time = times[last + 1] if last + 1 < len(times) else times[last] + 1. / source_fps
        return next_time - times[first]

    i = 0
    while i < len(merged) and len(merged) > 1:
        if duration(*merged[i]) >= threshold_seconds:
            i += 1
            continue
        if i + 1 < len(merged):
            merged[i][1] = merged[i + 1][1]
            del merged[i + 1]
        else:
            merged[i - 1][1] = merged[i][1]
            del merged[i]
            i -= 1
    return np.asarray(merged, dtype=scenes.dtype).reshape(-1, 2)


def save_embedding_cache(data, path: Path) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.partial')
    try:
        with temporary.open('wb') as handle:
            pickle.dump(data,handle,protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def probe_video_timeline(video: Path) -> tuple[np.ndarray, float]:
    """Read packet PTS in presentation order, without CPU decoding RGB frames.

    This path requires one timestamped video packet per displayed frame (normal
    MP4/H.264/H.265). Counts are checked against metadata and again against NVDEC;
    missing/duplicate PTS or unmatched counts fail instead of inventing timestamps.
    """
    info = json.loads(subprocess.check_output([
        'ffprobe','-v','error','-select_streams','v:0','-show_entries',
        'stream=avg_frame_rate,nb_frames','-of','json',str(video),
    ],text=True))
    streams = info.get('streams',[])
    if not streams:
        raise ValueError(f'No video stream: {video}')
    stream = streams[0]
    fps = float(Fraction(stream.get('avg_frame_rate','0/1')))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('Video has no valid original FPS')
    result = subprocess.run([
        'ffprobe','-v','error','-select_streams','v:0','-show_entries',
        'packet=pts_time','-of','csv=p=0',str(video),
    ],capture_output=True,text=True,check=True)
    try:
        timestamps = np.asarray([float(line.split(',')[0])
                                 for line in result.stdout.splitlines() if line.strip()],dtype=np.float64)
    except ValueError as exc:
        raise ValueError('Video packets contain missing/invalid PTS; cannot map sampled frames safely') from exc
    timestamps.sort()  # Compressed packets can be in B-frame decode order.
    if (not len(timestamps) or not np.isfinite(timestamps).all()
            or np.any(np.diff(timestamps) <= 0)):
        raise ValueError('Video requires unique increasing presentation timestamps')
    declared = stream.get('nb_frames')
    if declared not in (None,'N/A','0') and int(declared) != len(timestamps):
        raise ValueError('Packet/frame counts differ; cannot use packet PTS as the frame timeline')
    timestamps -= timestamps[0]  # Time relative to the first displayed video frame.
    return timestamps,fps


def build_sample_plan(scenes: np.ndarray, timestamps: np.ndarray, source_fps: float,
                      sample_fps: float = 3., max_scene_seconds: float | None = 20.,
                      max_gap: int = 5000) -> dict:
    times = np.asarray(timestamps,dtype=np.float64)
    scenes = np.asarray(scenes,dtype=np.int64).reshape(-1,2)
    if (not np.isfinite(sample_fps) or sample_fps <= 0 or not np.isfinite(source_fps)
            or source_fps <= 0 or max_gap < 1):
        raise ValueError('Sampling FPS/source FPS/max_gap must be positive')
    if (max_scene_seconds is not None
            and (not np.isfinite(max_scene_seconds) or max_scene_seconds <= 0)):
        raise ValueError('max_scene_seconds must be positive')
    if (not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
            or not len(scenes) or np.any(scenes < 0) or np.any(scenes[:,0] > scenes[:,1])
            or np.any(scenes[:,1] >= len(times))
            or np.any(scenes[1:,0] <= scenes[:-1,1])):
        raise ValueError('Scenes do not match the source frame timeline')
    indexes, ranges = [], []
    step = 1. / sample_fps
    for first,last in scenes.tolist():
        start = first
        while start <= last:
            if max_scene_seconds is None:
                end = min(last,start+max_gap)
            else:
                end = min(last,max(start,int(np.searchsorted(times,times[start]+max_scene_seconds-1e-6))-1))
            local_times = times[start:end+1]
            if local_times[-1]-local_times[0] < step-1e-6:
                picked = np.array([(start+end)//2],dtype=np.int64)
            else:
                count = int(np.floor((local_times[-1]-local_times[0]+1e-6)*sample_fps))+1
                targets = local_times[0]+np.arange(count,dtype=np.float64)*step
                offsets = np.searchsorted(local_times,targets-1e-6)
                picked = np.unique(start+offsets[offsets < len(local_times)])
            offset = len(indexes)
            indexes.extend(picked.tolist())
            ranges.append([offset,len(indexes)-1])
            start = end+1
    indexes = np.asarray(indexes,dtype=np.int64)
    return dict(format=CACHE_FORMAT, frame_indexes=indexes, timestamps=times[indexes],
                sample_ranges=np.asarray(ranges,dtype=np.int64), sample_fps=float(sample_fps),
                source_total_frames=len(times), source_fps=float(source_fps),
                model_name='PE-Core-L14-336')


def filter_sample_plan(plan: dict, mask: np.ndarray) -> dict:
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != plan['frame_indexes'].shape or not mask.any():
        raise ValueError('Invalid quality mask')
    ranges = []
    offset = 0
    for first, last in plan['sample_ranges']:
        count = int(mask[first:last + 1].sum())
        if count:
            ranges.append((offset, offset + count - 1))
            offset += count
    return dict(plan, frame_indexes=plan['frame_indexes'][mask],
                timestamps=plan['timestamps'][mask],
                sample_ranges=np.asarray(ranges, dtype=np.int64).reshape(-1, 2))


def load_sampled_cache(path: Path, plan: dict) -> dict | None:
    """Reject legacy dense caches before allocating their potentially huge matrices."""
    expected = len(plan['frame_indexes'])
    # Our format is float32 Nx1024 plus small index/time arrays. A dense cache for
    # a high-FPS source is much larger; do not unpickle it just to reject its type.
    if not path.is_file() or path.stat().st_size > expected*(1024*4+32)+65536:
        return None
    try:
        with path.open('rb') as handle:
            data = pickle.load(handle)
        if not isinstance(data,dict) or data.get('format') != CACHE_FORMAT:
            return None
        for key in ['source_total_frames','sample_fps','source_fps','model_name']:
            if data.get(key) != plan[key]:
                return None
        for key in ['frame_indexes','timestamps','sample_ranges']:
            if not np.array_equal(data.get(key),plan[key]):
                return None
        features = data['features']
        if not isinstance(features,np.ndarray) or features.dtype != np.float32 or features.shape != (expected,1024):
            return None
        return data
    except (OSError,EOFError,pickle.UnpicklingError,KeyError,ValueError,TypeError):
        return None


def load_quality_sampled_cache(path: Path, source_plan: dict, source_scenes: np.ndarray,
                               min_hist_bins: int = 5) -> dict | None:
    """Reuse fused quality+PE features without re-decoding the source video."""
    count = len(source_plan['frame_indexes'])
    if not path.is_file() or path.stat().st_size > count * (1024 * 4 + 64) + 65536:
        return None
    try:
        with path.open('rb') as handle:
            data = pickle.load(handle)
        if not isinstance(data, dict) or data.get('quality_policy') != f'fused-v5-sharpness-{min_hist_bins}bins':
            return None
        if not np.array_equal(data.get('source_scenes'), source_scenes):
            return None
        kept_scenes = np.asarray(data.get('kept_scenes'))
        if kept_scenes.ndim != 2 or kept_scenes.shape[1] != 2:
            return None
        if any(not np.any(np.all(source_scenes == shot, axis=1)) for shot in kept_scenes):
            return None
        if not np.array_equal(data.get('source_frame_indexes'), source_plan['frame_indexes']):
            return None
        for key in ('source_total_frames', 'sample_fps', 'source_fps', 'model_name'):
            if data.get(key) != source_plan[key]:
                return None
        chosen = np.asarray(data['frame_indexes'])
        if chosen.ndim != 1 or not len(chosen) or np.any(np.diff(chosen) <= 0):
            return None
        positions = np.searchsorted(source_plan['frame_indexes'], chosen)
        if (np.any(positions >= count) or
                not np.array_equal(source_plan['frame_indexes'][positions], chosen) or
                not np.array_equal(source_plan['timestamps'][positions], data['timestamps'])):
            return None
        mask = np.zeros(count, dtype=bool)
        mask[positions] = True
        expected = filter_sample_plan(source_plan, mask)
        if not np.array_equal(data.get('sample_ranges'), expected['sample_ranges']):
            return None
        features = data['features']
        if not isinstance(features, np.ndarray) or features.dtype != np.float32 or features.shape != (len(chosen), 1024):
            return None
        sharpness = data.get('sharpness')
        if (not isinstance(sharpness, np.ndarray) or sharpness.shape != (len(chosen),)
                or not np.isfinite(sharpness).all() or np.any(sharpness < 0)):
            return None
        return data
    except (OSError, EOFError, pickle.UnpicklingError, KeyError, ValueError, TypeError):
        return None
