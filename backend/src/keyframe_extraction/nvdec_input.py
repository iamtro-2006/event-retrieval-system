"""Convert only NVDEC-incompatible inputs; preserve display indexes/timestamps."""
from pathlib import Path
import json
import subprocess

import numpy as np

from src.utils.frame_sampling import probe_video_timeline


def probe_video_format(video):
    result = subprocess.run([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
        'stream=codec_name,profile,pix_fmt,width,height', '-of', 'json', str(video),
    ], capture_output=True, text=True, check=True)
    streams = json.loads(result.stdout).get('streams', [])
    if not streams:
        raise ValueError(f'No video stream: {video}')
    return streams[0]


def incompatibility(info, compute_major):
    """Known NVDEC constraints; uncertain formats still go through a real probe."""
    codec, pixel = info.get('codec_name', ''), info.get('pix_fmt', '')
    supported = {'h264', 'hevc', 'vp8', 'vp9', 'mpeg1video', 'mpeg2video',
                 'mpeg4', 'vc1', 'wmv3', 'mjpeg', 'av1'}
    if codec not in supported:
        return f'NVDEC does not support codec {codec or "unknown"}'
    if codec == 'av1' and compute_major < 8:
        return 'AV1 hardware decoding is unavailable on Turing/T4'
    if codec == 'h264' and compute_major < 10 and pixel and pixel not in {'yuv420p', 'yuvj420p'}:
        return f'H.264 {info.get("profile", "")} / {pixel} is unsupported on this GPU'
    if codec == 'hevc' and compute_major < 10 and '422' in pixel:
        return f'HEVC 4:2:2 is unsupported on this GPU ({pixel})'
    return None


def explicit_unsupported_error(exc):
    """Do not convert on OOM, corrupt data, frame mismatch or generic error 300."""
    while exc is not None:
        text = str(exc).lower()
        if any(token in text for token in ('cuda_error_not_supported', 'codec not supported',
                'unsupported codec', 'unsupported profile', 'unsupported chroma',
                'unsupported bit depth', 'codec is not supported')):
            return True
        exc = exc.__cause__
    return False


def validate_conversion(source, converted):
    source_times, _ = probe_video_timeline(source)
    converted_times, _ = probe_video_timeline(converted)
    if (source_times.shape != converted_times.shape
            or not np.allclose(source_times, converted_times, rtol=0., atol=2e-6)):
        raise ValueError('H.264 conversion changed display frame count/timestamps; refusing wrong mapping')


def prepare_nvdec_input(source, workspace, probe_nvdec, compute_major=7):
    """Keep compatible source; otherwise CPU decode once + NVENC H.264 encode.

    The source is never overwritten. Output retains source stem, resolution and
    presentation cadence. Transcoding changes pixels (4:2:0/lossy compression).
    A valid conversion is reusable across retries; all files live in job workspace.
    """
    source, workspace = Path(source), Path(workspace)
    info = probe_video_format(source)
    reason = incompatibility(info, compute_major)
    if reason is None:
        try:
            probe_nvdec(source)
            return source
        except RuntimeError as exc:
            if not explicit_unsupported_error(exc):
                raise
            reason = str(exc)
    print(f'[Transcode] {source.name}: {reason}; CPU decode -> H.264 NVENC', flush=True)
    folder = workspace / 'transcoded'
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f'{source.stem}.mp4'
    partial = folder / f'{source.stem}.partial.mp4'
    manifest = folder / f'{source.stem}.json'
    stat = source.stat()
    signature = dict(version=1, source=str(source.resolve()), size=stat.st_size,
                     mtime_ns=stat.st_mtime_ns, codec=info)
    if target.is_file() and manifest.is_file():
        try:
            reusable = json.loads(manifest.read_text(encoding='utf-8')) == signature
        except (ValueError, OSError):
            reusable = False
        if reusable:
            validate_conversion(source, target)
            probe_nvdec(target)
            print(f'[Transcode] reuse {target.name}', flush=True)
            return target
    # Do not force CFR, set -r, or resize: those could change source frame mapping.
    command = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning', '-nostdin', '-stats', '-stats_period', '2',
        '-i', str(source), '-map', '0:v:0', '-an', '-sn', '-dn',
        '-c:v', 'h264_nvenc', '-gpu', '0', '-preset', 'p1', '-rc', 'vbr',
        '-cq', '18', '-b:v', '0', '-pix_fmt', 'yuv420p', '-bf', '0',
        '-fps_mode', 'passthrough', '-enc_time_base:v', '1:1000000',
        '-video_track_timescale', '1000000', str(partial)]
    try:
        subprocess.run(command, check=True)  # stream FFmpeg progress; no full log in RAM
        if not partial.is_file() or not partial.stat().st_size:
            raise RuntimeError('H.264 conversion produced no video')
        validate_conversion(source, partial)
        probe_nvdec(partial)
        partial.replace(target)
        manifest.write_text(json.dumps(signature, indent=2), encoding='utf-8')
    finally:
        partial.unlink(missing_ok=True)
    print(f'[Transcode] ready {target.name}; frame count/timeline preserved', flush=True)
    return target
