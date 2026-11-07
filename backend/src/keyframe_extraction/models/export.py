"""Export current VLM keyframes as CUDA JPEGs and source-timestamp CSV maps."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.utils.gpu_video import CudaVideoReader
from src.utils.video_io import get_video_fps


def save_keyframe_map(indexes: list[int], video_path: Path, csv_path: Path,
                      fps: float | None = None, timestamps: np.ndarray | None = None) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fps = get_video_fps(video_path) if fps is None else fps
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('Original video FPS must be positive')
    times = [idx/fps for idx in indexes] if timestamps is None else np.asarray(timestamps,dtype=np.float64)
    if len(times) != len(indexes) or not np.isfinite(times).all():
        raise ValueError('Keyframe timestamps must align with source frame indexes')
    pd.DataFrame(
        {
            "keyframe_id": list(range(len(indexes))),
            "video_id": video_path.stem,
            "frame_idx": indexes,
            "timestamp_sec": times,
            "fps": [fps] * len(indexes),
        }
    ).to_csv(csv_path, index=False)


def _write_encoded(path, encoded):
    path.write_bytes(encoded.cpu().numpy().tobytes())


def export_selected_cuda(video: Path, indexes: list[int], image_dir: Path,
                         quality: int, batch_size: int = 8, save_images: bool = True) -> list[int]:
    if indexes != sorted(set(indexes)) or (indexes and indexes[0] < 0):
        raise ValueError('Selected indexes must be sorted and unique')
    if not indexes or not save_images:
        return indexes
    from torchvision.io import encode_jpeg
    image_dir.mkdir(parents=True, exist_ok=True)
    pointer = 0
    pending = []
    with CudaVideoReader(video, batch_size) as reader, \
            ThreadPoolExecutor(max_workers=2) as writers, \
            tqdm(total=len(indexes), desc='Keyframe JPEG', unit='frame') as bar, \
            torch.inference_mode():
        for start, rgb in reader:
            frames, paths = [], []
            while pointer < len(indexes) and indexes[pointer] < start + len(rgb):
                if indexes[pointer] < start:
                    raise RuntimeError('Selected frame skipped by decoder')
                frames.append(rgb[indexes[pointer] - start].contiguous())
                paths.append(image_dir / f'{pointer:06d}.jpg')
                pointer += 1
                bar.update(1)
            if frames:
                for path, encoded in zip(paths, encode_jpeg(frames, quality=quality)):
                    pending.append(writers.submit(_write_encoded, path, encoded))
                    if len(pending) >= 2:
                        pending.pop(0).result()
            if pointer == len(indexes):
                break
        for future in pending:
            future.result()
    if pointer != len(indexes):
        raise RuntimeError('Video ended before keyframes were exported')
    return indexes
