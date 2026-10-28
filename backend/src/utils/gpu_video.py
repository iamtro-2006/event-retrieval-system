"""Bounded NVDEC -> CUDA tensors. No host RGB cache and no CPU decode fallback."""
from __future__ import annotations

import gc
from pathlib import Path
import torch
import torch.nn.functional as F


class CudaVideoReader:
    def __init__(self, path: Path, batch_size: int = 16, diagnostics: bool = False):
        if not torch.cuda.is_available():
            raise RuntimeError('NVDEC requires a CUDA GPU; select Kaggle T4.')
        if batch_size < 1:
            raise ValueError('decode batch_size must be positive')
        self.diagnostics = diagnostics
        self.path = path
        if diagnostics:
            print(f'[NVDEC] {path.name} importing PyNvVideoCodec', flush=True)
        import PyNvVideoCodec as nvc
        self.batch_size = batch_size
        self.decoder = None
        try:
            if diagnostics:
                print(f'[NVDEC] {path.name} creating ThreadedDecoder; '
                      f'PyNvVideoCodec={getattr(nvc, "__version__", "unknown")} '
                      f'torch={torch.__version__} CUDA={torch.version.cuda} '
                      f'gpu={torch.cuda.current_device()}', flush=True)
            self.decoder = nvc.ThreadedDecoder(
                enc_file_path=str(path), gpu_id=torch.cuda.current_device(),
                buffer_size=2 * batch_size, use_device_memory=True,
                output_color_type=nvc.OutputColorType.RGBP,
            )
            if diagnostics:
                print(f'[NVDEC] {path.name} decoder created; reading stream metadata', flush=True)
            self.total = int(self.decoder.get_stream_metadata().num_frames)
            if self.total < 1:
                raise ValueError('Decoder metadata contains no frames')
            if diagnostics:
                print(f'[NVDEC] {path.name} ready total_frames={self.total}', flush=True)
        except Exception as exc:
            self.close()
            raise RuntimeError(
                'Cannot initialize NVDEC. Check T4, NVIDIA driver, libnvcuvid and codec support. '
                'CPU fallback is disabled.'
            ) from exc

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __iter__(self):
        count = 0
        while True:
            if self.diagnostics and count == 0:
                print(f'[NVDEC] {self.path.name} requesting first batch', flush=True)
            frames = self.decoder.get_batch_frames(self.batch_size)
            if not frames:
                break
            if self.diagnostics and count == 0:
                print(f'[NVDEC] {self.path.name} first batch received frames={len(frames)}; DLPack -> torch', flush=True)
            batch = torch.stack([torch.from_dlpack(frame) for frame in frames])
            if not batch.is_cuda or batch.dtype != torch.uint8 or batch.ndim != 4 or batch.shape[1] != 3:
                raise RuntimeError('NVDEC must return CUDA uint8 RGB in NCHW order')
            if self.diagnostics and count == 0:
                print(f'[NVDEC] {self.path.name} first CUDA tensor ready shape={tuple(batch.shape)}', flush=True)
            yield count, batch
            count += len(batch)
            del batch, frames
        if count != self.total:
            raise RuntimeError(f'NVDEC decoded {count}/{self.total} frames; refusing misaligned embeddings')

    def close(self):
        if self.decoder is not None:
            torch.cuda.synchronize()
            self.decoder = None
            gc.collect()


def pe_preprocess_cuda(rgb: torch.Tensor, size: int) -> torch.Tensor:
    """PE's default squash bilinear resize, RGB, mean/std .5, on GPU.

    Tensor interpolation can differ slightly from the original PIL implementation.
    Resize uint8 before float conversion to retain the 8-bit preprocessing contract.
    """
    from torchvision.transforms.functional import resize
    from torchvision.transforms import InterpolationMode
    resized = resize(rgb, [size, size], interpolation=InterpolationMode.BILINEAR, antialias=True)
    return resized.float().div_(255).sub_(0.5).div_(0.5)


def normalized_omni_clip(rgb: torch.Tensor) -> torch.Tensor:
    """Same channel normalization as OmniShotCut's validation transform."""
    x = rgb.float().div(255)
    mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x - mean) / std


def iter_gpu_windows(batches, total: int, window: int, overlap: int):
    """Match upstream split_videos padding/stride/valid regions, retaining one window."""
    if total < 1 or window < 1 or not 0 <= overlap < window:
        raise ValueError('Invalid OmniShotCut window geometry')
    batches = iter(batches)
    buffer = None
    start = 0
    while start < total:
        valid_len = min(window, total - start)
        while buffer is None or len(buffer) < valid_len:
            try:
                batch = next(batches)
            except StopIteration as exc:
                raise RuntimeError('Decoder ended before the final shot window') from exc
            buffer = batch if buffer is None else torch.cat([buffer, batch])
        clip = buffer[:valid_len]
        pad = window - valid_len
        if pad:
            clip = torch.cat([clip, clip.new_zeros((pad, *clip.shape[1:]))])
        end = start + window
        left = start if start == 0 else start + overlap // 2
        right = total if end >= total else end - (overlap - overlap // 2)
        yield clip, pad, start, left, right, valid_len
        if end >= total:
            # Exhaust the underlying reader so its decoded frame count is validated.
            if len(buffer) != valid_len:
                raise RuntimeError('Decoder produced excess frames')
            for extra in batches:
                if len(extra):
                    raise RuntimeError('Decoder produced excess frames')
            break
        stride = window - overlap
        buffer = buffer[stride:].clone()
        start += stride


def resized_batches(reader: CudaVideoReader, height: int, width: int):
    for _, rgb in reader:
        # Area resize on GPU; no full-video host decode. Quantize before normalization.
        yield F.interpolate(rgb.float(), size=(height, width), mode='area').round().clamp_(0, 255).byte()
