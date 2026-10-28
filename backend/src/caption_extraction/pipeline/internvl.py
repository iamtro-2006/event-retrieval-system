"""InternVL GitHub-format video inputs, following its official model card."""
from __future__ import annotations

import numpy as np


def prepare_video(frames, image_size: int, max_tiles: int = 1):
    import torch
    from PIL import Image

    ratios = sorted({(i, j) for i in range(1, max_tiles + 1)
                     for j in range(1, max_tiles + 1) if i * j <= max_tiles},
                    key=lambda ratio: ratio[0] * ratio[1])
    if not frames or image_size <= 0 or not ratios:
        raise ValueError("InternVL needs frames, a positive image size and max_tiles")
    mean = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
    pixels, counts = [], []
    for frame in frames:
        image = Image.fromarray(np.ascontiguousarray(frame.transpose(1, 2, 0))).convert("RGB")
        width, height = image.size
        best, best_diff = (1, 1), float("inf")
        for cols, rows in ratios:
            diff = abs(width / height - cols / rows)
            if diff < best_diff or (diff == best_diff and
                                   width * height > 0.5 * image_size ** 2 * cols * rows):
                best, best_diff = (cols, rows), diff
        cols, rows = best
        resized = image.resize((image_size * cols, image_size * rows), Image.Resampling.BICUBIC)
        tiles = [resized.crop((x * image_size, y * image_size,
                               (x + 1) * image_size, (y + 1) * image_size))
                 for y in range(rows) for x in range(cols)]
        if len(tiles) > 1:
            tiles.append(image.resize((image_size, image_size), Image.Resampling.BICUBIC))
        counts.append(len(tiles))
        for tile in tiles:
            tensor = torch.from_numpy(np.array(tile, copy=True)).permute(2, 0, 1).float() / 255
            pixels.append((tensor - mean) / std)
    return torch.stack(pixels), counts


def video_question(timestamps, prompt: str) -> str:
    return "".join(f"Frame{i} ({timestamp:.3f}s): <image>\n"
                   for i, timestamp in enumerate(timestamps, 1)) + prompt
