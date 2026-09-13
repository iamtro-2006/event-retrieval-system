from __future__ import annotations

import numpy as np

COLOR_NAMES = (
    "black", "blue", "brown", "grey", "green", "orange",
    "pink", "purple", "red", "white", "yellow",
)

# Stable, recognizable representatives for the 11-colour UI palette.
PALETTE_RGB = np.asarray([
    (0, 0, 0), (35, 90, 200), (130, 80, 45), (128, 128, 128),
    (30, 150, 60), (245, 140, 20), (245, 130, 180),
    (130, 70, 180), (210, 40, 40), (245, 245, 245), (245, 220, 40),
], dtype=np.uint8)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert uint8/float sRGB arrays ending in RGB channels to CIE Lab."""
    values = np.asarray(rgb, dtype=np.float32) / 255.0
    linear = np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)
    xyz = linear @ np.asarray([
        [0.4124564, 0.2126729, 0.0193339],
        [0.3575761, 0.7151522, 0.1191920],
        [0.1804375, 0.0721750, 0.9503041],
    ], dtype=np.float32)
    xyz /= np.asarray([0.95047, 1.0, 1.08883], dtype=np.float32)
    delta = 6 / 29
    f = np.where(xyz > delta ** 3, np.cbrt(xyz), xyz / (3 * delta ** 2) + 4 / 29)
    return np.stack((116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])), axis=-1)


PALETTE_LAB = rgb_to_lab(PALETTE_RGB)


def dominant_palette_index(rgb: np.ndarray) -> int:
    pixels = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)
    if not len(pixels):
        return 0
    # Sampling bounds extraction time for large source frames without changing
    # the equal-area 5 x 5 geometry.
    step = max(1, len(pixels) // 4096)
    lab = rgb_to_lab(pixels[::step])
    nearest = np.argmin(np.sum((lab[:, None, :] - PALETTE_LAB[None, :, :]) ** 2, axis=2), axis=1)
    return int(np.bincount(nearest, minlength=len(COLOR_NAMES)).argmax())


def palette_indices(rgb: np.ndarray) -> np.ndarray:
    """Label every pixel in one vectorized Lab-distance operation."""
    values = np.asarray(rgb, dtype=np.uint8)
    lab = rgb_to_lab(values.reshape(-1, 3))
    nearest = np.argmin(np.sum((lab[:, None, :] - PALETTE_LAB[None, :, :]) ** 2, axis=2), axis=1)
    return nearest.reshape(values.shape[:-1]).astype(np.uint8)
