"""Full-frame GPU quality gate for bounded NVDEC batches."""
import math
import torch
import torch.nn.functional as F


def score_batch(rgb: torch.Tensor, min_hist_bins: int = 5):
    """Return valid, quality [0,1], nonflat, sharpness; retain no host RGB.

    v3c thresholds/weights and Canny-style GPU edge density (80/160).
    Color eligibility/fallback follows VLM. Exposure clipping is not a gate.
    """
    height, width = rgb.shape[-2:]
    scale = min(1., 320. / width)
    small = F.interpolate(rgb.float(), size=(max(1, round(height * scale)),
                                             max(1, round(width * scale))), mode='area')
    gray = .299 * small[:, 0] + .587 * small[:, 1] + .114 * small[:, 2]
    x = F.pad(gray[:, None], (1, 1, 1, 1), mode='replicate')
    lap = gray.new_tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])[None, None]
    blur = F.conv2d(x, lap).flatten(1).var(1, unbiased=False)
    brightness = gray.mean((1, 2))
    gray_bins = gray.flatten(1).long().clamp(0, 255)
    hist_gray = torch.zeros((len(rgb), 256), device=rgb.device)
    hist_gray.scatter_add_(1, gray_bins, torch.ones_like(gray_bins, dtype=torch.float32))
    probability = hist_gray / gray_bins.shape[1]
    entropy = -(probability * probability.clamp_min(1e-12).log2()).sum(1)
    sobel = gray.new_tensor([[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                             [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]])[:, None]
    gradients = F.conv2d(x, sobel)
    edge_density = canny_density(gradients, 80., 160.)
    color_small = F.interpolate(rgb.float(), size=(128, 128), mode='area')
    bins = (color_small.permute(0, 2, 3, 1).flatten(1, 2) * (8 / 256)).long().clamp(0, 7)
    indexes = bins[:, :, 0] * 64 + bins[:, :, 1] * 8 + bins[:, :, 2]
    hist = torch.zeros((len(rgb), 512), device=rgb.device, dtype=torch.int32)
    hist.scatter_add_(1, indexes, torch.ones_like(indexes, dtype=torch.int32))
    nonflat = (hist > 0).sum(1) > min_hist_bins
    valid = (nonflat & (blur >= 20.) & (brightness >= 10.) & (brightness <= 245.)
             & ~((entropy < 1.2) & (edge_density < .002)))
    sharpness = (torch.log1p(blur) / math.log1p(500.)).clamp(max=1.)
    exposure = (1. - (brightness - 127.5).abs() / 127.5).clamp(min=0.)
    clipped = ((gray <= 5.) | (gray >= 250.)).float().mean((1, 2))
    exposure *= (1. - clipped / .60).clamp(min=0.)
    quality = (.35 * sharpness + .25 * exposure + .20 * (entropy / 7.).clamp(max=1.)
               + .20 * (edge_density / .12).clamp(max=1.))
    return tuple(value.cpu().numpy() for value in (valid, quality, nonflat, blur))


def canny_density(gradients, low, high):
    """Sobel L1, directional suppression and converged 8-neighbor hysteresis.

    GPU implementation of the Canny stages used for density; floating point
    resize/gray and border ties may differ from OpenCV, so not bitwise equal.
    """
    gx, gy = gradients[:, 0], gradients[:, 1]
    magnitude = gx.abs() + gy.abs()
    angle = (torch.atan2(gy, gx) * (180. / math.pi)) % 180.
    directions = ((angle + 22.5) / 45.).long() % 4
    peaks = torch.zeros_like(magnitude, dtype=torch.bool)
    for direction, (dy, dx) in enumerate(((0, 1), (1, 1), (1, 0), (1, -1))):
        first = torch.roll(magnitude, (dy, dx), dims=(1, 2))
        second = torch.roll(magnitude, (-dy, -dx), dims=(1, 2))
        peaks |= (directions == direction) & (magnitude > first) & (magnitude >= second)
    peaks[:, 0] = False
    peaks[:, -1] = False
    peaks[:, :, 0] = False
    peaks[:, :, -1] = False
    eligible = peaks & (magnitude > low)
    edges = peaks & (magnitude > high)
    while True:
        expanded = eligible & (F.max_pool2d(edges[:, None].float(), 3, stride=1, padding=1)[:, 0] > 0)
        if torch.equal(expanded, edges):
            break
        edges = expanded
    return edges.float().mean((1, 2))
