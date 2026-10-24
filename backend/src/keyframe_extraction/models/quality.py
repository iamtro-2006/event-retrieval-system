"""Batched GPU quality scores for sampled laparoscopic frames."""
import torch
import torch.nn.functional as F


def score_batch(rgb: torch.Tensor, min_hist_bins: int = 5):
    """Bounded batched quality measurement; return only decisions and scores to CPU."""
    _, _, height, width = rgb.shape
    roi = rgb[:, :, height // 4:3 * height // 4, width // 5:4 * width // 5]
    small = F.interpolate(roi.float(), size=(128, 128), mode='area')
    gray = .299 * small[:, 0] + .587 * small[:, 1] + .114 * small[:, 2]
    bright = (gray > 220).float().mean(dim=(1, 2))
    contrast = gray.flatten(1).std(dim=1)
    kernel = gray.new_tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])[None, None]
    sharpness = F.conv2d(gray[:, None], kernel)[:, 0].flatten(1).var(dim=1)
    # Flatness must inspect the whole frame: captions/indicators near the border
    # can make a dark frame informative even when the surgical center is black.
    full_small = F.interpolate(rgb.float(), size=(128, 128), mode='area')
    bins = (full_small.permute(0, 2, 3, 1).reshape(len(rgb), -1, 3) * (8 / 256)).long().clamp(0, 7)
    indexes = bins[:, :, 0] * 64 + bins[:, :, 1] * 8 + bins[:, :, 2]
    hist = torch.zeros((len(rgb), 512), device=rgb.device, dtype=torch.int32)
    hist.scatter_add_(1, indexes, torch.ones_like(indexes, dtype=torch.int32))
    occupied = (hist > 0).sum(1)
    valid = ((occupied > min_hist_bins) & (bright < .50) &
             (sharpness >= 15.) & (contrast >= 12.))
    score = sharpness + contrast - 100 * bright + .1 * occupied
    return (valid.cpu().numpy(), score.cpu().numpy(),
            (occupied > min_hist_bins).cpu().numpy(), sharpness.cpu().numpy())



