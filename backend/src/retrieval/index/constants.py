"""Shared constants and helpers for the retrieval index layer.

Both `ClipFaissIndex` and `ClipMilvusIndex` import from here so that the
display columns and device resolution stay single-sourced across backends.
"""

from __future__ import annotations

import torch

# Cột metadata copy nguyên văn khi enrich OCR/ASR hits (orchestrator.py),
# để OCR/ASR trả về shape giống hệt semantic search.
METADATA_DISPLAY_COLUMNS = (
    "dataset",
    "video_id",
    "keyframe_id",
    "keyframe_id_int",
    "source_name",
    "frame_idx",
    "timestamp_sec",
    "fps",
    "keyframe_path",
)


def resolve_device(device_name: str) -> torch.device:
    """Resolve target device cho các phép toán PyTorch."""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_name)
    return torch.device("cpu")
