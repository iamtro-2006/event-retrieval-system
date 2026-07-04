from __future__ import annotations

from pathlib import Path

import cv2
from torch.utils.data import DataLoader, Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class ImageDataset(Dataset):
    """Loads all keyframe images inside a single video's keyframe folder.

    Equivalent to the original PaddleOCR/my_dataset.py::ImageDataset,
    but recognizes the multiple image extensions used across the backend
    (see src/embedding_extraction/models/embedder.py).
    """

    def __init__(self, image_dir: str | Path):
        self.image_paths = sorted(
            p for p in Path(image_dir).iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        img = cv2.imread(str(path))
        if img is None:
            return None
        return str(path), img


def collate_fn(batch):
    return [item for item in batch if item is not None]


def build_loader(
    image_dir: str | Path,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = False,
) -> tuple[DataLoader, int]:
    dataset = ImageDataset(image_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        shuffle=shuffle,
    )
    return loader, len(dataset)
