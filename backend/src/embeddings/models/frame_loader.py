from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import cv2
import decord
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

Transform = Callable[[Image.Image], torch.Tensor]


@dataclass(frozen=True)
class FrameBatch:
    frames: torch.Tensor
    frame_indexes: torch.Tensor


class VideoFrameDataset(Dataset):
    def __init__(self, video_path: str | Path, transform: Transform | None = None, backend: str = "decord"):
        self.video_path = Path(video_path)
        self.transform = transform
        self.backend = backend.lower()
        self._reader = None
        self._length = self._probe_length()

    def _probe_length(self) -> int:
        if self.backend == "decord":
            return len(decord.VideoReader(str(self.video_path), ctx=decord.cpu(0)))
        if self.backend == "opencv":
            cap = cv2.VideoCapture(str(self.video_path))
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return length
        raise ValueError(f"Unsupported frame backend: {self.backend}")

    def _get_decord_reader(self):
        if self._reader is None:
            self._reader = decord.VideoReader(str(self.video_path), ctx=decord.cpu(0))
        return self._reader

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        index = int(index)
        if self.backend == "decord":
            array = self._get_decord_reader()[index].asnumpy()
        else:
            cap = cv2.VideoCapture(str(self.video_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            cap.release()
            if not ok:
                raise IndexError(f"Cannot read frame {index} from {self.video_path}")
            array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(array)
        tensor = self.transform(image) if self.transform else torch.from_numpy(array).permute(2, 0, 1)
        return {"frames": tensor, "frame_indexes": torch.tensor(index, dtype=torch.long)}


def build_frame_loader(
    video_path: str | Path,
    transform: Transform | None,
    batch_size: int,
    backend: str = "decord",
    num_workers: int = 0,
    pin_memory: bool = True,
    persistent_workers: bool = False,
) -> DataLoader:
    dataset = VideoFrameDataset(video_path=video_path, transform=transform, backend=backend)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        drop_last=False,
    )


def iter_frame_batches(loader: DataLoader) -> Iterator[FrameBatch]:
    for batch in loader:
        yield FrameBatch(frames=batch["frames"], frame_indexes=batch["frame_indexes"])
