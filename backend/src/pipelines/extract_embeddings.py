from __future__ import annotations

from pathlib import Path
import logging

import numpy as np
from tqdm import tqdm

from src.models.embedder import (
    load_clip_model,
    list_image_paths,
    encode_keyframe_images,
)


class ExtractEmbeddingPipeline:
    def __init__(self, cfg: dict, logger: logging.Logger | None = None):
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)

        self.input_root = Path(cfg["input_keyframes_root"])
        self.output_root = Path(cfg["output_embeddings_root"])

        self.batch_size = int(cfg["model"]["batch_size"])
        self.normalize = bool(cfg["model"].get("normalize", True))
        self.overwrite = bool(cfg["save"].get("overwrite", True))
        self.output_ext = str(cfg["save"].get("extension", ".npy"))

        self.model, self.preprocess, self.device, self.precision = load_clip_model(
            model_name=cfg["model"]["name"],
            pretrained=cfg["model"]["pretrained"],
            precision=cfg["model"].get("precision", "fp32"),
            device_name=cfg["model"].get("device", "auto"),
            logger=self.logger,
        )

    def scan_video_dirs(self) -> list[Path]:
        if not self.input_root.exists():
            raise FileNotFoundError(f"Keyframes root not found: {self.input_root}")

        video_dirs = []

        for dataset_dir in sorted(self.input_root.iterdir()):
            if not dataset_dir.is_dir():
                continue

            for video_dir in sorted(dataset_dir.iterdir()):
                if not video_dir.is_dir():
                    continue

                if list_image_paths(video_dir):
                    video_dirs.append(video_dir)

        return video_dirs

    def output_dir_for(self, video_dir: Path) -> Path:
        relative = video_dir.relative_to(self.input_root)
        return self.output_root / relative

    def output_path_for_image(self, image_path: Path) -> Path:
        relative = image_path.relative_to(self.input_root)
        return (self.output_root / relative).with_suffix(self.output_ext)

    def process_video_dir(self, video_dir: Path) -> int:
        image_paths = list_image_paths(video_dir)

        if not image_paths:
            self.logger.warning("No keyframes found: %s", video_dir)
            return 0

        pending_paths = []

        for image_path in image_paths:
            output_path = self.output_path_for_image(image_path)

            if output_path.exists() and not self.overwrite:
                continue

            pending_paths.append(image_path)

        if not pending_paths:
            self.logger.info("Skip existing: %s", video_dir)
            return 0

        output_dir = self.output_dir_for(video_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        embeddings, valid_paths = encode_keyframe_images(
            model=self.model,
            preprocess=self.preprocess,
            device=self.device,
            image_paths=pending_paths,
            batch_size=self.batch_size,
            precision=self.precision,
            normalize=self.normalize,
            logger=self.logger,
        )

        if embeddings.size == 0:
            self.logger.warning("No embeddings generated: %s", video_dir)
            return 0

        saved = 0

        for image_path, embedding in zip(valid_paths, embeddings):
            output_path = self.output_path_for_image(image_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, embedding.astype(np.float32))
            saved += 1

        self.logger.info("Saved %d embeddings: %s", saved, output_dir)
        return saved

    def run(self):
        video_dirs = self.scan_video_dirs()
        self.logger.info("Found %d video keyframe folders", len(video_dirs))

        total_saved = 0

        for video_dir in tqdm(video_dirs, desc="Videos"):
            try:
                total_saved += self.process_video_dir(video_dir)
            except Exception as e:
                self.logger.error("Failed %s: %s", video_dir, e)

        self.logger.info("Total saved embeddings: %d", total_saved)