from __future__ import annotations

from pathlib import Path
import logging
import time

import numpy as np

from src.utils.config import AppConfig
from src.core.selector import extract_keyframe_indexes, save_keyframe_images, save_keyframe_map
from src.models.encoder import encode_video_frames, load_clip_model, load_embeddings
from src.models.detector import detect_scenes, load_transnet
from src.utils.logger import setup_logger
from src.utils.seed import seed_everything
from src.utils.video_io import ensure_h264, list_videos


class KeyframeExtractionPipeline:
    def __init__(self, cfg: AppConfig, logger: logging.Logger | None = None):
        self.cfg = cfg
        self.logger = logger or setup_logger(
            cfg.project.name,
            cfg.logging.level,
            cfg.logging.log_dir,
            cfg.logging.log_to_file,
            cfg.logging.filename,
        )
        self.output_dir = cfg.paths.output_dir
        self.scenes_dir = self.output_dir / "scenes"
        self.features_dir = self.output_dir / "embeddings"
        self.maps_dir = self.output_dir / "map_keyframes"
        self.images_dir = self.output_dir / "keyframes"

    def _relative_group(self, video_path: Path) -> Path:
        try:
            rel = video_path.parent.relative_to(self.cfg.paths.input_dir)
            return rel if str(rel) != "." else Path("root")
        except ValueError:
            return Path("root")

    def _load_scenes(self, scene_path: Path) -> np.ndarray:
        data = np.loadtxt(scene_path, dtype=np.int32)
        return np.asarray(data, dtype=np.int32).reshape(-1, 2)

    def run(self) -> None:
        seed_everything(self.cfg.project.seed)
        videos = list_videos(self.cfg.paths.input_dir)
        if not videos:
            raise FileNotFoundError(f"No videos found in {self.cfg.paths.input_dir}")

        self.logger.info("Input directory: %s", self.cfg.paths.input_dir)
        self.logger.info("Output directory: %s", self.cfg.paths.output_dir)
        self.logger.info("Found %d video(s)", len(videos))

        transnet_model, transnet_device = load_transnet(
            self.cfg.transnet.repo_dir,
            self.cfg.transnet.weights_path,
            self.cfg.transnet.device,
            self.logger,
        )

        clip_model, preprocess, clip_device = load_clip_model(
            self.cfg.embedding.model_name,
            self.cfg.embedding.pretrained,
            self.cfg.embedding.precision,
            self.cfg.embedding.device,
            self.logger,
        )

        started = time.time()
        for idx, raw_video in enumerate(videos, start=1):
            video_started = time.time()
            self.logger.info("[%d/%d] Processing %s", idx, len(videos), raw_video)

            video = ensure_h264(raw_video, self.logger)
            group = self._relative_group(video)
            stem = video.stem

            scene_path = self.scenes_dir / group / f"{stem}.scenes.txt"
            feature_path = self.features_dir / group / f"{stem}.pkl"
            map_path = self.maps_dir / group / f"{stem}.csv"
            image_dir = self.images_dir / group / stem

            if scene_path.exists() and self.cfg.transnet.skip_existing:
                self.logger.info("Reuse scenes: %s", scene_path)
                scenes = self._load_scenes(scene_path)
            else:
                scenes = detect_scenes(
                    transnet_model,
                    transnet_device,
                    video,
                    self.cfg.transnet.batch_size,
                    self.cfg.transnet.threshold,
                    scene_path,
                    self.logger,
                )

            if feature_path.exists() and self.cfg.embedding.skip_existing:
                self.logger.info("Reuse embeddings: %s", feature_path)
                features = load_embeddings(feature_path)
            else:
                features = encode_video_frames(
                    clip_model,
                    preprocess,
                    clip_device,
                    video,
                    feature_path,
                    self.cfg.embedding.batch_size,
                    self.cfg.frame_loader,
                    self.logger,
                )

            if map_path.exists() and self.cfg.keyframe.skip_existing:
                self.logger.info("Skip existing keyframe map: %s", map_path)
                continue

            indexes = extract_keyframe_indexes(
                video,
                scenes,
                features,
                self.cfg.keyframe.min_scene_frames,
                self.cfg.keyframe.max_scene_gap_frames,
                self.cfg.keyframe.hist_threshold,
                self.cfg.keyframe.min_hist_bins,
                self.logger,
            )
            save_keyframe_map(indexes, video, map_path)
            self.logger.info("Saved keyframe map: %s", map_path)

            if self.cfg.keyframe.save_images:
                save_keyframe_images(indexes, video, image_dir, self.cfg.keyframe.image_quality)
                self.logger.info("Saved keyframe images: %s", image_dir)

            self.logger.info("Finished %s in %.2fs", raw_video.name, time.time() - video_started)

        self.logger.info("Pipeline finished in %.2fs", time.time() - started)
