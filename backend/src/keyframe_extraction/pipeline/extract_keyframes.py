from __future__ import annotations

from pathlib import Path
import logging
import time

import numpy as np

from src.utils.config import AppConfig
from src.keyframe_extraction.models.export import save_keyframe_map
from src.embedding_extraction.models.encoder import encode_video_frames, load_clip_model
from src.keyframe_extraction.models.omni_detector import detect_omni_scenes
from src.keyframe_extraction.model_cache import KeyframeModels
from src.utils.logger import setup_logger
from src.utils.seed import seed_everything
from src.utils.video_io import list_videos


class KeyframeExtractionPipeline:
    def __init__(self, cfg: AppConfig, logger: logging.Logger | None = None,
                 models: KeyframeModels | None = None):
        self.cfg = cfg
        self.models = models if models is not None else KeyframeModels()
        self._owns_models = models is None
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

    def run(self, videos: list[Path] | None = None, group: Path | None = None) -> None:
        try:
            self._run(videos, group)
        finally:
            if self._owns_models:
                self.models.close()

    def _run(self, videos: list[Path] | None, group: Path | None) -> None:
        sample_fps = self.cfg.embedding.sample_fps
        if (not np.isfinite(self.cfg.keyframe.final_similarity_threshold)
                or not 0 <= self.cfg.keyframe.final_similarity_threshold <= 1):
            raise ValueError('keyframe.final_similarity_threshold must be in [0,1]')
        selection_policy = dict(method='p3_timestamp_quality_v2', budget=None,
                                shot_mode='clean_shot',
                                dedup_scope='adjacent_clean_shots_only',
                                final_similarity_threshold=self.cfg.keyframe.final_similarity_threshold,
                                max_scene_seconds=self.cfg.keyframe.max_scene_seconds,
                                min_hist_bins=self.cfg.keyframe.min_hist_bins,
                                image_quality=self.cfg.keyframe.image_quality)
        if sample_fps is None or not np.isfinite(sample_fps) or sample_fps <= 0:
            raise ValueError('VLM keyframes requires positive embedding.sample_fps')
        if self.cfg.frame_loader.backend != 'nvdec':
            raise ValueError('VLM keyframes requires the NVDEC backend')
        if self.cfg.embedding.model_name not in {'PE-Core-L14-336', 'facebook/PE-Core-L14-336'}:
            raise ValueError('VLM keyframes supports PE-Core-L14-336 only')
        if (self.cfg.frame_loader.image_size or 336) != 336:
            raise ValueError('PE-Core-L14-336 requires 336px preprocessing')
        seed_everything(self.cfg.project.seed)
        videos = videos if videos is not None else list_videos(self.cfg.paths.input_dir)
        if not videos:
            raise FileNotFoundError(f"No videos found in {self.cfg.paths.input_dir}")

        self.logger.info("Input directory: %s", self.cfg.paths.input_dir)
        self.logger.info("Output directory: %s", self.cfg.paths.output_dir)
        self.logger.info("Found %d video(s)", len(videos))

        started = time.time()
        for idx, raw_video in enumerate(videos, start=1):
            video_started = time.time()
            self.logger.info("[%d/%d] Processing %s", idx, len(videos), raw_video)

            video = raw_video  # Never transcode/overwrite the source video.
            video_group = group if group is not None else self._relative_group(video)
            if video_group.is_absolute() or '..' in video_group.parts:
                raise ValueError('Video group must be a relative path inside output')
            stem = video.stem

            scene_path = self.scenes_dir / video_group / f"{stem}.scenes.txt"
            feature_path = self.features_dir / video_group / f"{stem}.pkl"
            map_path = self.maps_dir / video_group / f"{stem}.csv"
            image_dir = self.images_dir / video_group / stem

            raw_scene_path = self.output_dir / '.cache' / 'omni_clean_shot_timeline_v2' / video_group / f'{stem}.scenes.txt'
            if raw_scene_path.exists() and self.cfg.omni.skip_existing:
                self.logger.info("Reuse scenes: %s", scene_path)
                scenes = self._load_scenes(raw_scene_path)
            else:
                scenes = detect_omni_scenes(video, raw_scene_path, self.cfg.omni,
                                            self.cfg.frame_loader.backend, model_cache=self.models)

            reuse_features = feature_path.exists() and self.cfg.embedding.skip_existing
            from src.utils.frame_sampling import (probe_video_timeline, build_sample_plan,
                                                  load_quality_sampled_cache)
            self.logger.info('Reading packet PTS/flags (demux only; no CPU frame decode)')
            timeline, source_fps = probe_video_timeline(video)
            plan = build_sample_plan(
                scenes,timeline,source_fps,sample_fps,self.cfg.keyframe.max_scene_seconds,
                self.cfg.keyframe.max_scene_gap_frames,
            )
            del timeline
            self.logger.info('Quality + embedding plan: %d/%d frames at %.3g FPS in clean shots',
                             len(plan['frame_indexes']),plan['source_total_frames'],sample_fps)
            cached = load_quality_sampled_cache(
                feature_path, plan, scenes, self.cfg.keyframe.min_hist_bins,
            ) if reuse_features else None
            reuse_features = cached is not None
            if reuse_features:
                plan = cached
                scenes = cached['kept_scenes']
            if (reuse_features and map_path.exists() and self.cfg.keyframe.skip_existing
                    and scene_path.is_file()
                    and cached.get('selection_policy') == selection_policy):
                self.logger.info('Skip completed extraction with matching sampled cache: %s',map_path)
                del cached,plan
                continue
            if feature_path.exists() and not reuse_features:
                self.logger.info('Rebuild incompatible/dense embedding cache for timestamp sampling')

            if reuse_features:
                self.logger.info("Reuse embeddings: %s", feature_path)
                features = cached['features']
            else:
                clip_model, preprocess, clip_device = self.models.embedding(
                    self.cfg.embedding, load_clip_model, self.logger)
                try:
                    options = dict(sample_plan=plan,decode_batch_size=self.cfg.omni.decode_batch_size,
                                   quality_scenes=scenes,min_hist_bins=self.cfg.keyframe.min_hist_bins)
                    encoded = encode_video_frames(
                        clip_model, preprocess, clip_device, video, feature_path,
                        self.cfg.embedding.batch_size, self.cfg.frame_loader, self.logger,
                        **options,
                    )
                    features = encoded['features']
                    cached = encoded
                    plan = encoded
                    scenes = encoded['kept_scenes']
                    del encoded
                finally:
                    del clip_model, preprocess

            if features.ndim != 2 or not len(features):
                raise ValueError('Invalid embedding cache')
            if features.shape[1] != 1024:
                raise ValueError('Expected 1024D PE-Core-L cache; remove/rebuild incompatible embeddings')
            for start in range(0,len(features),8192):
                if not np.isfinite(features[start:start+8192]).all():
                    raise ValueError('Nonfinite embedding cache')
            source_total = plan['source_total_frames']
            if not len(scenes) or np.any(scenes < 0) or np.any(scenes[:,1] >= source_total) or np.any(scenes[:,0] > scenes[:,1]):
                raise ValueError('Scene ranges do not match embeddings')

            map_path.unlink(missing_ok=True)
            if image_dir.exists():
                for image in image_dir.glob('[0-9][0-9][0-9][0-9][0-9][0-9].jpg'):
                    image.unlink()

            from src.keyframe_extraction.models.export import export_selected_cuda
            from src.keyframe_extraction.models.p3 import select_sampled_p3, cosine_dedup

            primary, diagnostics = select_sampled_p3(plan, features, cached['quality'])
            indexes = cosine_dedup(primary, scenes, self.cfg.keyframe.final_similarity_threshold)
            self.logger.info('P3: %d candidates -> %d primary -> %d deduplicated; %s',
                             len(features), len(primary), len(indexes), diagnostics)
            export_selected_cuda(video, indexes, image_dir, self.cfg.keyframe.image_quality,
                                 self.cfg.omni.decode_batch_size, self.cfg.keyframe.save_images)
            final_scenes = np.asarray([shot for shot in scenes
                                       if any(int(shot[0]) <= fid <= int(shot[1]) for fid in indexes)],
                                      dtype=np.int64).reshape(-1, 2)
            if not len(final_scenes):
                raise RuntimeError('No final shots/keyframes remain after selection')
            scene_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = scene_path.with_suffix(scene_path.suffix + '.partial')
            try:
                np.savetxt(temporary, final_scenes, fmt='%d')
                temporary.replace(scene_path)
            finally:
                temporary.unlink(missing_ok=True)
            # Publish the map only after images have been written successfully.
            positions = np.searchsorted(plan['frame_indexes'],indexes)
            if (np.any(positions >= len(plan['frame_indexes']))
                    or not np.array_equal(plan['frame_indexes'][positions],indexes)):
                raise RuntimeError('Selected keyframes are not members of the sampling plan')
            from src.utils.frame_sampling import save_embedding_cache
            cached['selection_policy'] = selection_policy
            save_embedding_cache(cached,feature_path)
            save_keyframe_map(indexes,video,map_path,fps=plan['source_fps'],timestamps=plan['timestamps'][positions])
            self.logger.info("Saved keyframe map: %s (%d images)", map_path,len(indexes))
            del features
            del cached,plan

            self.logger.info("Finished %s in %.2fs", raw_video.name, time.time() - video_started)

        self.logger.info("Pipeline finished in %.2fs", time.time() - started)
