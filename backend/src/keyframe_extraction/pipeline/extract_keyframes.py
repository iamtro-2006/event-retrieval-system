from __future__ import annotations

from pathlib import Path
import logging
import time
import gc

import numpy as np

from src.utils.config import AppConfig
from src.keyframe_extraction.models.selector import extract_keyframe_indexes, save_keyframe_images, save_keyframe_map
from src.embedding_extraction.models.encoder import encode_video_frames, load_clip_model, load_embeddings
from src.keyframe_extraction.models.omni_detector import detect_omni_scenes
from src.utils.logger import setup_logger
from src.utils.seed import seed_everything
from src.utils.video_io import get_video_fps, list_videos


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

    def run(self, videos: list[Path] | None = None, group: Path | None = None) -> None:
        sample_fps = self.cfg.embedding.sample_fps
        if not np.isfinite(self.cfg.keyframe.dedup_threshold) or not 0 <= self.cfg.keyframe.dedup_threshold <= 1:
            raise ValueError('keyframe.dedup_threshold must be in [0,1]')
        if (not np.isfinite(self.cfg.keyframe.final_similarity_threshold)
                or not 0 <= self.cfg.keyframe.final_similarity_threshold <= 1):
            raise ValueError('keyframe.final_similarity_threshold must be in [0,1]')
        selection_policy = dict(method='shot_mmr_fused_quality_v4',
                                budget_fraction=self.cfg.keyframe.mmr_budget_fraction,
                                anchors_per_shot=self.cfg.keyframe.anchors_per_shot,
                                relevance_weight=self.cfg.keyframe.mmr_relevance_weight,
                                final_similarity_threshold=self.cfg.keyframe.final_similarity_threshold,
                                final_dedup_strategy='neighbor_novelty_sharpness_v1',
                                min_hist_bins=self.cfg.keyframe.min_hist_bins,
                                min_scene_frames=self.cfg.keyframe.min_scene_frames,
                                image_quality=self.cfg.keyframe.image_quality)
        if sample_fps is not None and (not np.isfinite(sample_fps) or sample_fps <= 0):
            raise ValueError('embedding.sample_fps must be positive or null')
        if sample_fps is not None and self.cfg.frame_loader.backend != 'nvdec':
            raise ValueError('Timestamp sampling currently requires the NVDEC backend')
        if self.cfg.frame_loader.backend == 'nvdec':
            if self.cfg.embedding.model_name not in {'PE-Core-L14-336','facebook/PE-Core-L14-336'}:
                raise ValueError('The NVDEC VLM path currently supports PE-Core-L14-336 only')
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

            if sample_fps is None and map_path.exists() and self.cfg.keyframe.skip_existing:
                self.logger.info("Skip completed keyframe map: %s", map_path)
                continue

            if scene_path.exists() and self.cfg.omni.skip_existing:
                self.logger.info("Reuse scenes: %s", scene_path)
                scenes = self._load_scenes(scene_path)
            else:
                scenes = detect_omni_scenes(video, scene_path, self.cfg.omni, self.cfg.frame_loader.backend)

            plan, cached = None, None
            reuse_features = feature_path.exists() and self.cfg.embedding.skip_existing
            if sample_fps is not None:
                from src.utils.frame_sampling import (probe_video_timeline, build_sample_plan,
                                                      load_quality_sampled_cache,
                                                      merge_consecutive_short_scenes)
                self.logger.info('Reading source timestamps (packet demux only, no CPU frame decode)')
                timeline, source_fps = probe_video_timeline(video)
                original_shots = len(scenes)
                merged_scenes = merge_consecutive_short_scenes(scenes, timeline, source_fps)
                if not np.array_equal(merged_scenes, scenes):
                    temporary = scene_path.with_suffix(scene_path.suffix + '.partial')
                    try:
                        np.savetxt(temporary, merged_scenes, fmt='%d')
                        temporary.replace(scene_path)
                    finally:
                        temporary.unlink(missing_ok=True)
                scenes = merged_scenes
                self.logger.info('Sub-1s shot merging: %d Omni shots -> %d selection shots',
                                 original_shots, len(scenes))
                plan = build_sample_plan(
                    scenes,timeline,source_fps,sample_fps,self.cfg.keyframe.max_scene_seconds,
                    self.cfg.keyframe.max_scene_gap_frames,
                )
                del timeline
                self.logger.info('Quality + embedding plan: %d/%d frames at %.3g FPS in general shots',
                                 len(plan['frame_indexes']),plan['source_total_frames'],sample_fps)
                cached = load_quality_sampled_cache(
                    feature_path, plan, scenes, self.cfg.keyframe.min_hist_bins,
                ) if reuse_features else None
                reuse_features = cached is not None
                if reuse_features:
                    plan = cached
                    scenes = cached['kept_scenes']
                if (reuse_features and map_path.exists() and self.cfg.keyframe.skip_existing
                        and cached.get('selection_policy') == selection_policy):
                    self.logger.info('Skip completed extraction with matching sampled cache: %s',map_path)
                    del cached,plan
                    continue
                if feature_path.exists() and not reuse_features:
                    self.logger.info('Rebuild incompatible/dense embedding cache for timestamp sampling')

            if reuse_features:
                self.logger.info("Reuse embeddings: %s", feature_path)
                features = cached['features'] if plan is not None else load_embeddings(feature_path)
            else:
                clip_model, preprocess, clip_device = load_clip_model(
                    model_name=self.cfg.embedding.model_name,
                    pretrained=self.cfg.embedding.pretrained,
                    precision=self.cfg.embedding.precision,
                    device_name=self.cfg.embedding.device,
                    backend=self.cfg.embedding.backend,
                    logger=self.logger,
                )
                try:
                    options = dict(sample_plan=plan,decode_batch_size=self.cfg.omni.decode_batch_size,
                                   quality_scenes=scenes,min_hist_bins=self.cfg.keyframe.min_hist_bins) if plan is not None else {}
                    encoded = encode_video_frames(
                        clip_model, preprocess, clip_device, video, feature_path,
                        self.cfg.embedding.batch_size, self.cfg.frame_loader, self.logger,
                        **options,
                    )
                    features = encoded['features'] if plan is not None else encoded
                    if plan is not None:
                        cached = encoded
                        plan = encoded
                        scenes = encoded['kept_scenes']
                    del encoded
                finally:
                    del clip_model, preprocess
                    gc.collect()
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            if features.ndim != 2 or not len(features):
                raise ValueError('Invalid embedding cache')
            if self.cfg.embedding.model_name in {'PE-Core-L14-336','facebook/PE-Core-L14-336'} and features.shape[1] != 1024:
                raise ValueError('Expected 1024D PE-Core-L cache; remove/rebuild incompatible embeddings')
            for start in range(0,len(features),8192):
                if not np.isfinite(features[start:start+8192]).all():
                    raise ValueError('Nonfinite embedding cache')
            source_total = plan['source_total_frames'] if plan is not None else len(features)
            if not len(scenes) or np.any(scenes < 0) or np.any(scenes[:,1] >= source_total) or np.any(scenes[:,0] > scenes[:,1]):
                raise ValueError('Scene ranges do not match embeddings')

            map_path.unlink(missing_ok=True)
            if image_dir.exists():
                for image in image_dir.glob('[0-9][0-9][0-9][0-9][0-9][0-9].jpg'):
                    image.unlink()

            if self.cfg.frame_loader.backend == 'nvdec':
                from src.keyframe_extraction.models.gpu_selection import select_candidates_cuda, export_selected_cuda
                from src.keyframe_extraction.models.shot_mmr import (select_shot_mmr,
                                                                    dedup_selected_keyframes)
                fps = plan['source_fps'] if plan is not None else get_video_fps(video)
                if fps <= 0:
                    raise ValueError('Cannot determine video FPS')
                max_gap = self.cfg.keyframe.max_scene_gap_frames
                if self.cfg.keyframe.max_scene_seconds is not None:
                    if self.cfg.keyframe.max_scene_seconds <= 0:
                        raise ValueError('max_scene_seconds must be positive')
                    max_gap = max(1,round(self.cfg.keyframe.max_scene_seconds*fps)-1)
                selection_options = dict(frame_indexes=plan['frame_indexes'],sample_ranges=plan['sample_ranges']) if plan is not None else {}
                candidates = select_candidates_cuda(scenes,features,self.cfg.keyframe.min_scene_frames,max_gap,**selection_options)
                if plan is None:
                    candidate_features = features[candidates]
                else:
                    rows = np.searchsorted(plan['frame_indexes'],candidates)
                    if (np.any(rows >= len(plan['frame_indexes']))
                            or not np.array_equal(plan['frame_indexes'][rows],candidates)):
                        raise RuntimeError('Candidate frame indexes do not match sampled embeddings')
                    candidate_features = features[rows]
                indexes = select_shot_mmr(candidates,candidate_features,scenes,
                                          self.cfg.keyframe.mmr_budget_fraction,
                                          self.cfg.keyframe.anchors_per_shot,
                                          self.cfg.keyframe.mmr_relevance_weight)
                self.logger.info('Shot MMR: %d candidates -> %d keyframes',len(candidates),len(indexes))
                selected_rows = np.searchsorted(candidates, indexes)
                if plan is not None:
                    sample_rows = np.searchsorted(plan['frame_indexes'], indexes)
                    sharpness = cached['sharpness'][sample_rows]
                else:
                    sharpness = np.zeros(len(indexes), dtype=np.float32)
                before_dedup = len(indexes)
                indexes = dedup_selected_keyframes(
                    indexes, candidate_features[selected_rows], sharpness,
                    self.cfg.keyframe.final_similarity_threshold, device='cuda',
                )
                self.logger.info('Final cosine dedup (>%.2f, local novelty + sharpness): %d -> %d keyframes',
                                 self.cfg.keyframe.final_similarity_threshold,before_dedup,len(indexes))
                export_selected_cuda(video,indexes,image_dir,self.cfg.keyframe.image_quality,
                                     self.cfg.omni.decode_batch_size,self.cfg.keyframe.save_images)
            else:
                indexes = extract_keyframe_indexes(
                    video, scenes, features, self.cfg.keyframe.min_scene_frames,
                    self.cfg.keyframe.max_scene_gap_frames, self.cfg.keyframe.dedup_threshold,
                    self.cfg.keyframe.min_hist_bins, self.logger, self.cfg.keyframe.max_scene_seconds,
                )
                if self.cfg.keyframe.save_images:
                    save_keyframe_images(indexes, video, image_dir, self.cfg.keyframe.image_quality)
            # Publish the map only after images have been written successfully.
            if plan is None:
                save_keyframe_map(indexes,video,map_path)
            else:
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
