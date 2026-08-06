from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import gc
import json
import logging
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.embedding_extraction.models.encoder import encode_video_frames, load_clip_model, load_embeddings
from src.keyframe_extraction.models.candidate_sampler import sample_candidates
from src.keyframe_extraction.models.deduplicator import deduplicate
from src.keyframe_extraction.models.detector import detect_scenes, load_transnet, repair_and_split_scenes
from src.keyframe_extraction.models.p3_selector import select_p3
from src.keyframe_extraction.models.quality_filter import evaluate_quality
from src.keyframe_extraction.models.selector import (
    extract_keyframe_indexes,
    save_keyframe_images,
    save_keyframe_map,
)
from src.keyframe_extraction.schemas import Candidate, DedupRecord
from src.utils.config import AppConfig
from src.utils.logger import setup_logger
from src.utils.seed import seed_everything
from src.utils.video_io import ensure_h264, get_video_fps, iter_resized_video_frames, list_videos


class KeyframeExtractionPipeline:
    STRATEGIES = {"legacy_lmske", "p3"}

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
        self.cache_dir = cfg.paths.cache_dir
        self.scenes_dir = self.cache_dir / "scenes"
        self.features_dir = self.cache_dir / "legacy_embeddings"
        self.selection_features_dir = self.cache_dir / "selection_embeddings"
        self.manifests_dir = self.cache_dir / "manifests"
        self.maps_dir = self.output_dir / "map_keyframes"
        self.images_dir = self.output_dir / "keyframes"
        self.diagnostics_dir = self.cache_dir / "diagnostics"
        if cfg.keyframe.strategy not in self.STRATEGIES:
            raise ValueError(
                f"Unsupported keyframe strategy {cfg.keyframe.strategy!r}; expected one of {sorted(self.STRATEGIES)}"
            )

    def _relative_group(self, video_path: Path) -> Path:
        try:
            rel = video_path.parent.relative_to(self.cfg.paths.input_dir)
            return rel if str(rel) != "." else Path("root")
        except ValueError:
            return Path("root")

    def _diagnostic_directory(self, group: Path, video_stem: str) -> Path:
        return self.diagnostics_dir / group / video_stem

    def _manifest_path(self, group: Path, video_stem: str) -> Path:
        return self.manifests_dir / group / f"{video_stem}.json"

    def _existing_strategy(self, manifest_path: Path) -> str | None:
        config_path = manifest_path
        if not config_path.exists():
            return None
        try:
            return str(json.loads(config_path.read_text(encoding="utf-8")).get("strategy"))
        except (OSError, ValueError, TypeError):
            return None

    def _should_skip_existing(self, map_path: Path, manifest_path: Path) -> bool:
        if not map_path.exists() or not self.cfg.keyframe.skip_existing:
            return False
        existing_strategy = self._existing_strategy(manifest_path)
        if self.cfg.keyframe.strategy == "p3":
            return existing_strategy == "p3"
        return existing_strategy in {None, "legacy_lmske"}

    @staticmethod
    def _write_strategy_manifest(manifest_path: Path, strategy: str) -> None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps({"strategy": strategy}, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @staticmethod
    def _load_scenes(scene_path: Path) -> np.ndarray:
        data = np.loadtxt(scene_path, dtype=np.int32)
        return np.asarray(data, dtype=np.int32).reshape(-1, 2)

    def _load_or_detect_scenes(self, video: Path, scene_path: Path, model, device) -> np.ndarray:
        if scene_path.exists() and self.cfg.transnet.skip_existing:
            self.logger.info("Reuse scenes: %s", scene_path)
            return self._load_scenes(scene_path)
        return detect_scenes(
            model,
            device,
            video,
            self.cfg.transnet.batch_size,
            self.cfg.transnet.threshold,
            scene_path,
            self.logger,
            merge_transition_runs=self.cfg.keyframe.strategy == "p3",
        )

    def _load_legacy_features(self, video: Path, group: Path, clip_bundle) -> np.ndarray:
        model, preprocess, device = clip_bundle
        feature_path = self.features_dir / group / f"{video.stem}.pkl"
        if feature_path.exists() and self.cfg.embedding.skip_existing:
            self.logger.info("Reuse embeddings: %s", feature_path)
            features = load_embeddings(feature_path)
        else:
            features = encode_video_frames(
                model,
                preprocess,
                device,
                video,
                feature_path,
                self.cfg.embedding.batch_size,
                self.cfg.frame_loader,
                self.logger,
            )
        return features

    def _select_legacy(self, video: Path, scenes: np.ndarray, features: np.ndarray) -> list[int]:
        return extract_keyframe_indexes(
            video,
            scenes,
            features,
            self.cfg.keyframe.min_scene_frames,
            self.cfg.keyframe.max_scene_gap_frames,
            self.cfg.keyframe.hist_threshold,
            self.cfg.keyframe.min_hist_bins,
            self.logger,
        )

    def _encode_candidates(
        self,
        candidates: list[Candidate],
        images: dict[int, np.ndarray],
        cache_path: Path,
        clip_bundle,
    ) -> None:
        model, preprocess, device = clip_bundle
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        mapping: dict[int, np.ndarray] = {}
        if cache_path.exists() and self.cfg.keyframe.selection_embedding.skip_existing:
            cached = np.load(cache_path)
            mapping = {
                int(frame_idx): np.asarray(feature, dtype=np.float32)
                for frame_idx, feature in zip(cached["frame_indexes"], cached["features"])
            }
        missing = [candidate for candidate in candidates if candidate.frame_idx not in mapping]
        batch_size = self.cfg.keyframe.selection_embedding.batch_size
        use_amp = device.type == "cuda"
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=use_amp
        ):
            for start in range(0, len(missing), batch_size):
                chunk = missing[start : start + batch_size]
                readable = [candidate for candidate in chunk if candidate.frame_idx in images]
                if not readable:
                    continue
                batch = torch.stack(
                    [preprocess(Image.fromarray(images[candidate.frame_idx])) for candidate in readable]
                ).to(device, non_blocking=True)
                features = model.encode_image(batch)
                features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                for candidate, feature in zip(readable, features.float().cpu().numpy()):
                    mapping[candidate.frame_idx] = feature.astype(np.float32)
        if mapping:
            frame_indexes = np.asarray(sorted(mapping), dtype=np.int32)
            np.savez_compressed(
                cache_path,
                frame_indexes=frame_indexes,
                features=np.stack([mapping[index] for index in frame_indexes]),
            )
        for candidate in candidates:
            candidate.feature = mapping.get(candidate.frame_idx)

    def _write_diagnostics(
        self,
        directory: Path,
        video: Path,
        fps: float,
        scenes: np.ndarray,
        candidates: list[Candidate],
        selected: list[Candidate],
        dropped: list[DedupRecord],
        shot_diagnostics: dict[int, dict[str, object]],
        metrics: dict[str, object],
    ) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        candidate_rows = [
            {
                "candidate_id": index,
                "video_id": video.stem,
                "shot_id": candidate.shot_id,
                "frame_idx": candidate.frame_idx,
                "timestamp_sec": candidate.timestamp_sec,
                "source": candidate.source,
                "quality": candidate.quality,
                "valid": int(candidate.valid),
                "rejection_reason": candidate.rejection_reason,
                "blur": candidate.blur,
                "brightness": candidate.brightness,
                "entropy": candidate.entropy,
                "edge_density": candidate.edge_density,
                "clipped_fraction": candidate.clipped_fraction,
            }
            for index, candidate in enumerate(candidates)
        ]
        shot_rows = []
        for shot_id, (start, end) in enumerate(scenes.tolist()):
            diagnostic = shot_diagnostics.get(shot_id, {})
            shot_rows.append(
                {
                    "video_id": video.stem,
                    "shot_id": shot_id,
                    "start_frame": start,
                    "end_frame": end,
                    "duration_sec": (end - start + 1) / fps,
                    "candidate_count": sum(candidate.shot_id == shot_id for candidate in candidates),
                    "valid_candidate_count": sum(
                        candidate.shot_id == shot_id and candidate.valid for candidate in candidates
                    ),
                    **diagnostic,
                }
            )
        selected_rows = [
            {
                "keyframe_id": index,
                "video_id": video.stem,
                "frame_idx": candidate.frame_idx,
                "timestamp_sec": candidate.timestamp_sec,
                "fps": fps,
                "shot_id": candidate.shot_id,
                "selection_source": candidate.selection_source,
                "quality": candidate.quality,
                "cluster_id": candidate.cluster_id,
                "cluster_count": candidate.cluster_count,
                "cluster_aucc": candidate.cluster_aucc,
                "representativeness": candidate.representativeness,
                "typicality": candidate.typicality,
                "volatility": candidate.volatility,
                "unique_score": candidate.unique_score,
                "mmr_score": candidate.mmr_score,
            }
            for index, candidate in enumerate(selected)
        ]
        dropped_rows = [
            {
                "video_id": video.stem,
                "frame_idx": record.dropped.frame_idx,
                "timestamp_sec": record.dropped.timestamp_sec,
                "selection_source": record.dropped.selection_source,
                "duplicate_of_frame_idx": record.kept.frame_idx,
                "drop_reason": record.reason,
                "phash_hamming": record.phash_hamming,
                "dense_cosine": record.dense_cosine,
            }
            for record in dropped
        ]
        pd.DataFrame(candidate_rows, columns=[
            "candidate_id", "video_id", "shot_id", "frame_idx", "timestamp_sec", "source", "quality",
            "valid", "rejection_reason", "blur", "brightness", "entropy", "edge_density", "clipped_fraction",
        ]).to_csv(directory / "candidates.csv", index=False)
        pd.DataFrame(shot_rows).to_csv(directory / "shots.csv", index=False)
        pd.DataFrame(selected_rows, columns=[
            "keyframe_id", "video_id", "frame_idx", "timestamp_sec", "fps", "shot_id", "selection_source",
            "quality", "cluster_id", "cluster_count", "cluster_aucc", "representativeness", "typicality",
            "volatility", "unique_score", "mmr_score",
        ]).to_csv(directory / "selected_keyframes.csv", index=False)
        pd.DataFrame(dropped_rows, columns=[
            "video_id", "frame_idx", "timestamp_sec", "selection_source", "duplicate_of_frame_idx",
            "drop_reason", "phash_hamming", "dense_cosine",
        ]).to_csv(directory / "dedup_dropped.csv", index=False)
        (directory / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        run_config = {
            "strategy": "p3",
            "selection_embedding": asdict(self.cfg.keyframe.selection_embedding),
            "scene": asdict(self.cfg.keyframe.scene),
            "candidate": asdict(self.cfg.keyframe.candidate),
            "quality": asdict(self.cfg.keyframe.quality),
            "clustering": asdict(self.cfg.keyframe.clustering),
            "selector": asdict(self.cfg.keyframe.selector),
            "dedup": asdict(self.cfg.keyframe.dedup),
        }
        (directory / "run_config.json").write_text(
            json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _run_p3(self, video: Path, raw_scenes: np.ndarray, group: Path, clip_bundle) -> list[int]:
        started = time.perf_counter()
        fps = get_video_fps(video)
        scenes = repair_and_split_scenes(raw_scenes, fps, self.cfg.keyframe.scene.max_duration_sec)
        sampled_at = time.perf_counter()
        candidate_images: dict[int, np.ndarray] = {}
        candidates = sample_candidates(
            iter_resized_video_frames(video),
            scenes,
            fps,
            self.cfg.keyframe.candidate,
            observer=lambda candidate, rgb: evaluate_quality(candidate, rgb, self.cfg.keyframe.quality),
            image_cache=candidate_images,
        )
        sample_elapsed = time.perf_counter() - sampled_at

        embedding_path = self.selection_features_dir / group / f"{video.stem}.npz"
        encoded_at = time.perf_counter()
        self._encode_candidates(candidates, candidate_images, embedding_path, clip_bundle)
        embedding_elapsed = time.perf_counter() - encoded_at
        by_shot: dict[int, list[Candidate]] = {}
        for candidate in candidates:
            if candidate.feature is not None:
                by_shot.setdefault(candidate.shot_id, []).append(candidate)
        if not by_shot:
            raise RuntimeError(f"P3 produced no embeddable candidates for {video}")

        selection_at = time.perf_counter()
        primary, shot_diagnostics, selection_metrics = select_p3(
            by_shot,
            self.cfg.keyframe.clustering,
            self.cfg.keyframe.selector,
            self.cfg.keyframe.dedup,
        )
        selected_images = {
            candidate.frame_idx: candidate_images[candidate.frame_idx]
            for candidate in primary
            if candidate.frame_idx in candidate_images
        }
        primary = [candidate for candidate in primary if candidate.frame_idx in selected_images]
        selected, dropped = deduplicate(primary, selected_images, self.cfg.keyframe.dedup)
        selection_elapsed = time.perf_counter() - selection_at
        metrics: dict[str, object] = {
            "video_id": video.stem,
            "strategy": "p3",
            "raw_scenes": len(raw_scenes),
            "repaired_shots": len(scenes),
            "total_candidates": len(candidates),
            "valid_candidates": sum(candidate.valid for candidate in candidates),
            "filtered_candidates": sum(not candidate.valid for candidate in candidates),
            "fallback_shots": 0,
            "common_anchors": sum(candidate.selection_source == "shot_common_anchor" for candidate in primary),
            "before_dedup": len(primary),
            "dedup_dropped": len(dropped),
            "final_keyframes": len(selected),
            "sample_elapsed_sec": sample_elapsed,
            "embedding_elapsed_sec": embedding_elapsed,
            "selection_elapsed_sec": selection_elapsed,
            "total_elapsed_sec": time.perf_counter() - started,
            **selection_metrics,
        }
        if self.cfg.keyframe.write_diagnostics:
            self._write_diagnostics(
                self._diagnostic_directory(group, video.stem),
                video,
                fps,
                scenes,
                candidates,
                selected,
                dropped,
                shot_diagnostics,
                metrics,
            )
        self.logger.info(
            "P3 %s raw_scenes=%d repaired_shots=%d candidates=%d valid=%d filtered=%d "
            "common=%d before_dedup=%d dropped=%d final=%d",
            video.name,
            metrics["raw_scenes"],
            metrics["repaired_shots"],
            metrics["total_candidates"],
            metrics["valid_candidates"],
            metrics["filtered_candidates"],
            metrics["common_anchors"],
            metrics["before_dedup"],
            metrics["dedup_dropped"],
            metrics["final_keyframes"],
        )
        return [candidate.frame_idx for candidate in selected]

    @staticmethod
    def _filter_videos(
        videos: list[Path],
        input_dir: Path,
        video_ids: list[str] | None,
        groups: list[str] | None,
    ) -> list[Path]:
        selected = videos
        if groups:
            requested_groups = {group.casefold(): group for group in groups}

            def group_name(video: Path) -> str:
                relative = video.relative_to(input_dir)
                return relative.parts[0] if len(relative.parts) > 1 else "root"

            selected = [video for video in selected if group_name(video).casefold() in requested_groups]
            found_groups = {group_name(video).casefold() for video in selected}
            missing_groups = [
                original for normalized, original in requested_groups.items() if normalized not in found_groups
            ]
            if missing_groups:
                raise FileNotFoundError(f"Requested group(s) not found: {', '.join(missing_groups)}")
        if video_ids:
            requested_ids = {video_id.casefold(): video_id for video_id in video_ids}
            selected = [video for video in selected if video.stem.casefold() in requested_ids]
            found_ids = {video.stem.casefold() for video in selected}
            missing_ids = [
                original for normalized, original in requested_ids.items() if normalized not in found_ids
            ]
            if missing_ids:
                raise FileNotFoundError(f"Requested video ID(s) not found: {', '.join(missing_ids)}")
        return selected

    def run(self, video_ids: list[str] | None = None, groups: list[str] | None = None) -> None:
        seed_everything(self.cfg.project.seed)
        videos = list_videos(self.cfg.paths.input_dir)
        if not videos:
            raise FileNotFoundError(f"No videos found in {self.cfg.paths.input_dir}")
        videos = self._filter_videos(videos, self.cfg.paths.input_dir, video_ids, groups)
        self.logger.info("Strategy: %s", self.cfg.keyframe.strategy)
        self.logger.info("Input directory: %s", self.cfg.paths.input_dir)
        self.logger.info("Output directory: %s", self.cfg.paths.output_dir)
        self.logger.info("Found %d video(s)", len(videos))

        transnet_model, transnet_device = load_transnet(
            self.cfg.transnet.repo_dir,
            self.cfg.transnet.weights_path,
            self.cfg.transnet.device,
            self.logger,
        )
        embedding_cfg = (
            self.cfg.embedding
            if self.cfg.keyframe.strategy == "legacy_lmske"
            else self.cfg.keyframe.selection_embedding
        )
        clip_bundle = load_clip_model(
            embedding_cfg.model_name,
            embedding_cfg.pretrained,
            embedding_cfg.precision,
            embedding_cfg.device,
            self.logger,
        )

        started = time.time()
        for index, raw_video in enumerate(videos, start=1):
            video_started = time.time()
            self.logger.info("[%d/%d] Processing %s", index, len(videos), raw_video)
            video = ensure_h264(raw_video, self.logger)
            group = self._relative_group(video)
            scene_suffix = ".p3.scenes.txt" if self.cfg.keyframe.strategy == "p3" else ".scenes.txt"
            scene_path = self.scenes_dir / group / f"{video.stem}{scene_suffix}"
            map_path = self.maps_dir / group / f"{video.stem}.csv"
            image_dir = self.images_dir / group / video.stem
            manifest_path = self._manifest_path(group, video.stem)
            scenes = self._load_or_detect_scenes(video, scene_path, transnet_model, transnet_device)
            if self.cfg.keyframe.strategy == "legacy_lmske":
                features = self._load_legacy_features(video, group, clip_bundle)
                if self._should_skip_existing(map_path, manifest_path):
                    self.logger.info("Skip existing keyframe map: %s", map_path)
                    continue
                indexes = self._select_legacy(video, scenes, features)
            else:
                if self._should_skip_existing(map_path, manifest_path):
                    self.logger.info("Skip existing keyframe map: %s", map_path)
                    continue
                indexes = self._run_p3(video, scenes, group, clip_bundle)
            save_keyframe_map(indexes, video, map_path)
            self.logger.info("Saved keyframe map: %s", map_path)
            if self.cfg.keyframe.save_images:
                save_keyframe_images(indexes, video, image_dir, self.cfg.keyframe.image_quality)
                self.logger.info("Saved keyframe images: %s", image_dir)
            self._write_strategy_manifest(manifest_path, self.cfg.keyframe.strategy)
            self.logger.info("Finished %s in %.2fs", raw_video.name, time.time() - video_started)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.logger.info("Pipeline finished in %.2fs", time.time() - started)
