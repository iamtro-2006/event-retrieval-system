from __future__ import annotations

from pathlib import Path
import logging

import numpy as np
from tqdm import tqdm

from src.embedding_extraction.models.embedder import (
    load_embedding_model,
    list_image_paths,
    encode_keyframe_images,
)


class ExtractEmbeddingPipeline:
    """Runs embedding extraction for ONE model config.

    `cfg["model"]` holds the config for that single model (name, backend,
    pretrained, batch_size, ...). To extract embeddings for several models
    (e.g. SigLIP2 + DFN + LongCLIP-L + BLIP-2 + BEiT-3) in one go, see
    `run_multi_model()` at the bottom of this file / `scripts/embedding_extraction/run.py`,
    which just constructs one `ExtractEmbeddingPipeline` per entry in
    `cfg["models"]` and writes each one to its own output subfolder.
    """

    def __init__(self, cfg: dict, logger: logging.Logger | None = None):
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)

        model_cfg = cfg["model"]

        self.input_root = Path(cfg["input_keyframes_root"])
        self.output_root = Path(cfg["output_embeddings_root"])
        # When several models share one config (cfg["models"]: [...]), each
        # model's embeddings are written under its own subfolder so they
        # never collide, keyed by "key" (falls back to the model name).
        output_subdir = model_cfg.get("key") or model_cfg["name"]
        self.output_root = self.output_root / output_subdir

        self.batch_size = int(model_cfg["batch_size"])
        self.normalize = bool(model_cfg.get("normalize", True))
        self.overwrite = bool(cfg["save"].get("overwrite", True))
        self.output_ext = str(cfg["save"].get("extension", ".npy"))

        # Backend-specific extras (e.g. `pooling` for blip2, `beit3` dict
        # for beit3) are passed straight through without needing this
        # pipeline class to know about them.
        known_keys = {
            "key", "name", "backend", "pretrained", "device", "precision",
            "batch_size", "normalize",
        }
        backend_extra = {k: v for k, v in model_cfg.items() if k not in known_keys}

        self.model_key = output_subdir
        self.model, self.preprocess, self.device, self.precision = load_embedding_model(
            model_name=model_cfg["name"],
            backend=model_cfg.get("backend"),
            pretrained=model_cfg.get("pretrained"),
            precision=model_cfg.get("precision", "fp32"),
            device_name=model_cfg.get("device", "auto"),
            logger=self.logger,
            **backend_extra,
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

        for video_dir in tqdm(video_dirs, desc=f"Videos [{self.model_key}]"):
            try:
                total_saved += self.process_video_dir(video_dir)
            except Exception as e:
                self.logger.error("Failed %s: %s", video_dir, e)

        self.logger.info("Total saved embeddings: %d", total_saved)


def iter_model_configs(cfg: dict, selected: list[str] | None = None) -> list[dict]:
    """Normalize config to a list of per-model configs.

    Supports both the original single-model schema (`cfg["model"]: {...}`)
    and the multi-model schema (`cfg["models"]: [{...}, {...}]`), so
    existing single-model YAML files keep working unchanged.

    `selected`: optional list of model "key" or "name" values (from
    `--model` on the CLI). When given, ONLY those models run - matched
    against `key` first, then `name` - and the `enabled` flag in the
    config is ignored (explicitly asking for a model always runs it).
    When omitted, every model with `enabled: true` (default) runs.
    """
    if "models" in cfg:
        models = cfg["models"]
    elif "model" in cfg:
        models = [cfg["model"]]
    else:
        raise KeyError("Config must define either 'model' (single) or 'models' (list).")

    if selected:
        wanted = set(selected)
        chosen = [m for m in models if m.get("key") in wanted or m.get("name") in wanted]
        found = {m.get("key") or m.get("name") for m in chosen}
        missing = wanted - found
        if missing:
            available = sorted({m.get("key") or m.get("name") for m in models})
            raise ValueError(
                f"Model(s) not found in config: {sorted(missing)}. "
                f"Available: {available}"
            )
        return chosen

    return [m for m in models if m.get("enabled", True)]


def run_multi_model(
    cfg: dict,
    logger: logging.Logger | None = None,
    models: list[str] | None = None,
) -> None:
    """Extract embeddings for the selected model(s).

    - `models=None` (default): run every model with `enabled: true` in the config.
    - `models=["siglip2_so400m_384"]` (or any `key`/`name` from the config,
      e.g. from `--model` on the CLI): run only that model, ignoring `enabled`.
    """
    logger = logger or logging.getLogger(__name__)
    base_cfg = {k: v for k, v in cfg.items() if k not in ("model", "models")}

    for model_cfg in iter_model_configs(cfg, selected=models):
        model_key = model_cfg.get("key") or model_cfg["name"]
        logger.info("=== Running embedding extraction for model: %s ===", model_key)
        try:
            pipeline = ExtractEmbeddingPipeline(
                cfg={**base_cfg, "model": model_cfg}, logger=logger
            )
            pipeline.run()
        except Exception as e:
            logger.error("Model '%s' failed, skipping: %s", model_key, e)
