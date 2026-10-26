from __future__ import annotations

import json
import logging
import math
from pathlib import Path

from faster_whisper import WhisperModel


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm"}


class ExtractASRPipeline:
    """Transcribe videos into the JSON schema consumed by the ASR indexer."""

    def __init__(
        self,
        cfg: dict,
        input_dir: Path,
        output_dir: Path,
        logger: logging.Logger | None = None,
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.input_dir = input_dir
        self.output_dir = output_dir
        extraction = cfg["extraction"]
        self.language = extraction.get("language")
        self.beam_size = int(extraction.get("beam_size", 5))
        self.vad_filter = bool(extraction.get("vad_filter", True))
        self.model_name = extraction["model"]
        self.device = extraction.get("device", "cuda")
        self.compute_type = extraction.get("compute_type", "int8")
        self.model: WhisperModel | None = None

    def video_files(self, video_ids: list[str] | None = None) -> list[Path]:
        if not self.input_dir.is_dir():
            raise FileNotFoundError(f"Video directory not found: {self.input_dir}")

        files = sorted(
            path for path in self.input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        )
        by_id: dict[str, Path] = {}
        for path in files:
            if path.stem in by_id:
                raise ValueError(f"Multiple video files share ID {path.stem}: {by_id[path.stem]}, {path}")
            by_id[path.stem] = path

        if video_ids:
            missing = sorted(set(video_ids) - by_id.keys())
            if missing:
                raise FileNotFoundError(f"Video IDs not found in {self.input_dir}: {missing}")
            return [by_id[video_id] for video_id in dict.fromkeys(video_ids)]
        return files

    def process_video(self, video_path: Path, overwrite: bool = False) -> int:
        output_path = self.output_dir / f"{video_path.stem}.json"
        if output_path.exists() and not overwrite:
            self.logger.info("Skip existing: %s", output_path)
            return 0

        if self.model is None:
            self.model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
            )
        segments, _info = self.model.transcribe(
            str(video_path),
            language=self.language,
            task="transcribe",
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
        )
        records = []
        for segment in segments:  # Faster-Whisper starts inference while iterating.
            text = segment.text.strip()
            start = float(segment.start)
            end = float(segment.end)
            if not text:
                continue
            if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
                raise ValueError(f"Invalid ASR timestamps in {video_path}: {start}, {end}")
            records.append({
                "video_id": video_path.stem,
                "start": start,
                "end": end,
                "transcript": text,
            })

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(".json.tmp")
        try:
            with temporary_path.open("w", encoding="utf-8") as file:
                json.dump(records, file, ensure_ascii=False, indent=2)
            temporary_path.replace(output_path)
        finally:
            temporary_path.unlink(missing_ok=True)

        self.logger.info("Saved %d ASR segments -> %s", len(records), output_path)
        return len(records)
