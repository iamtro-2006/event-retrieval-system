from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.retrieval.indexer.elasticsearch.multimodal_text.repository import (
    MultimodalTextRepository,
)
from src.retrieval.indexer.elasticsearch.multimodal_text.schemas import (
    MultimodalTextDocument,
)


class IndexPipeline:
    """Fuse per-video OCR + ASR extraction output into keyframe-level documents.

    Reads the canonical ``metadata.csv`` (keyframe -> ``timestamp_sec``) to snap
    each ASR speech segment to the keyframes whose timestamp falls inside the
    segment's ``[start, end]`` window. A keyframe is indexed only when it
    carries at least one of ``ocr_text`` / ``asr_text`` (empty keyframes are
    skipped to keep the index lean).

    Data layout (mirrors the OCR/ASR extraction pipelines):

        ocr_root/<dataset>/<video_id>.json   -> { "<keyframe_stem>": [boxes, texts], ... }
        asr_root/<dataset>/<video_id>.json   -> [ {video_id, start, end, transcript}, ... ]
        metadata.csv                         -> dataset, video_id, keyframe_id,
                                                keyframe_id_int, source_name, timestamp_sec, ...
    """

    def __init__(
        self,
        repository: MultimodalTextRepository,
        metadata_path: str | Path,
        ocr_root: str | Path | None = None,
        asr_root: str | Path | None = None,
    ) -> None:
        self.repository = repository
        self.metadata_path = Path(metadata_path)
        self.ocr_root = Path(ocr_root) if ocr_root else None
        self.asr_root = Path(asr_root) if asr_root else None

    def create_index(self) -> None:
        self.repository.create_index()

    def run(self) -> int:
        """Build and bulk-insert multimodal-text docs, returning the total count."""
        metadata = pd.read_csv(self.metadata_path, low_memory=False)
        metadata.columns = [str(col).lstrip("\ufeff") for col in metadata.columns]
        for col in ("dataset", "video_id", "keyframe_id", "source_name"):
            if col in metadata.columns:
                metadata[col] = metadata[col].astype(str)

        total = 0
        grouped = metadata.groupby(["dataset", "video_id"], sort=False)
        for (dataset, video_id), group in grouped:
            ocr = self._load_ocr(dataset, video_id)
            asr_segments = self._load_asr(dataset, video_id)
            docs = self._build_video_docs(dataset, video_id, group, ocr, asr_segments)
            if docs:
                self.repository.bulk_insert(docs)
                print(f"[text-index] {video_id}: {len(docs)} keyframes")
            total += len(docs)
        return total

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _load_ocr(self, dataset: str, video_id: str) -> dict[str, list]:
        if self.ocr_root is None:
            return {}
        path = self.ocr_root / str(dataset) / f"{video_id}.json"
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load_asr(self, dataset: str, video_id: str) -> list[dict]:
        if self.asr_root is None:
            return []
        path = self.asr_root / str(dataset) / f"{video_id}.json"
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        segments: list[dict] = []
        for seg in data:
            segments.append(
                {
                    "start": self._safe_float(seg.get("start")),
                    "end": self._safe_float(seg.get("end")),
                    "transcript": str(seg.get("transcript", "")).strip(),
                }
            )
        return segments

    # ------------------------------------------------------------------
    # Fusion
    # ------------------------------------------------------------------

    def _build_video_docs(
        self,
        dataset: str,
        video_id: str,
        group: pd.DataFrame,
        ocr: dict[str, list],
        asr_segments: list[dict],
    ) -> list[MultimodalTextDocument]:
        docs: list[MultimodalTextDocument] = []
        for row in group.to_dict(orient="records"):
            ocr_text = self._ocr_text(row, ocr)
            timestamp_sec = self._safe_float(row.get("timestamp_sec"))
            asr_text = self._asr_text(timestamp_sec, asr_segments)

            if not ocr_text and not asr_text:
                continue

            docs.append(
                MultimodalTextDocument(
                    dataset=str(dataset),
                    video_id=str(video_id),
                    keyframe_id=str(row.get("keyframe_id", "")),
                    timestamp_sec=timestamp_sec,
                    ocr_text=ocr_text,
                    asr_text=asr_text,
                )
            )
        return docs

    @staticmethod
    def _ocr_text(row: dict, ocr: dict[str, list]) -> str:
        """Join the OCR text lines for a keyframe, matched by filename stem.

        The OCR JSON is keyed by the keyframe image filename stem. The metadata
        row stores that same stem under ``source_name`` (plus ``keyframe_id`` /
        ``keyframe_id_int`` variants), so we try a few candidate keys to absorb
        zero-padding differences.
        """
        if not ocr:
            return ""
        for key in IndexPipeline._candidate_keys(row):
            if key in ocr:
                _, texts = ocr[key]
                return " ".join(str(t) for t in texts)
        return ""

    @staticmethod
    def _candidate_keys(row: dict) -> set[str]:
        keys: set[str] = set()
        for col in ("source_name", "keyframe_id"):
            value = row.get(col)
            if value is None or str(value) in ("", "nan", "None", "<NA>"):
                continue
            keys.add(str(value).strip())
        keyframe_int = row.get("keyframe_id_int")
        if keyframe_int is not None and str(keyframe_int) not in (
            "",
            "nan",
            "None",
            "<NA>",
        ):
            keys.add(str(int(keyframe_int)))
            keys.add(f"{int(keyframe_int):06d}")
        return keys

    @staticmethod
    def _asr_text(timestamp_sec: float, segments: list[dict]) -> str:
        """Return the transcript of the segment covering ``timestamp_sec``.

        Keeps the FULL segment transcript (not trimmed to the keyframe span),
        so each keyframe inside a speech segment carries the complete sentence.
        """
        for seg in segments:
            if seg["start"] <= timestamp_sec <= seg["end"]:
                return seg["transcript"]
        return ""

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return default
        if pd.isna(result):
            return default
        return result
