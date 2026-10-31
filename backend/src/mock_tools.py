"""Dataset-isolated mock retrieval and one tool registry for HTTP and Codex.

No model, FAISS, Elasticsearch, or external image service is used. Scores are
synthetic; playback uses CSV timing when available and images come from the dataset.
"""
from __future__ import annotations

import base64
import csv
import io
import random
import re
from pathlib import Path
from typing import Literal
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field
from src.api.legacy.video_keyframes import finite_number

DATASETS = ("v3c", "lhe")
MODEL = "qwen3-vl-embedding-2b"
MODES = ("semantic", "temporal", "ocr", "asr", "caption", "hierarchical", "auto", "fusion", "color", "similarity")


class ToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CapabilitiesArgs(ToolArgs):
    pass


class SearchArgs(ToolArgs):
    query: str = Field(min_length=1, max_length=4000)
    mode: Literal["semantic", "temporal", "ocr", "asr", "caption", "hierarchical", "auto", "fusion", "color", "similarity"] = "semantic"
    top_k: int = Field(default=12, ge=1, le=100)


class InspectArgs(ToolArgs):
    frame_ids: list[str] = Field(min_length=1, max_length=6, description="Opaque IDs returned by search, not filenames.")


class SurroundArgs(ToolArgs):
    video_id: str = Field(min_length=1, max_length=200)
    frame_id: int = Field(ge=0)
    radius: int = Field(default=3, ge=1, le=10)


class SimilarArgs(ToolArgs):
    video_id: str = Field(min_length=1, max_length=200)
    frame_id: int = Field(ge=0)
    top_k: int = Field(default=12, ge=1, le=100)


class PresentArgs(ToolArgs):
    frame_ids: list[str] = Field(max_length=100, description="Choose known frame IDs in desired display order. Empty clears results.")


class ParallelInspectArgs(ToolArgs):
    frame_ids: list[str] = Field(min_length=1, max_length=48)
    instruction: str = Field(default="Describe visible evidence and relevance for each frame.", min_length=1, max_length=2000)


class WorkerTask(ToolArgs):
    instruction: str = Field(min_length=1, max_length=2000)
    frame_ids: list[str] = Field(default_factory=list, max_length=6)


class ParallelTasksArgs(ToolArgs):
    tasks: list[WorkerTask] = Field(min_length=1, max_length=6)


TOOL_DEFINITIONS = {
    "list_capabilities": (CapabilitiesArgs, "List available datasets, image counts, mock search modes and limitations."),
    "search": (SearchArgs, "Search the selected dataset. Returns RANDOM real dataset images, not semantic matches. All modes are simulated."),
    "inspect_frames": (InspectArgs, "View actual image pixels for up to 6 known frame IDs. Describe only what you see; search scores are random."),
    "get_surrounding_frames": (SurroundArgs, "Get adjacent keyframes from the same dataset and video. Check synthetic_timing before trusting timestamps."),
    "similarity_search": (SimilarArgs, "Simulate image similarity search from an existing frame; results are random and dataset-isolated."),
    "present_results": (PresentArgs, "Display selected known frame IDs in the frontend grid. Call after inspecting to choose/reorder results."),
    "parallel_inspect": (ParallelInspectArgs, "Delegate up to 48 known frames in batches of 6 to independent vision workers running concurrently. Returns evidence reports. Use for bulk inspection; then select results with present_results."),
    "parallel_tasks": (ParallelTasksArgs, "Delegate up to 6 independent search/inspection tasks to bounded concurrent Codex workers. Workers use the same dataset and video filter. Returns reports and candidate frames for synthesis."),
}


class MockCatalog:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.frames: dict[str, list[dict]] = {key: [] for key in DATASETS}
        self.by_id: dict[str, dict] = {}
        self.by_video: dict[tuple[str, str], list[dict]] = {}

    def load(self):
        timing_by_video = {}
        for dataset in DATASETS:
            folder = self.root / "keyframes" / dataset.upper()
            videos = {}
            video_root = self.root / "videos" / dataset.upper()
            if video_root.exists():
                for path in video_root.rglob("*"):
                    if path.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov", ".avi"}:
                        videos.setdefault(path.stem, path.relative_to(self.root).as_posix())
            if not folder.exists():
                continue
            for path in sorted(folder.rglob("*")):
                if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"} or not path.is_file():
                    continue
                # Never follow a dataset symlink to another dataset or arbitrary files.
                if not path.resolve().is_relative_to(folder.resolve()):
                    continue
                relative = path.relative_to(folder).as_posix()
                video_id = path.parent.name
                video_key = (dataset, video_id)
                if video_key not in timing_by_video:
                    timing_by_video[video_key] = self._video_timing(dataset, video_id)
                digits = re.findall(r"\d+", path.stem)
                frame_id = int(digits[-1]) if digits else len(self.frames[dataset])
                identifier = f"{dataset}:{relative}"
                row = {
                    "id": identifier, "dataset": dataset, "collection": dataset,
                    "video_id": video_id, "frame_id": frame_id, "frame_idx": frame_id,
                    "keyframe_id": str(frame_id), "frame_name": path.name,
                    "timestamp": float(frame_id), "timestamp_sec": float(frame_id),
                    "image_url": "/" + quote(path.relative_to(self.root).as_posix(), safe="/"),
                    "image_rel_path": path.relative_to(self.root / "keyframes").as_posix(),
                    "video_url": "/" + quote(videos[video_id], safe="/") if video_id in videos else "",
                    "video_rel_path": videos[video_id].removeprefix("videos/") if video_id in videos else "",
                    "map_url": "", "keyframe_path": str(path),
                    "caption": "Mock sample · random ranking · synthetic timestamp",
                    "mock": True, "synthetic_timing": True,
                    "raw": {"frame_idx": frame_id, "dataset": dataset, "collection": dataset},
                }
                timing = timing_by_video[video_key].get(frame_id)
                if timing:
                    row.update(timing, synthetic_timing=False,
                               caption="Mock sample · random ranking · mapped timestamp")
                    row["raw"].update(frame_idx=row["frame_idx"], fps=row["fps"])
                    row["map_rel_path"] = f"{dataset.upper()}/{video_id}.csv"
                    row["map_url"] = "/mapkeyframes/" + quote(row["map_rel_path"], safe="/")
                self.frames[dataset].append(row)
                self.by_id[identifier] = row
                self.by_video.setdefault((dataset, video_id), []).append(row)
        for rows in self.by_video.values():
            rows.sort(key=lambda item: (item["frame_id"], item["id"]))

    def _video_timing(self, dataset, video_id):
        folder = self.root / "map_keyframes" / dataset.upper()
        path = folder / f"{video_id}.csv"
        if not path.is_file() or not path.resolve().is_relative_to(folder.resolve()):
            return {}
        timings = {}
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for record in csv.DictReader(stream):
                if record.get("video_id") not in (None, "", video_id):
                    continue
                keyframe_id = finite_number(record.get("keyframe_id"))
                frame_idx = finite_number(record.get("frame_idx"))
                fps = finite_number(record.get("fps"))
                timestamp = finite_number(record.get("timestamp_sec", record.get("timestamp")))
                if timestamp is None and frame_idx is not None and fps and fps > 0:
                    timestamp = frame_idx / fps
                if keyframe_id is None or timestamp is None or timestamp < 0 or frame_idx is None or frame_idx < 0:
                    continue
                timings[int(keyframe_id)] = {"frame_idx": int(frame_idx), "timestamp": timestamp,
                                            "timestamp_sec": timestamp, "fps": fps if fps and fps > 0 else None}
        return timings

    def capabilities(self):
        return {
            "mock": True, "notice": "Random dataset images. Scores/transcripts are simulated; timing uses CSV maps when available.",
            "datasets": [{"key": key, "label": key.upper(), "frame_count": len(self.frames[key]),
                          "video_count": len(self.video_ids(key))} for key in DATASETS],
            "modes": list(MODES), "models": [MODEL],
        }

    def video_ids(self, dataset):
        return sorted(video for key, video in self.by_video if key == dataset)

    def find(self, dataset, video_id, frame_id):
        rows = self.by_video.get((dataset, video_id), [])
        if not rows:
            raise ValueError(f"Video {video_id!r} does not exist in {dataset.upper()}")
        return min(rows, key=lambda row: abs(row["frame_id"] - frame_id))

    def selected(self, dataset, ids):
        rows = []
        for identifier in dict.fromkeys(ids):
            row = self.by_id.get(identifier)
            if row is None or row["dataset"] != dataset:
                raise ValueError(f"Unknown frame in selected dataset: {identifier}")
            rows.append(dict(row))
        return rows

    def search(self, dataset, query="", mode="semantic", top_k=20, model_key=MODEL, video_ids=None):
        rows = self.frames[dataset]
        if video_ids:
            allowed = set(video_ids)
            rows = [row for row in rows if row["video_id"] in allowed]
        chosen = random.sample(rows, min(top_k, len(rows)))
        results = []
        for index, row in enumerate(chosen):
            score = round(1 - index / max(len(chosen), 1) * .65, 4)
            if mode in {"asr", "caption"}:
                # Model each matched segment as 1-4 adjacent frames from the
                # same video. Keep the sampled hit inside its sequence as the
                # representative frame shown in the result header/card.
                video_frames = self.by_video[(dataset, row["video_id"])]
                center = next(i for i, frame in enumerate(video_frames) if frame["id"] == row["id"])
                sequence_size = random.randint(1, min(4, len(video_frames)))
                first_min = max(0, center - sequence_size + 1)
                first_max = min(center, len(video_frames) - sequence_size)
                first = random.randint(first_min, first_max)
                sequence = [dict(frame) for frame in video_frames[first:first + sequence_size]]
                text = f"[MOCK {mode.upper()}] {query}"
                sequence = [
                    {**item, "sub_query_idx": frame_index, "sub_query": text, "score": score}
                    for frame_index, item in enumerate(sequence)
                ]
                anchor = next(frame for frame in sequence if frame["id"] == row["id"])
                start = float(sequence[0]["timestamp_sec"])
                end = float(sequence[-1]["timestamp_sec"])
                result = {**anchor, "matched_sequence": sequence, "matched_texts": [text],
                          "caption": text, "temporal_start_time": start,
                          "temporal_end_time": end, "temporal_duration_sec": max(0.0, end - start)}
                if mode == "asr":
                    result["playback_start"] = start
            else:
                result = dict(row)
            result.update(rank=index + 1, similarity=score, retrieval_score=score,
                          search_mode=mode, model_key=model_key, source_models=[model_key])
            if mode == "ocr":
                result["matched_texts"] = [f"[MOCK {mode.upper()}] {query}"]
            if mode == "temporal":
                sequence = self.surrounding(dataset, row["video_id"], row["frame_id"], 1)
                result["matched_sequence"] = [
                    {**item, "sub_query_idx": idx, "sub_query": f"Mock event {idx + 1}", "score": score}
                    for idx, item in enumerate(sequence)
                ]
            results.append(result)
        return {"mock": True, "query": query, "original_query": query, "mode": mode,
                "search_mode": mode, "model_key": model_key, "count": len(results),
                "results": results, "latency_ms": 15, "notice": "Random ranking; not real retrieval."}

    def surrounding(self, dataset, video_id, frame_id, radius):
        center = self.find(dataset, video_id, frame_id)
        rows = self.by_video[(dataset, video_id)]
        index = next(i for i, row in enumerate(rows) if row["id"] == center["id"])
        return [dict(row) for row in rows[max(0, index - radius):index + radius + 1]]

    def image_content(self, row):
        from PIL import Image, ImageOps
        path = Path(row["keyframe_path"]).resolve()
        if not path.is_relative_to((self.root / "keyframes" / row["dataset"].upper()).resolve()):
            raise ValueError("Image is outside selected dataset")
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((768, 768))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=75)
        return {"type": "inputImage", "imageUrl": "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()}


class MockTools:
    """One per turn; IDs must be seen by this session before inspect/present."""
    def __init__(self, catalog, dataset, known_ids=None, video_ids=None, displayed=None, max_workers=3, worker_slots=None):
        self.catalog = catalog
        self.dataset = dataset
        self.known_ids = set(known_ids or [])
        self.video_ids = video_ids or []
        self.displayed = displayed or []
        self.calls = 0
        self.max_calls = 12
        self.max_workers = max_workers
        self.worker_slots = worker_slots
        self.is_worker = False
        self.input_images = []
        self.preferred_search_mode = "semantic"

    @staticmethod
    def specs():
        return [{"type": "function", "name": name, "description": description,
                 "inputSchema": schema.model_json_schema()}
                for name, (schema, description) in TOOL_DEFINITIONS.items()]

    def execute(self, name, arguments):
        if name not in TOOL_DEFINITIONS:
            raise ValueError(f"Tool not allowed: {name}")
        self.calls += 1
        if self.calls > self.max_calls:
            raise ValueError("Tool budget exceeded. Finish using the existing results.")
        args = TOOL_DEFINITIONS[name][0].model_validate(arguments)
        if name in {"parallel_inspect", "parallel_tasks"}:
            raise ValueError("Parallel tools require the asynchronous Codex dispatcher")
        content = []
        if name == "list_capabilities":
            data = self.catalog.capabilities()
        elif name == "search":
            data = self.catalog.search(self.dataset, args.query, args.mode, args.top_k, video_ids=self.video_ids)
        elif name in {"inspect_frames", "present_results"}:
            if any(identifier not in self.known_ids for identifier in args.frame_ids):
                raise ValueError("Use only frame IDs returned by a previous tool in this session")
            rows = self.catalog.selected(self.dataset, args.frame_ids)
            if self.video_ids and any(row["video_id"] not in self.video_ids for row in rows):
                raise ValueError("Frame excluded by current video filter")
            data = {"mock": True, "results": rows, "count": len(rows)}
            if name == "inspect_frames":
                for row in rows:
                    content.append({"type": "inputText", "text": f"Image {row['id']}"})
                    content.append(self.catalog.image_content(row))
        elif name == "get_surrounding_frames":
            if self.video_ids and args.video_id not in self.video_ids:
                raise ValueError("Video excluded by current video filter")
            rows = self.catalog.surrounding(self.dataset, args.video_id, args.frame_id, args.radius)
            data = {"mock": True, "results": rows, "count": len(rows)}
        else:
            self.catalog.find(self.dataset, args.video_id, args.frame_id)
            data = self.catalog.search(self.dataset, f"similarity:{args.video_id}/{args.frame_id}",
                                       "similarity", args.top_k, video_ids=self.video_ids)
        self.known_ids.update(row["id"] for row in data.get("results", []))
        return data, content
