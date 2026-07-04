from __future__ import annotations

import json
from pathlib import Path

from src.retrieval.indexer.elasticsearch.asr.repository import ASRRepository
from src.retrieval.indexer.elasticsearch.asr.schemas import ASRDocument


class IndexPipeline:

    def __init__(
        self,
        repository: ASRRepository,
    ) -> None:

        self.repository = repository

    def create_index(self):

        self.repository.create_index()

    def index_json(
        self,
        json_path: str | Path,
    ):
        """Index one video's transcript file (extract_asr output).

        video_id is taken from the filename (json_path.stem), same convention
        as the OCR pipeline. The "video_id" field inside each segment is
        redundant/ignored.

        Expected JSON shape (list of segments, sorted by time, no segment_id):

            [
                {"video_id": "L21_V001", "start": 4.43, "end": 37.94, "transcript": "..."},
                {"video_id": "L21_V001", "start": 37.94, "end": 69.39, "transcript": "..."},
                ...
            ]
        """

        json_path = Path(json_path)

        dataset = json_path.parent.name

        video_id = json_path.stem

        with open(
            json_path,
            "r",
            encoding="utf-8",
        ) as f:

            data = json.load(f)

        documents: list[ASRDocument] = []

        for segment in data:

            documents.append(

                ASRDocument(

                    dataset=dataset,

                    video_id=video_id,

                    start_time=float(segment["start"]),

                    end_time=float(segment["end"]),

                    text=str(segment.get("transcript", "")),

                )

            )

        self.repository.bulk_insert(
            documents
        )

        print(
            f"Indexed {len(documents)} segments from {video_id}"
        )

    def index_folder(
        self,
        folder: str | Path,
    ):

        folder = Path(folder)

        json_files = sorted(
            folder.rglob("*.json")
        )

        print(f"Found {len(json_files)} json files")

        for json_file in json_files:

            self.index_json(json_file)
