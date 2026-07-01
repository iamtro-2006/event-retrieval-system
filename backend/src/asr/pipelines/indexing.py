from __future__ import annotations

import json
from pathlib import Path

from src.asr.pipelines.repository import ASRRepository
from src.asr.schemas.document import ASRDocument


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
        """Index one video's transcript file.

        Expected JSON shape (list of segments, sorted by time):

            [
                {"segment_id": "000000", "start_time": 0.0, "end_time": 4.2, "text": "..."},
                {"segment_id": "000001", "start_time": 4.2, "end_time": 9.8, "text": "..."},
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

                    segment_id=str(segment["segment_id"]),

                    start_time=float(segment["start_time"]),

                    end_time=float(segment["end_time"]),

                    text=str(segment.get("text", "")),

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
