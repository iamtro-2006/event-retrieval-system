from __future__ import annotations

import json
from pathlib import Path

from src.ocr.pipelines.repository import OCRRepository
from src.ocr.schemas.document import OCRDocument


class IndexPipeline:

    def __init__(
        self,
        repository: OCRRepository,
    ) -> None:

        self.repository = repository

    def create_index(self):

        self.repository.create_index()

    def index_json(
        self,
        json_path: str | Path,
    ):

        json_path = Path(json_path)

        dataset = json_path.parent.name

        video_id = json_path.stem

        with open(
            json_path,
            "r",
            encoding="utf-8",
        ) as f:

            data = json.load(f)

        documents: list[OCRDocument] = []

        for keyframe_id, bboxs, texts in data.items():

            documents.append(

                OCRDocument(

                    dataset=dataset,

                    video_id=video_id,

                    keyframe_id=keyframe_id,

                    bboxs=bboxs,

                    texts=texts,

                )

            )

        self.repository.bulk_insert(
            documents
        )

        print(
            f"Indexed {len(documents)} documents from {video_id}"
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