from __future__ import annotations

import json
from pathlib import Path

from src.retrieval.indexer.elasticsearch.ocr.repository import OCRRepository
from src.retrieval.indexer.elasticsearch.ocr.schemas import OCRDocument


def _resolve_labels(json_path: Path, collection: str | None) -> tuple[str, str]:
    """Derive (collection, dataset) from a per-video OCR json path.

    Both layouts the repo documents must land on the same values, because the
    retriever joins these fields against FAISS metadata:

        <root>/ocr/<video_id>.json           -> dataset=<root name>, collection=AIC
        <root>/ocr/<DATASET>/<video_id>.json -> dataset=<DATASET>,    collection=AIC

    Deriving them positionally (`parents[2].name` / `parent.name`) gets them
    swapped in the flat layout, which silently breaks OCR hits: the metadata
    join looks up `dataset` and finds the string "ocr".
    """
    parts = [part for part in json_path.parts if part]
    video_id_index = len(parts) - 1
    parent = parts[video_id_index - 1] if video_id_index >= 1 else ""

    # A directory that is just the media kind ("ocr"/"asr") is the root the
    # caller passed in, not a dataset label. Everything above it is.
    dataset = ""
    if parent and parent.lower() in {"ocr", "asr"}:
        for part in reversed(parts[: video_id_index - 1]):
            if part.lower() in {"ocr", "asr"}:
                continue
            dataset = part
            break
    else:
        dataset = parent

    resolved_collection = str(collection or "").strip().lower()
    if not resolved_collection or resolved_collection in {"ocr", "asr"}:
        resolved_collection = dataset.lower()
    return resolved_collection, dataset


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
        collection: str | None = None,
    ):

        json_path = Path(json_path)

        video_id = json_path.stem

        collection, dataset = _resolve_labels(json_path, collection)

        with open(
            json_path,
            "r",
            encoding="utf-8",
        ) as f:

            data = json.load(f)

        documents: list[OCRDocument] = []

        for keyframe_id, values in data.items():

            bboxs, texts = values 
            documents.append(

                OCRDocument(

                    collection=collection,

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

            self.index_json(json_file, collection=folder.parent.name)
