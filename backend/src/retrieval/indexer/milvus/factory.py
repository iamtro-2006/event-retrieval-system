# TODO(refactor): placeholder - dung khi thuc su tich hop Milvus, doc config
# configs/milvus.yaml (chua ton tai) theo dung pattern cua asr/ocr factory.py.
from __future__ import annotations

from src.retrieval.indexer.milvus.client import MilvusClient
from src.retrieval.indexer.milvus.repository import MilvusRepository


def build_milvus_repository(host: str, port: int, collection_name: str) -> MilvusRepository:
    client = MilvusClient(host=host, port=port, collection_name=collection_name)
    return MilvusRepository(client=client)
