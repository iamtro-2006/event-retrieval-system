# TODO(refactor): placeholder - chua trien khai, chi dung de dam bao khung
# thu muc dung theo REFACTOR_PLAN.md muc 1 (backend moi cung interface base_indexer).
from __future__ import annotations

from typing import Any, Iterable

from src.retrieval.base.base_indexer import BaseIndexer
from src.retrieval.indexer.milvus.client import MilvusClient


class MilvusRepository(BaseIndexer):
    """Placeholder `BaseIndexer` implementation for a future Milvus backend."""

    def __init__(self, client: MilvusClient) -> None:
        self.client = client

    def create_index(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Milvus backend chua duoc trien khai.")

    def insert(self, document: Any) -> Any:
        raise NotImplementedError("Milvus backend chua duoc trien khai.")

    def bulk_insert(self, documents: Iterable[Any]) -> Any:
        raise NotImplementedError("Milvus backend chua duoc trien khai.")

    def count(self) -> int:
        raise NotImplementedError("Milvus backend chua duoc trien khai.")
