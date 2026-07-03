from __future__ import annotations

from typing import Any

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk


class ElasticsearchService:
    """Thin wrapper around the Elasticsearch client, scoped to a single index.

    Dùng chung cho cả ASR và OCR (trước đây là 2 bản copy-paste gần như
    giống hệt nhau, chỉ khác field dùng để full-text search).
    """

    def __init__(
        self,
        host: str,
        port: int,
        index_name: str,
        scheme: str = "http",
        search_field: str = "text",
    ) -> None:
        self.index_name = index_name
        self.search_field = search_field
        self.client = Elasticsearch(f"{scheme}://{host}:{port}")

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        return self.client.ping()

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------

    def index_exists(self) -> bool:
        return self.client.indices.exists(index=self.index_name)

    def delete_index(self) -> None:
        if self.index_exists():
            self.client.indices.delete(index=self.index_name)

    def create_index(self, mapping: dict[str, Any]) -> None:
        if self.index_exists():
            return

        self.client.indices.create(
            index=self.index_name,
            mappings=mapping,
        )

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def insert(self, document: dict[str, Any]) -> None:
        self.client.index(
            index=self.index_name,
            document=document,
        )

    def bulk_insert(self, documents: list[dict[str, Any]]) -> None:
        actions = [
            {"_index": self.index_name, "_source": doc}
            for doc in documents
        ]
        bulk(self.client, actions)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, size: int = 10) -> list[dict]:
        response = self.client.search(
            index=self.index_name,
            query={
                "match": {
                    self.search_field: {
                        "query": query,
                        "fuzziness": "AUTO",
                    }
                }
            },
            size=size,
        )
        return response["hits"]["hits"]

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------

    def count(self) -> int:
        response = self.client.count(index=self.index_name)
        return response["count"]
