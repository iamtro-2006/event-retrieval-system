# TODO(refactor): placeholder cho backend Milvus, chua co ket noi that.
# Muc dich: giu dung interface/pattern voi elasticsearch/client.py de sau nay
# cam vao BaseIndexer (retrieval/base/base_indexer.py) ma khong phai sua noi
# goi (orchestrator/router). Chua trien khai logic ket noi Milvus thuc te.
from __future__ import annotations

from typing import Any


class MilvusClient:
    """Thin wrapper around a Milvus connection, mirroring the constructor
    shape of `elasticsearch/client.py::ElasticsearchService` (host, port,
    collection_name as explicit args) so the two backends stay swappable.
    """

    def __init__(self, host: str, port: int, collection_name: str, **kwargs: Any) -> None:
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self._extra_kwargs = kwargs
        self._connection: Any = None

    def connect(self) -> Any:
        raise NotImplementedError("Milvus backend chua duoc trien khai.")

    def close(self) -> None:
        self._connection = None
