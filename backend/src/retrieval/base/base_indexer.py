from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable


class BaseIndexer(ABC):
    """Common interface for all indexing backends (Elasticsearch, FAISS, Milvus...).

    An indexer is only responsible for creating/writing an index. It must not
    contain query/search logic — that belongs to a `BaseRetriever` implementation
    which reads from the index this indexer builds.
    """

    @abstractmethod
    def create_index(self, *args: Any, **kwargs: Any) -> Any:
        """Create the underlying index/collection if it does not already exist."""
        raise NotImplementedError

    @abstractmethod
    def insert(self, document: Any) -> Any:
        """Insert a single document/record into the index."""
        raise NotImplementedError

    @abstractmethod
    def bulk_insert(self, documents: Iterable[Any]) -> Any:
        """Insert many documents/records into the index in one batch operation."""
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        """Return the number of documents/records currently stored in the index."""
        raise NotImplementedError
