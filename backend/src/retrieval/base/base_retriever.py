from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Hit:
    """A single normalized search result, shared across all retriever types
    (semantic / temporal / ocr / asr) so callers (orchestrator, API routers)
    can treat results uniformly regardless of backend."""

    id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseRetriever(ABC):
    """Common interface for all retriever/search backends.

    A retriever is only responsible for querying — it must not manage
    writing/ingesting data into the index (that is `BaseIndexer`'s job).
    """

    @abstractmethod
    def search(self, query: Any, top_k: int = 10, **kwargs: Any) -> list[Hit]:
        """Run a search for `query` and return up to `top_k` ranked `Hit`s."""
        raise NotImplementedError
