"""
Common interface the translation backend must implement, so `main.py`
can call `translate()` without depending on the concrete implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseTranslator(ABC):
    """Minimal translator contract shared by all backends."""

    @abstractmethod
    def translate(self, text: str, source: str = "vi", target: str = "en") -> str:
        """Translate `text` from `source` language to `target` language.

        Implementations should return the original `text` unchanged when it
        is empty/whitespace-only.
        """
        raise NotImplementedError
