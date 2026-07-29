"""
Remote translation backend that calls an LLM chat-completions API (via
CKEY.VN's OpenAI-compatible gateway: https://ckey.vn/llm-api/setup) instead
of running a local model.

Trade-offs vs. the local EnviT5 backend:
  + No local GPU/VRAM cost, no model load time.
  + Can be "personalized" per call via a prompt template (domain glossary,
    tone, preserve certain terms untranslated, etc.) -- something a fixed
    seq2seq model can't do without fine-tuning.
  + Often higher quality on ambiguous/colloquial Vietnamese queries.
  - Network round-trip latency (typically 200ms-1s+ depending on provider/
    model/load), so for pure "fastest possible" single-query latency the
    local EnviT5 GPU backend usually still wins. Use this backend when
    personalization/quality matters more than raw latency, or when there's
    no local GPU available at all.
  - Costs money per call; a small in-process cache is included to avoid
    re-paying for repeated identical queries.

Drop-in replacement for the other translators: implements the same
`BaseTranslator.translate()` contract, so it can be swapped in purely via
`configs/app.yaml`'s `translate_agent: llm`.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from .base_translator import BaseTranslator

_LANG_NAMES = {"vi": "Vietnamese", "en": "English"}

# Kept short and strict: the model should behave like a translation *engine*,
# not a chatty assistant -- no preamble, no explanation, no markdown fences.
_DEFAULT_SYSTEM_TEMPLATE = (
    "You are a translation engine embedded in a video search system. "
    "Translate the user's search query from {source_lang} to {target_lang}. "
    "Preserve named entities, numbers, and the original search intent. "
    "Output ONLY the translated query text -- no quotes, no explanation, "
    "no markdown.{persona_block}"
)


class _LRUCache:
    """Tiny thread-safe LRU cache so repeated identical queries don't
    re-hit the API (saves both latency and cost)."""

    def __init__(self, maxsize: int = 2048):
        self._maxsize = maxsize
        self._data: "OrderedDict[tuple, str]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def set(self, key, value):
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            if len(self._data) > self._maxsize:
                self._data.popitem(last=False)


class LLMTranslator(BaseTranslator):
    """Thin singleton wrapper around an OpenAI-compatible chat-completions
    endpoint (e.g. CKEY.VN), used as a translation engine.

    `persona` lets you inject a per-deployment customization block into the
    system prompt -- e.g. a domain glossary ("always translate 'công an' as
    'police', never 'public security'"), a tone instruction, or a note about
    what kind of video corpus the queries come from. This is the
    "personalization" a fixed local model can't do without fine-tuning.
    """

    _instance: Optional["LLMTranslator"] = None
    _lock = threading.Lock()

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.xah.io/v1",
        model: str = "claude-haiku-4.5",
        timeout: float = 15.0,
        max_retries: int = 2,
        max_workers: int = 8,
        temperature: float = 0.0,
        max_output_tokens: int = 128,
        persona: str = "",
        cache_size: int = 2048,
    ):
        if not api_key:
            raise ValueError(
                "LLMTranslator requires an API key. Set it via "
                "configs/app.yaml -> translate.llm.api_key or the "
                "CKEY_API_KEY / LLM_TRANSLATE_API_KEY env var."
            )

        # Imported lazily so `openai` is only required if this backend is
        # actually selected (mirrors the lazy `torch`/`transformers` imports
        # in the local backends).
        from openai import OpenAI

        self._client = OpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries
        )
        self._model = model
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._persona = persona.strip()
        self._cache = _LRUCache(maxsize=cache_size)
        # A ThreadPoolExecutor is the right concurrency primitive here (not
        # GPU batching): each translate() call is an independent HTTP
        # request, so wall-clock time for N queries drops from N*latency to
        # roughly latency + queueing, bounded by max_workers.
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="llm-translate"
        )

    @classmethod
    def get_instance(
        cls,
        api_key: str,
        base_url: str = "https://api.xah.io/v1",
        model: str = "claude-haiku-4.5",
        timeout: float = 15.0,
        max_retries: int = 2,
        max_workers: int = 8,
        temperature: float = 0.0,
        max_output_tokens: int = 128,
        persona: str = "",
        cache_size: int = 2048,
    ) -> "LLMTranslator":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(
                        api_key=api_key,
                        base_url=base_url,
                        model=model,
                        timeout=timeout,
                        max_retries=max_retries,
                        max_workers=max_workers,
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                        persona=persona,
                        cache_size=cache_size,
                    )
        return cls._instance

    def _system_prompt(self, source: str, target: str) -> str:
        persona_block = f"\n\nAdditional instructions:\n{self._persona}" if self._persona else ""
        return _DEFAULT_SYSTEM_TEMPLATE.format(
            source_lang=_LANG_NAMES.get(source, source),
            target_lang=_LANG_NAMES.get(target, target),
            persona_block=persona_block,
        )

    def translate(self, text: str, source: str = "vi", target: str = "en") -> str:
        """Synchronous translate call, matching the other backends' signature."""
        if not text or not text.strip():
            return text

        stripped = text.strip()
        cache_key = (stripped, source, target, self._model, self._persona)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        response = self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_output_tokens,
            messages=[
                {"role": "system", "content": self._system_prompt(source, target)},
                {"role": "user", "content": stripped},
            ],
        )
        out = (response.choices[0].message.content or "").strip()
        # Strip stray wrapping quotes some models add despite instructions.
        if len(out) >= 2 and out[0] == out[-1] and out[0] in "\"'":
            out = out[1:-1].strip()

        self._cache.set(cache_key, out)
        return out

    def translate_batch(
        self, texts: List[str], source: str = "vi", target: str = "en"
    ) -> List[str]:
        """Translate many strings concurrently (thread pool, one HTTP
        request per string) instead of sequentially -- this is the
        throughput lever for a remote API backend, analogous to GPU batching
        for the local backend.

        On a per-item failure, falls back to the original untranslated text
        for that item rather than failing the whole batch.
        """
        if not texts:
            return []

        results = list(texts)
        indices_to_run = [i for i, t in enumerate(texts) if t and t.strip()]
        if not indices_to_run:
            return results

        futures = {
            self._executor.submit(self.translate, texts[i], source, target): i
            for i in indices_to_run
        }
        for future in futures:
            i = futures[future]
            try:
                results[i] = future.result()
            except Exception:
                # Network/API failure: degrade gracefully to the original
                # query rather than breaking the whole search request.
                results[i] = texts[i]

        return results
