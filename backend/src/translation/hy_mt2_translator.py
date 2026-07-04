"""
Local translation backend using Tencent Hy-MT2-1.8B (2-bit GGUF quantization)
run through llama.cpp / llama-cpp-python.

Drop-in replacement for deep_translator.GoogleTranslator, so it can be swapped
into main.py's `translate_query_if_needed` with minimal changes.

Model card: https://huggingface.co/tencent/Hy-MT2-1.8B-2Bit-GGUF
"""

from __future__ import annotations

import re
import threading
from typing import Optional

from llama_cpp import Llama

from .base_translator import BaseTranslator

# Full language names expected by the Hy-MT2 prompt template.
# (Hy-MT2 was trained on full language names, not ISO codes — see model card.)
_LANG_NAMES = {
    "vi": "Vietnamese",
    "en": "English",
    "zh": "Chinese",
    "fr": "French",
    "pt": "Portuguese",
    "es": "Spanish",
    "ja": "Japanese",
    "tr": "Turkish",
    "ru": "Russian",
    "ar": "Arabic",
    "ko": "Korean",
    "th": "Thai",
    "it": "Italian",
    "de": "German",
    "ms": "Malay",
    "id": "Indonesian",
    "hi": "Hindi",
}

_PROMPT_TEMPLATE = (
    "Strictly translate the following text into {target_lang}. "
    "Note that you should only output the translated result "
    "without any additional explanation:\n{source_text}"
)

# Special/control tokens that sometimes leak into the raw completion when the
# GGUF's embedded chat template doesn't line up exactly with chat_format="chatml".
_STOP_TOKENS = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]

# Common preambles the model tacks on despite being told not to
# (e.g. "Translation:", "Đây là bản dịch:", "Here is the translation:").
_PREAMBLE_RE = re.compile(
    r"^\s*(here('?s| is)( the)? translation|translation|"
    r"đây là bản dịch|bản dịch)\s*[:：]\s*",
    re.IGNORECASE,
)

# Leaked control tokens (e.g. mid-generation, or from a mismatched template).
_SPECIAL_TOKEN_RE = re.compile(r"<\|.*?\|>")

# A leading role label left behind once special tokens are stripped
# (may repeat, e.g. "assistant\nassistant: ...").
_ROLE_LABEL_RE = re.compile(
    r"^\s*(user|assistant|system)\s*[:：]?\s*", re.IGNORECASE
)


def _clean_output(raw: str) -> str:
    """
    Post-process a raw Hy-MT2 completion into a clean translated string.

    Handles, in order:
      1. Leaked chat-template / special tokens (<|im_start|>, role labels, ...)
      2. Boilerplate preambles ("Translation:", "Đây là bản dịch:", ...)
      3. Wrapping quotes the model sometimes adds around the whole result
      4. Stray leading/trailing whitespace and blank lines
    """
    if not raw:
        return raw

    text = raw

    # 1. Strip any leaked special tokens anywhere in the string, then any
    #    leading role label(s) left behind (may repeat after stripping).
    text = _SPECIAL_TOKEN_RE.sub("", text)
    prev = None
    while prev != text:
        prev = text
        text = _ROLE_LABEL_RE.sub("", text.strip())

    # 2. Strip a leading "Translation:" style preamble, if present.
    text = _PREAMBLE_RE.sub("", text)

    text = text.strip()

    # 3. If the whole thing is wrapped in a single matching pair of quotes,
    #    unwrap it (model sometimes echoes the result as a literal string).
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("\"", "'", "“", "”"):
        inner = text[1:-1].strip()
        # Only unwrap if it doesn't just remove a genuinely-quoted phrase
        # (heuristic: no other unmatched quote of the same kind inside).
        if inner.count(text[0]) == 0:
            text = inner

    # 4. Collapse accidental blank lines / trailing whitespace.
    text = "\n".join(line.rstrip() for line in text.splitlines()).strip()

    return text


class HyMT2Translator(BaseTranslator):
    """
    Thin singleton wrapper around a local Hy-MT2-1.8B GGUF model.

    Loading a ~440MB-1GB GGUF model on every request would be slow, so the
    underlying Llama instance is created once per process and reused
    (thread-safe via a lock, since llama.cpp contexts aren't reentrant).
    """

    _instance: Optional["HyMT2Translator"] = None
    _lock = threading.Lock()

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_threads: int | None = None,
        n_gpu_layers: int = 0,
        verbose: bool = False,
    ):
        self._call_lock = threading.Lock()
        self._llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,        # None -> llama.cpp autodetects
            n_gpu_layers=n_gpu_layers,  # 0 = CPU only; set e.g. -1 for full GPU offload
            verbose=verbose,
            chat_format="chatml",       # Hy-MT2 ships a chat template; llama-cpp-python
                                         # will actually prefer the GGUF's embedded
                                         # jinja template if present.
        )

    @classmethod
    def get_instance(
        cls,
        model_path: str,
        n_ctx: int = 2048,
        n_threads: int | None = None,
        n_gpu_layers: int = 0,
    ) -> "HyMT2Translator":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(
                        model_path=model_path,
                        n_ctx=n_ctx,
                        n_threads=n_threads,
                        n_gpu_layers=n_gpu_layers,
                    )
        return cls._instance

    @staticmethod
    def _resolve_lang_name(code_or_name: str) -> str:
        code = code_or_name.strip().lower()
        return _LANG_NAMES.get(code, code_or_name)

    def translate(self, text: str, source: str = "vi", target: str = "en") -> str:
        """
        Synchronous translate call, mirroring
        deep_translator.GoogleTranslator(source, target).translate(text) API shape.
        """
        if not text or not text.strip():
            return text

        target_lang = self._resolve_lang_name(target)
        prompt = _PROMPT_TEMPLATE.format(target_lang=target_lang, source_text=text)

        messages = [{"role": "user", "content": prompt}]

        # llama.cpp contexts hold internal KV-cache state; serialize calls
        # so concurrent FastAPI requests don't corrupt each other's generation.
        with self._call_lock:
            result = self._llm.create_chat_completion(
                messages=messages,
                temperature=0.2,
                top_p=0.6,
                top_k=20,
                repeat_penalty=1.05,
                max_tokens=1024,
                stop=_STOP_TOKENS,
            )

        raw = result["choices"][0]["message"]["content"]
        return _clean_output(raw)