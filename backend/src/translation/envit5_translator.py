"""
Local translation backend using VietAI/envit5-translation (T5-based,
fine-tuned specifically for the vi<->en pair).

Model card: https://huggingface.co/VietAI/envit5-translation

Small, purpose-built seq2seq model (no chat-template parsing / preamble
stripping needed), so it's both faster and cheaper to run than a generic
instruction-following LLM.

NOTE ON TOKENIZATION: newer `transformers` releases dropped the pure-Python
"slow" tokenizer implementations and always convert SentencePiece vocabs to
the Rust `tokenizers` library on load. That conversion path has a known
incompatibility with some `tokenizers` versions for T5-style Unigram vocabs
(raises `TypeError: argument 'vocab': 'dict' object cannot be converted to
'Sequence'`). To sidestep it entirely, this backend talks to the model's
`spiece.model` SentencePiece file directly via the `sentencepiece` package
instead of going through `AutoTokenizer` at all.

PERFORMANCE NOTES (local GPU):
- `dtype="auto"` loads the model in fp16 on CUDA (bf16 if the GPU supports
  it better) instead of the fp32 default, which is ~2x faster and halves
  VRAM with negligible quality loss for this model size.
- SDPA (`attn_implementation="sdpa"`) is requested when available, which
  uses PyTorch's fused/Flash-Attention kernels instead of the slower eager
  attention path.
- `num_beams` defaults to 1 (greedy decoding). Beam search (the previous
  default of 4) multiplies decode cost ~4x for a small accuracy gain that
  rarely matters for short search-query translations; override via
  `ENVIT5_NUM_BEAMS` / `configs/app.yaml` if you need beams back.
- `translate_batch()` lets callers translate many queries in a single
  forward pass instead of one-by-one, which is where GPU throughput is
  actually won (single-sequence decoding is latency-, not throughput-,
  bound). `translate()` is kept as a thin wrapper around it for API
  compatibility.
- cuDNN autotuning (`torch.backends.cudnn.benchmark = True`) is enabled
  since input shapes for short queries are fairly stable across calls.
"""

from __future__ import annotations

import threading
from typing import List, Optional

from .base_translator import BaseTranslator

# envit5 was trained with an explicit "<lang>: " prefix on the source text
# and expects it echoed back on the output, e.g. input "vi: xin chào" ->
# output "en: hello". See the model card's usage example.
_PREFIX = {"vi": "vi: ", "en": "en: "}


def _strip_output_prefix(text: str) -> str:
    """Remove the leading "en: " / "vi: " tag envit5 echoes on its output."""
    stripped = text.strip()
    for tag in ("en:", "vi:"):
        if stripped.lower().startswith(tag):
            return stripped[len(tag):].strip()
    return stripped


class EnviT5Translator(BaseTranslator):
    """Thin singleton wrapper around VietAI/envit5-translation, tuned for
    fast local GPU inference.

    Tokenization is done directly through `sentencepiece` (bypassing
    `transformers.AutoTokenizer`) to avoid a vocab-conversion bug between
    recent `transformers`/`tokenizers` versions and T5-style SentencePiece
    models -- see module docstring.
    """

    _instance: Optional["EnviT5Translator"] = None
    _lock = threading.Lock()

    def __init__(
        self,
        model_name: str = "VietAI/envit5-translation",
        device: str = "auto",
        max_length: int = 512,
        num_beams: int = 1,
        dtype: str = "auto",
        max_new_tokens: int = 96,
        warmup: bool = True,
    ):
        # Imported lazily so the (fairly heavy) transformers/torch import
        # only happens if this backend is actually selected.
        import sentencepiece as spm
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoModelForSeq2SeqLM

        from src.utils.device import resolve_device

        self._call_lock = threading.Lock()
        self._max_length = max_length
        # Search-query translations are short (a few words to a sentence),
        # so capping generate()'s output length well below max_length avoids
        # generate() reserving/bookkeeping for a 512-token ceiling it will
        # never approach. EOS still stops generation early regardless, this
        # is just a tighter safety ceiling.
        self._max_new_tokens = max(1, max_new_tokens)
        self._num_beams = max(1, num_beams)
        self._torch = torch
        self._device = resolve_device(device)

        if self._device.type == "cuda":
            # Stable-shaped short-sequence workload -> autotuning pays off.
            torch.backends.cudnn.benchmark = True
            # TF32 matmuls are free throughput on Ampere+ for the fp32 bits
            # that remain (e.g. layernorm accumulation) when running fp16/bf16.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        model_dtype = self._resolve_dtype(dtype)

        spiece_path = hf_hub_download(repo_id=model_name, filename="spiece.model")
        self._sp = spm.SentencePieceProcessor()
        self._sp.load(spiece_path)

        load_kwargs = {"torch_dtype": model_dtype}
        try:
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, attn_implementation="sdpa", **load_kwargs
            )
        except (ValueError, TypeError):
            # Older transformers versions, or a build without SDPA support
            # for this model's attention class -> fall back to default attn.
            self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **load_kwargs)

        self._model.to(self._device)
        self._model.eval()

        # T5 SentencePiece vocabs reserve id 0 for <pad> and 1 for </s>
        # (no <bos>). Prefer the loaded model's own config if it disagrees.
        cfg = self._model.config
        self._pad_id = int(cfg.pad_token_id if cfg.pad_token_id is not None else 0)
        self._eos_id = int(cfg.eos_token_id if cfg.eos_token_id is not None else 1)

        if warmup and self._device.type == "cuda":
            # cuBLAS/cuDNN pick their fastest kernel per-shape the first time
            # that shape is seen ("autotuning"), which is a big one-time cost
            # (often several hundred ms). Paying it here at model-load time
            # means the first real user request doesn't eat it.
            try:
                self.translate_batch(["xin chào"], source="vi", target="en")
            except Exception:
                pass

    def _resolve_dtype(self, dtype: str):
        torch = self._torch
        dtype = (dtype or "auto").strip().lower()
        if dtype == "fp32":
            return torch.float32
        if dtype == "fp16":
            return torch.float16
        if dtype == "bf16":
            return torch.bfloat16
        # auto: half precision on GPU (bf16 when the GPU supports it well,
        # else fp16); fp32 on CPU where half precision isn't reliably faster.
        if self._device.type == "cuda":
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32

    @classmethod
    def get_instance(
        cls,
        model_name: str = "VietAI/envit5-translation",
        device: str = "auto",
        max_length: int = 512,
        num_beams: int = 1,
        dtype: str = "auto",
        max_new_tokens: int = 96,
        warmup: bool = True,
    ) -> "EnviT5Translator":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(
                        model_name=model_name,
                        device=device,
                        max_length=max_length,
                        num_beams=num_beams,
                        dtype=dtype,
                        max_new_tokens=max_new_tokens,
                        warmup=warmup,
                    )
        return cls._instance

    def _encode(self, text: str) -> List[int]:
        ids = self._sp.encode(text, out_type=int)[: self._max_length - 1]
        ids.append(self._eos_id)
        return ids

    def _decode(self, ids: list[int]) -> str:
        cleaned = [i for i in ids if i not in (self._pad_id, self._eos_id) and i >= 0]
        return self._sp.decode(cleaned)

    def _build_batch_inputs(self, prompts: List[str]):
        torch = self._torch
        encoded = [self._encode(p) for p in prompts]
        max_len = max(len(ids) for ids in encoded)

        input_ids = torch.full((len(encoded), max_len), self._pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(encoded), max_len), dtype=torch.long)
        for i, ids in enumerate(encoded):
            # Left padding wastes no positions on generate() for T5 encoder-
            # decoder models (encoder is bidirectional), right padding is
            # simplest and standard here.
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, : len(ids)] = 1

        return input_ids.to(self._device), attention_mask.to(self._device)

    def translate_batch(
        self, texts: List[str], source: str = "vi", target: str = "en"
    ) -> List[str]:
        """Translate many strings in a single forward pass.

        This is the throughput-efficient path: batching amortizes kernel
        launch overhead and keeps the GPU fed, unlike calling `translate()`
        in a loop. Empty/whitespace-only entries are passed through
        unchanged (matching `translate()`'s behaviour) without being sent
        through the model.
        """
        if not texts:
            return []

        prefix = _PREFIX.get(source, f"{source}: ")
        indices_to_run = [i for i, t in enumerate(texts) if t and t.strip()]
        results = list(texts)

        if not indices_to_run:
            return results

        prompts = [f"{prefix}{texts[i].strip()}" for i in indices_to_run]

        with self._call_lock:
            input_ids, attention_mask = self._build_batch_inputs(prompts)

            with self._torch.inference_mode():
                output_ids = self._model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self._max_new_tokens,
                    num_beams=self._num_beams,
                    do_sample=False,
                )

            for out_i, orig_i in enumerate(indices_to_run):
                raw = self._decode(output_ids[out_i].tolist())
                results[orig_i] = _strip_output_prefix(raw)

        return results

    def translate(self, text: str, source: str = "vi", target: str = "en") -> str:
        """Synchronous translate call, matching the other backends' signature.

        Thin wrapper over `translate_batch` for a single string; prefer
        `translate_batch` directly when translating multiple queries so the
        GPU sees one batched forward pass instead of N sequential ones.
        """
        if not text or not text.strip():
            return text
        return self.translate_batch([text], source=source, target=target)[0]
