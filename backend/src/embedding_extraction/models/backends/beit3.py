"""BEiT-3 backend.

BEiT-3 is NOT on `transformers` or `open_clip`. It only exists as raw code
+ checkpoint in the microsoft/unilm repo:
    https://github.com/microsoft/unilm/tree/master/beit3

Setup required once, outside this codebase:
  1. git clone https://github.com/microsoft/unilm.git <somewhere>
  2. pip install -r unilm/beit3/requirements.txt   (needs timm==0.4.12, torchscale)
  3. Download an *-itc retrieval checkpoint, e.g. beit3_large_patch16_384_coco_retrieval.pth
     (image-only embedding extraction does not need the beit3.spm sentencepiece
     model - that's only required for tokenizing text queries at search time,
     see `beit3.spm_path` below)

Then point `configs/embeddings.yaml` at it, e.g.:

    - key: beit3_large_itc
      backend: beit3
      name: BEiT3-Large-Retrieval
      device: auto
      precision: fp32
      batch_size: 16
      normalize: true
      beit3:
        repo_path: third_party/unilm/beit3
        model_arch: beit3_large_patch16_384_retrieval
        checkpoint_path: checkpoints/beit3_large_patch16_384_coco_retrieval.pth
        input_size: 384
        spm_path: checkpoints/beit3.spm       # optional - omit for image-only search
        max_text_len: 64                      # optional, default 64

This backend imports `modeling_finetune` from that repo to register the
`beit3_*` architectures with timm, builds the model, loads the checkpoint,
and wraps `BEiT3ForRetrieval.forward(only_infer=True)` so it returns the
L2-normalized `vision_cls`/`language_cls` embeddings - matching the same
`.encode_image(batch)` / `.encode_text(list[str])` contract every other
backend uses (see `backends/base.py`).

Note on `model_arch`: every architecture `unilm/beit3/modeling_finetune.py`
registers with `@register_model` has a task suffix (`_retrieval`,
`_captioning`, `_nlvr2`, ...) - there is no "bare" `beit3_large_patch16_384`.
`timm.create_model("beit3_large_patch16_384")` raises "unknown model" for
that bare name, so `model_arch` must always be one of the suffixed names
(e.g. `beit3_large_patch16_384_retrieval` for image-text retrieval/ITC).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

from src.embedding_extraction.models.backends.base import EncoderWrapper, LoadedModel
from src.utils.device import resolve_device

# BEiT-3 was trained with "inception-style" normalization, not the usual
# ImageNet mean/std. See unilm/beit3/datasets.py -> build_transform().
_BEIT3_MEAN = (0.5, 0.5, 0.5)
_BEIT3_STD = (0.5, 0.5, 0.5)

# Default text sequence length used by unilm/beit3's own finetuning/retrieval
# scripts (see get_started_for_retrieval.md / datasets.py `_get_text_segment`).
_DEFAULT_MAX_TEXT_LEN = 64


def _import_beit3_modeling(repo_path: str):
    """Add the local unilm/beit3 checkout to sys.path and import it once."""
    repo_path = Path(repo_path).expanduser().resolve()
    if not repo_path.exists():
        raise FileNotFoundError(
            f"BEiT-3 repo not found at '{repo_path}'. Clone "
            "https://github.com/microsoft/unilm and point "
            "configs/embeddings.yaml -> beit3.repo_path at its 'beit3' folder."
        )

    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

    try:
        import modeling_finetune  # noqa: F401  (registers beit3_* archs with timm)
    except ImportError as e:
        raise ImportError(
            "Could not import 'modeling_finetune' from the BEiT-3 repo. "
            "Make sure beit3.repo_path points at the 'beit3' folder that "
            "directly contains modeling_finetune.py, and that its "
            "requirements (timm, torchscale, ...) are installed."
        ) from e

    return modeling_finetune


def _build_transform(input_size: int):
    return T.Compose([
        T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=_BEIT3_MEAN, std=_BEIT3_STD),
    ])


def _load_tokenizer(spm_path: str, logger: logging.Logger | None = None):
    """Load the beit3.spm sentencepiece tokenizer via `transformers`'
    `XLMRobertaTokenizer` - this is the exact loading convention documented
    in unilm/beit3/README.md and used by every `datasets.py` builder in that
    repo (retrieval/captioning/vqa)."""
    from transformers import XLMRobertaTokenizer

    spm_path = str(Path(spm_path).expanduser().resolve())
    if not Path(spm_path).exists():
        raise FileNotFoundError(
            f"BEiT-3 sentencepiece model not found at '{spm_path}'. Point "
            "'beit3.spm_path' at the 'beit3.spm' file shipped in the unilm "
            "repo, or omit it to keep this model image-only."
        )
    if logger:
        logger.info("[beit3] loading tokenizer (beit3.spm) from %s", spm_path)
    # transformers <= 4.x accepted a SentencePiece filename here.  In
    # transformers 5 the slow tokenizer constructor expects the already
    # decoded vocabulary (a list of ``(piece, score)`` pairs), and passing the
    # filename produces ``Can't extract str to Vec``.  Build that vocabulary
    # explicitly so BEiT-3 keeps working with both API generations.
    try:
        tokenizer = XLMRobertaTokenizer(spm_path)
        # transformers 5.x may accept the path without raising, but interpret
        # it as a tiny custom vocabulary (vocab_size=5).  That tokenizer is
        # unusable for BEiT-3 and must go through the compatibility path below.
        if tokenizer.vocab_size >= 10000:
            return tokenizer
    except (TypeError, ValueError):
        pass

    import sentencepiece as spm

    processor = spm.SentencePieceProcessor(model_file=spm_path)
    vocab = [
        (processor.id_to_piece(i), float(processor.get_score(i)))
        for i in range(processor.vocab_size())
    ]
    # The legacy slow XLM-R tokenizer used by the official BEiT-3 code
    # exposes the SentencePiece ids as BOS=0 (<unk>), EOS=2 and PAD=1 (<s>).
    tokenizer = XLMRobertaTokenizer(
        vocab=vocab,
        bos_token="<unk>",
        cls_token="<s>",
        pad_token="<s>",
    )
    if tokenizer.vocab_size < 10000:
        raise RuntimeError(
            f"Invalid BEiT-3 tokenizer vocabulary size: {tokenizer.vocab_size}"
        )
    if logger:
        logger.info(
            "[beit3] transformers-5 SentencePiece compatibility path "
            "vocab=%d bos=%d eos=%d pad=%d",
            tokenizer.vocab_size,
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            tokenizer.pad_token_id,
        )
    return tokenizer


def _tokenize_batch(
    tokenizer, texts: list[str], max_text_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize `texts` the same way unilm/beit3/datasets.py's
    `_get_text_segment()` does: `[bos] + ids[:max_len-2] + [eos]`, right-padded
    with `pad_token_id`, with `padding_mask` set at padded positions - the
    convention `BEiT3ForRetrieval`/the shared `torchscale` encoder expects
    (same convention the VQA head in the same `modeling_finetune.py` uses).
    """
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    token_ids: list[list[int]] = []
    for text in texts:
        # Match unilm/beit3/datasets.py exactly: it calls tokenizer.tokenize()
        # (which returns SentencePiece ids for XLMRobertaTokenizer), then
        # adds BOS/EOS itself.  Calling encode() here is subtly different
        # across transformers versions because encode may add/handle special
        # tokens internally.
        pieces_or_ids = tokenizer.tokenize(str(text or ""))
        if not pieces_or_ids:
            raise ValueError("BEiT-3 text query must contain at least one token")
        # transformers 4's XLM-R tokenizer returned ids here in the BEiT-3
        # environment; newer versions return SentencePiece strings.
        ids = (
            tokenizer.convert_tokens_to_ids(pieces_or_ids)
            if isinstance(pieces_or_ids[0], str)
            else pieces_or_ids
        )
        ids = ids[: max_text_len - 2]
        ids = [bos_id, *ids, eos_id]
        token_ids.append(ids)

    # The official retrieval dataset pads every example to exactly
    # num_max_bpe_tokens (64), not to the longest item in the current batch.
    # Keeping this fixed is important for matching the trained text tower.
    padded = torch.full((len(token_ids), max_text_len), pad_id, dtype=torch.long)
    padding_mask = torch.ones((len(token_ids), max_text_len), dtype=torch.long)  # 1 = padded
    for row, ids in enumerate(token_ids):
        padded[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        padding_mask[row, : len(ids)] = 0  # 0 = real token

    return padded, padding_mask


def load(
    model_name: str,
    pretrained: str | None = None,  # unused, kept for interface parity
    precision: str = "fp32",
    device_name: str = "auto",
    beit3: dict | None = None,
    logger: logging.Logger | None = None,
    **_: object,
) -> LoadedModel:
    import timm  # local import: only required when this backend is used

    beit3_cfg = beit3 or {}
    repo_path = beit3_cfg.get("repo_path")
    # NOTE: must be one of the task-suffixed arches registered by
    # modeling_finetune.py (e.g. "*_retrieval") - see module docstring.
    model_arch = beit3_cfg.get("model_arch", "beit3_large_patch16_384_retrieval")
    checkpoint_path = beit3_cfg.get("checkpoint_path")
    input_size = int(beit3_cfg.get("input_size", 384))
    spm_path = beit3_cfg.get("spm_path")
    max_text_len = int(beit3_cfg.get("max_text_len", _DEFAULT_MAX_TEXT_LEN))

    if not repo_path or not checkpoint_path:
        raise ValueError(
            "BEiT-3 backend requires 'beit3.repo_path' and "
            "'beit3.checkpoint_path' in the model config."
        )

    device = resolve_device(device_name)
    if device.type == "cpu" and precision in {"fp16", "amp"}:
        precision = "fp32"

    if logger:
        logger.info(
            "[beit3] loading %s from %s on %s precision=%s",
            model_arch, checkpoint_path, device, precision,
        )

    _import_beit3_modeling(repo_path)

    model = timm.models.create_model(model_arch, pretrained=False, drop_path_rate=0.0)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if logger and (missing or unexpected):
        logger.warning(
            "[beit3] state_dict mismatch - missing=%d unexpected=%d "
            "(expected for text-tower/head keys if this checkpoint isn't "
            "an *-itc retrieval checkpoint)",
            len(missing), len(unexpected),
        )

    model = model.to(device).eval()
    if precision == "fp16" and device.type == "cuda":
        model = model.half()

    transform = _build_transform(input_size)

    def preprocess(image: Image.Image) -> torch.Tensor:
        return transform(image.convert("RGB"))

    @torch.inference_mode()
    def _encode_image(batch: torch.Tensor) -> torch.Tensor:
        if precision == "fp16" and device.type == "cuda":
            batch = batch.half()
        vision_cls, _ = model(image=batch, only_infer=True)
        return vision_cls  # already L2-normalized inside BEiT3ForRetrieval

    embedding_dim = getattr(getattr(model, "vision_head", None), "out_features", None)

    # Text tower: optional, only wired up when `beit3.spm_path` is given in
    # the model config - a checkpoint can still be used image-only (reverse
    # image / temporal-continuation search) without it.
    encode_text_fn = None
    supports_text = False
    if spm_path:
        try:
            tokenizer = _load_tokenizer(spm_path, logger=logger)

            @torch.inference_mode()
            def _encode_text(texts: list[str]) -> torch.Tensor:
                token_ids, padding_mask = _tokenize_batch(tokenizer, list(texts), max_text_len)
                token_ids = token_ids.to(device, non_blocking=True)
                padding_mask = padding_mask.to(device, non_blocking=True)
                # BEiT3ForRetrieval.forward(image=None, text_description=...,
                # only_infer=True) -> (vision_cls, language_cls); vision_cls
                # is None here since we only pass text.
                _, language_cls = model(
                    image=None,
                    text_description=token_ids,
                    padding_mask=padding_mask,
                    only_infer=True,
                )
                return language_cls  # already L2-normalized inside BEiT3ForRetrieval

            encode_text_fn = _encode_text
            supports_text = True
        except Exception as exc:
            if logger:
                logger.warning(
                    "[beit3] text tower disabled (spm_path='%s'): %s: %s - "
                    "this model will only support image-based search.",
                    spm_path, type(exc).__name__, exc,
                )
    elif logger:
        logger.info(
            "[beit3] no 'beit3.spm_path' configured - text search disabled "
            "for this model (image-only: reverse image / temporal continuation)."
        )

    return LoadedModel(
        model=EncoderWrapper(_encode_image, encode_text_fn, backend_label="beit3"),
        preprocess=preprocess,
        device=device,
        precision=precision,
        embedding_dim=embedding_dim,
        supports_text=supports_text,
    )
