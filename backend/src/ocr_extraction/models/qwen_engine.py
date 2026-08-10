from __future__ import annotations

import base64
import io
import logging

from PIL import Image

try:
    from huggingface_hub import hf_hub_download
except ImportError as _hf_err:  # pragma: no cover
    hf_hub_download = None
    _HF_IMPORT_ERROR = _hf_err
else:
    _HF_IMPORT_ERROR = None

try:
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Qwen3VLChatHandler
except ImportError as _llama_err:  # pragma: no cover
    Llama = None
    Qwen3VLChatHandler = None
    _LLAMA_IMPORT_ERROR = _llama_err
else:
    _LLAMA_IMPORT_ERROR = None


DEFAULT_OCR_PROMPT = (
    "Extract all text visible in this image exactly as written. "
    "Output each line or text block on its own line. "
    "Do not number lines, do not use markdown, do not translate, "
    "do not describe the image, preserve Vietnamese diacritics exactly. "
    "If there is no text in the image, output exactly: NO_TEXT"
)


def _image_to_data_uri(image: Image.Image) -> str:
    """Encode a PIL image as a base64 data URI llama.cpp's vision chat
    handler can consume via the OpenAI-style `image_url` content part."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def load_qwen_model(
    repo_id: str,
    filename: str,
    mmproj_filename: str,
    local_dir: str | None = None,
    n_ctx: int = 4096,
    n_gpu_layers: int = -1,
    logger: logging.Logger | None = None,
) -> tuple:
    """Load the Qwen3-VL-Instruct-Uncensored GGUF quant + its mmproj vision
    projector via llama-cpp-python and return a ready-to-use (llm, chat_handler)
    pair.

    repo_id / filename / mmproj_filename point at a GGUF repo such as
    "mradermacher/Qwen3-VL-4B-Instruct-Uncensored-GGUF" (main-model quant +
    "*.mmproj-*.gguf" vision projector, both downloaded and cached locally
    via huggingface_hub). This is a GGUF model, so it is NOT loaded through
    `transformers` -- it is run locally through llama.cpp's multimodal
    (mtmd) support.
    """
    if hf_hub_download is None:
        raise ImportError(
            "huggingface_hub is required to download the Qwen3-VL GGUF files "
            "(pip install huggingface_hub)."
        ) from _HF_IMPORT_ERROR

    if Llama is None or Qwen3VLChatHandler is None:
        raise ImportError(
            "llama-cpp-python with Qwen3-VL vision (chat) support is required "
            "for engine='qwen' (this is a GGUF model, loaded through llama.cpp's "
            "multimodal/mtmd support, not `transformers`). If `Qwen3VLChatHandler` "
            "is missing from `llama_cpp.llama_chat_format`, your installed "
            "llama-cpp-python build predates Qwen3-VL support -- upgrade to a "
            "recent build that ships it (e.g. a build tracking "
            "https://github.com/JamePeng/llama-cpp-python, which documents "
            "`Qwen3VLChatHandler`)."
        ) from _LLAMA_IMPORT_ERROR

    if logger:
        logger.info(
            "Downloading Qwen3-VL GGUF: repo_id=%s filename=%s mmproj=%s",
            repo_id,
            filename,
            mmproj_filename,
        )

    model_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
    mmproj_path = hf_hub_download(repo_id=repo_id, filename=mmproj_filename, local_dir=local_dir)

    if logger:
        logger.info(
            "Loading Qwen3-VL GGUF: model=%s mmproj=%s n_ctx=%d n_gpu_layers=%d",
            model_path,
            mmproj_path,
            n_ctx,
            n_gpu_layers,
        )

    chat_handler = Qwen3VLChatHandler(clip_model_path=mmproj_path)
    llm = Llama(
        model_path=model_path,
        chat_handler=chat_handler,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )
    return llm, chat_handler


def extract_text_lines(
    llm,
    chat_handler,
    image: Image.Image,
    max_new_tokens: int = 512,
    prompt: str | None = None,
    temperature: float = 0.0,
    logger: logging.Logger | None = None,
) -> list[str]:
    """Run one full-frame OCR pass through the loaded Qwen3-VL GGUF model.

    `chat_handler` is accepted for symmetry with the (model, processor) pair
    returned by load_qwen_model(), but is not used directly here -- it is
    already registered on `llm` (llama.cpp attaches the vision projector
    at construction time).
    """
    del chat_handler  # unused: already bound to llm via chat_handler=

    prompt = prompt or DEFAULT_OCR_PROMPT
    data_uri = _image_to_data_uri(image)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    try:
        result = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
    except Exception as e:
        if logger:
            logger.warning("Qwen3-VL inference failed on one frame: %s", e)
        return []

    decoded = (result["choices"][0]["message"]["content"] or "").strip()

    if not decoded or decoded == "NO_TEXT":
        return []

    lines = [line.strip() for line in decoded.split("\n")]
    return [line for line in lines if line]
