from __future__ import annotations

import logging

from PIL import Image

_DEFAULT_PROMPT = (
    "Extract all visible text in this image, in reading order, one line of "
    "text per output line. Output only the extracted text, no commentary, "
    "no numbering, no markdown. If there is no visible text, output nothing."
)


def load_qwen_model(
    model_id: str,
    device: str = "cuda",
    dtype: str = "bfloat16",
    logger: logging.Logger | None = None,
):
    """Load a Qwen3-VL model + processor for full-frame OCR.

    Returns:
        (model, processor) tuple.
    """
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if logger:
        logger.info("Loading Qwen3-VL model: model_id=%s device=%s dtype=%s", model_id, device, dtype)

    torch_dtype = getattr(torch, str(dtype), torch.bfloat16)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device,
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id)

    return model, processor


def extract_text_lines(
    model,
    processor,
    image: Image.Image,
    max_new_tokens: int = 512,
    prompt: str | None = None,
    logger: logging.Logger | None = None,
) -> list[str]:
    """Run Qwen3-VL OCR on a single PIL image, returning non-empty text lines."""
    import torch

    prompt = prompt or _DEFAULT_PROMPT

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    try:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        decoded = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]
    except Exception as e:
        if logger:
            logger.warning("Qwen3-VL inference failed on one image: %s", e)
        return []

    lines = [line.strip() for line in decoded.splitlines()]
    return [line for line in lines if line]
