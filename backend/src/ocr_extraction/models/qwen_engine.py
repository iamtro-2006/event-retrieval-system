from __future__ import annotations

import logging

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

DEFAULT_OCR_PROMPT = (
    "Extract all text visible in this image exactly as written. "
    "Output each line or text block on its own line. "
    "Do not number lines, do not use markdown, do not translate, "
    "do not describe the image, preserve Vietnamese diacritics exactly. "
    "If there is no text in the image, output exactly: NO_TEXT"
)


def load_qwen_model(
    model_id: str,
    device: str = "cuda",
    dtype: str = "bfloat16",
    logger: logging.Logger | None = None,
) -> tuple:
    if logger:
        logger.info(
            "Loading Qwen3-VL model: model_id=%s device=%s dtype=%s",
            model_id,
            device,
            dtype,
        )

    torch_dtype = getattr(torch, dtype, torch.bfloat16)
    model = (
        Qwen3VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch_dtype)
        .to(device)
        .eval()
    )
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
    prompt = prompt or DEFAULT_OCR_PROMPT

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    new_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    decoded = processor.batch_decode(
        new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0].strip()

    if not decoded or decoded == "NO_TEXT":
        return []

    lines = [line.strip() for line in decoded.split("\n")]
    return [line for line in lines if line]
