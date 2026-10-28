"""Batched InternVL inference; vLLM owns preprocessing, KV cache and scheduling."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

CAPTION_FRAME_FILTER_VERSION = "rgb8-full-before-resize-lt-v1"


def occupied_color_bins(rgb) -> int:
    """Full-frame 8x8x8 RGB histogram, [0,256), as in the keyframe quality gate.

    Include channel value 255. Count before resizing so interpolation cannot
    turn a flat/few-color frame into an apparently informative frame.
    """
    import cv2
    histogram = cv2.calcHist([rgb], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
    return int((histogram > 0).sum())


VLLM_DEFAULTS = {
    "gpu_memory_utilization": 0.88,
    "max_model_len": 6144,
    "tensor_parallel_size": "auto",
    "max_num_seqs": "auto",
    "max_num_batched_tokens": "auto",
    "enable_prefix_caching": True,
    "enforce_eager": False,
}


def validate_settings(model: str, max_frames: int, batch_size: int | str, settings: dict) -> dict:
    if "internvl" not in model.lower() or model.lower().endswith("-hf"):
        raise ValueError("Caption requires GitHub-format InternVL, e.g. OpenGVLab/InternVL3_5-2B")
    unknown = settings.keys() - VLLM_DEFAULTS.keys()
    if unknown:
        raise ValueError(f"Unknown caption vLLM settings: {sorted(unknown)}")
    result = {**VLLM_DEFAULTS, **settings}
    auto_keys = {"caption_batch_size", "tensor_parallel_size", "max_num_seqs", "max_num_batched_tokens"}
    for key, value in {"caption_max_frames": max_frames, "caption_batch_size": batch_size,
                       **{k: result[k] for k in ("max_model_len", "tensor_parallel_size", "max_num_seqs", "max_num_batched_tokens")}}.items():
        if key in auto_keys and value == "auto":
            continue
        if type(value) is not int or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if not 0 < result["gpu_memory_utilization"] < 1:
        raise ValueError("gpu_memory_utilization must be between 0 and 1")
    if any(type(result[key]) is not bool for key in ("enable_prefix_caching", "enforce_eager")):
        raise ValueError("enable_prefix_caching/enforce_eager must be boolean")
    return result


def resolve_gpu_settings(gpu_count: int, settings: dict) -> dict:
    """Use up to two visible GPUs: TP=1/2, concurrency=3/6, queued batch=6/12."""
    if gpu_count < 1:
        raise RuntimeError("No visible CUDA GPU; enable a GPU accelerator on Kaggle")
    resolved = dict(settings)
    tp = min(gpu_count, 2) if resolved["tensor_parallel_size"] == "auto" else resolved["tensor_parallel_size"]
    if tp not in (1, 2) or tp > gpu_count:
        raise ValueError(f"tensor_parallel_size={tp} requires 1 or 2 visible GPUs; detected {gpu_count}")
    resolved["tensor_parallel_size"] = tp
    if resolved["max_num_seqs"] == "auto":
        resolved["max_num_seqs"] = 3 * tp
    if resolved["max_num_batched_tokens"] == "auto":
        resolved["max_num_batched_tokens"] = 4096 * tp
    return resolved


def prepared_batches(windows: list[dict], batch_size: int, prepare):
    """One CPU producer, at most one batch ahead; never share a reader concurrently."""
    def timed_prepare(offset):
        started = perf_counter()
        batch = windows[offset:offset + batch_size]
        return offset, batch, prepare(offset, batch), perf_counter() - started

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="caption-decode") as executor:
        future = executor.submit(timed_prepare, 0) if windows else None
        next_offset = batch_size
        while future is not None:
            result = future.result()
            future = (executor.submit(timed_prepare, next_offset)
                      if next_offset < len(windows) else None)
            next_offset += batch_size
            yield result
            del result


class InternVLCaptioner:
    def __init__(self, model: str, max_frames: int, max_tokens: int, settings: dict, logger):
        # OmniShotCut has already initialized CUDA; fork is unsafe here.
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        import torch
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        if not torch.cuda.is_available():
            raise RuntimeError("InternVL caption inference requires an NVIDIA CUDA GPU")
        self.settings = settings = resolve_gpu_settings(torch.cuda.device_count(), settings)
        dtype = "bfloat16" if torch.cuda.is_bf16_supported(including_emulation=False) else "float16"
        logger.info("[CAPTION] Loading %s with vLLM dtype=%s settings=%s", model, dtype, settings)
        self.engine = LLM(
            model=model, trust_remote_code=True, dtype=dtype,
            limit_mm_per_prompt={"image": max_frames, "video": 0},
            mm_processor_kwargs={"min_dynamic_patch": 1, "max_dynamic_patch": 1,
                                 "dynamic_image_size": False},
            mm_processor_cache_gb=0, enable_chunked_prefill=True,
            generation_config="vllm", seed=0, **settings,
        )
        # This tokenizer is private to the CPU producer. vLLM may tokenize/process
        # the current batch concurrently; sharing a fast tokenizer risks borrowing errors.
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        config = self.engine.llm_engine.model_config.hf_config
        self.image_size = int(getattr(config, "force_image_size", None) or config.vision_config.image_size)
        self.image_tokens = int((self.image_size // config.vision_config.patch_size) ** 2
                                * config.downsample_ratio ** 2)
        self.max_model_len = settings["max_model_len"]
        self.max_frames = max_frames
        self.max_tokens = max_tokens
        self.logger = logger
        self.sampling = SamplingParams(temperature=0, max_tokens=max_tokens, n=1,
                                       stop=["<|im_end|>"], skip_special_tokens=True)
        logger.info("[CAPTION] vLLM ready tile=%dx%d tokens/frame=%d max_frames=%d; one tile/frame, thinking disabled",
                    self.image_size, self.image_size, self.image_tokens, max_frames)

    def request(self, images, timestamps, prompt: str) -> dict:
        if not images or len(images) != len(timestamps) or len(images) > self.max_frames:
            raise ValueError("InternVL frame/timestamp counts are invalid")
        frames = "".join(f"Frame{i} ({timestamp:.3f}s): <image>\n"
                         for i, timestamp in enumerate(timestamps, 1))
        formatted = self.tokenizer.apply_chat_template([
            {"role": "system", "content": "You are a precise video captioner."},
            {"role": "user", "content": frames + prompt},
        ], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        # Upper bound includes wrappers added by the multimodal processor.
        needed = len(self.tokenizer.encode(formatted, add_special_tokens=False)) + len(images) * (self.image_tokens + 8) + self.max_tokens
        if needed > self.max_model_len:
            raise ValueError(f"Caption needs about {needed} tokens; increase vllm.max_model_len "
                             "or reduce caption_max_frames/prompt/max_new_tokens")
        return {"prompt": formatted, "multi_modal_data": {"image": images}}

    def generate(self, requests: list[dict]) -> tuple[list[str], int]:
        outputs = self.engine.generate(requests, self.sampling, use_tqdm=False)
        # LLM.generate guarantees input order, even if requests finish out of order.
        if len(outputs) != len(requests):
            raise RuntimeError("vLLM returned an incomplete caption batch")
        captions, token_count = [], 0
        for output in outputs:
            if not output.outputs or not output.outputs[0].text.strip():
                raise RuntimeError("vLLM returned an empty caption; batch was not saved")
            completion = output.outputs[0]
            if getattr(completion, "finish_reason", None) == "length":
                self.logger.warning("[CAPTION] Output reached max_new_tokens=%d; consider increasing it", self.max_tokens)
            captions.append(completion.text.strip())
            token_count += len(completion.token_ids)
        return captions, token_count
