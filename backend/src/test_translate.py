"""
Interactive CLI to test Hy-MT2-1.8B translation speed (tokens/sec).

Usage:
    python test_speed.py --model "D:/path/to/Hy-MT2-1.8B-2Bit.gguf"

Then type text at the prompt, hit Enter, and see the translation
stream out live along with tokens/sec, TTFT, and total time.

Type 'exit' or 'quit' to stop. Type 'lang:xx' to change target
language code (default: en). e.g. `lang:zh` then `lang:vi`.
"""

import argparse
import time

from llama_cpp import Llama

_LANG_NAMES = {
    "vi": "Vietnamese", "en": "English", "zh": "Chinese", "fr": "French",
    "pt": "Portuguese", "es": "Spanish", "ja": "Japanese", "tr": "Turkish",
    "ru": "Russian", "ar": "Arabic", "ko": "Korean", "th": "Thai",
    "it": "Italian", "de": "German", "ms": "Malay", "id": "Indonesian",
    "hi": "Hindi",
}

PROMPT_TEMPLATE = (
    "Strictly translate the following text into {target_lang}. "
    "Focus on accuracy and naturalness, and avoid any additional commentary. "
    "Context of the translation should be preserved and can be aligned well with the target language and visual expression in CLIP Style Models."
    "Note that you should only output the translated result "
    "without any additional explanation:\n{source_text}"
)

STOP_TOKENS = ["<|im_start|>", "<|im_end|>"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to .gguf file")
    ap.add_argument("--n_ctx", type=int, default=2048)
    ap.add_argument("-ngl", "--n_gpu_layers", type=int, default=0)
    ap.add_argument("--n_threads", type=int, default=None)
    ap.add_argument("--max_tokens", type=int, default=256)
    args = ap.parse_args()

    print(f"Loading model: {args.model} ...")
    t0 = time.perf_counter()
    llm = Llama(
        model_path=args.model,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        n_threads=args.n_threads,
        verbose=False,
        chat_format="chatml",
    )
    print(f"Model loaded in {time.perf_counter() - t0:.2f}s\n")

    target = "en"
    print("Type text to translate. Commands: 'lang:xx' to switch target, 'exit' to quit.\n")

    while True:
        try:
            text = input(f"[{target}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not text:
            continue
        if text.lower() in ("exit", "quit"):
            break
        if text.lower().startswith("lang:"):
            code = text.split(":", 1)[1].strip().lower()
            if code in _LANG_NAMES:
                target = code
                print(f"-> target language set to {_LANG_NAMES[code]} ({code})\n")
            else:
                print(f"-> unknown code, available: {', '.join(_LANG_NAMES)}\n")
            continue

        target_lang = _LANG_NAMES.get(target, target)
        prompt = PROMPT_TEMPLATE.format(target_lang=target_lang, source_text=text)
        messages = [{"role": "user", "content": prompt}]

        first_token_time = None
        token_count = 0
        full_text = ""

        t_start = time.perf_counter()
        stream = llm.create_chat_completion(
            messages=messages,
            temperature=0.2,
            top_p=0.6,
            top_k=20,
            repeat_penalty=1.05,
            max_tokens=args.max_tokens,
            stop=STOP_TOKENS,
            stream=True,
        )

        print("-> ", end="", flush=True)
        for chunk in stream:
            delta = chunk["choices"][0]["delta"].get("content", "")
            if delta:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                token_count += 1
                full_text += delta
                print(delta, end="", flush=True)
        t_end = time.perf_counter()
        print()

        total_time = t_end - t_start
        ttft = (first_token_time - t_start) if first_token_time else total_time
        gen_time = t_end - first_token_time if first_token_time else 1e-9
        tps = token_count / gen_time if gen_time > 0 else 0

        print(
            f"   [tokens={token_count} | ttft={ttft:.2f}s | "
            f"total={total_time:.2f}s | speed={tps:.2f} tok/s]\n"
        )


if __name__ == "__main__":
    main()