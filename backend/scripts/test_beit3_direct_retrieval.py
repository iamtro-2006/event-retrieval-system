"""Direct BEiT-3 image/text retrieval smoke test.

Usage:
    python scripts/test_beit3_direct_retrieval.py --image-dir D:/path/to/images --top-k 10

Edit QUERY below to test another text query.  This intentionally bypasses
FAISS and the API so it validates only the BEiT-3 image/text embedding space.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Query to test.  Change this directly in the file, as requested.
QUERY = "a man standing next to a water fountain"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embedding_extraction.models.registry import load_model  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image-dir", type=Path)
    source.add_argument("--image-path", type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--query",
        action="append",
        help="Query to test; repeat this option for multiple queries.",
    )
    parser.add_argument(
        "--repo-path",
        default="D:/event-retrieval-system/backend/external/unilm/beit3",
    )
    parser.add_argument(
        "--checkpoint-path",
        default="D:/event-retrieval-system/backend/weights/beit3_large_patch16_384_coco_retrieval.pth",
    )
    parser.add_argument(
        "--spm-path",
        default="D:/event-retrieval-system/backend/weights/beit3.spm",
    )
    return parser.parse_args()


def normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def main() -> int:
    args = parse_args()
    if args.image_path:
        paths = [args.image_path.expanduser().resolve()]
        image_dir = paths[0].parent
    else:
        image_dir = args.image_dir.expanduser().resolve()
        paths = sorted(
            p for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
    if not paths:
        raise SystemExit(f"No supported images found in: {image_dir}")

    queries = args.query or [QUERY]
    loaded = load_model(
        model_name="BEiT-3",
        device_name=args.device,
        precision="fp32",
        beit3={
            "repo_path": args.repo_path,
            "model_arch": "beit3_large_patch16_384_retrieval",
            "checkpoint_path": args.checkpoint_path,
            "input_size": 384,
            "spm_path": args.spm_path,
            "max_text_len": 64,
        },
    )
    if not loaded.supports_text:
        raise RuntimeError(
            "BEiT-3 text encoder is disabled. Check --spm-path and the loader warning."
        )

    model, preprocess, device = loaded.model, loaded.preprocess, loaded.device
    image_vectors: list[torch.Tensor] = []
    valid_paths: list[Path] = []

    with torch.inference_mode():
        for start in range(0, len(paths), max(1, args.batch_size)):
            batch_paths = paths[start : start + args.batch_size]
            tensors = []
            tensor_paths = []
            for path in batch_paths:
                try:
                    with Image.open(path) as image:
                        tensors.append(preprocess(image.convert("RGB")))
                    tensor_paths.append(path)
                except Exception as exc:
                    print(f"[skip] {path}: {type(exc).__name__}: {exc}")
            if tensors:
                vectors = normalize(model.encode_image(torch.stack(tensors).to(device)))
                image_vectors.append(vectors.cpu())
                valid_paths.extend(tensor_paths)

        text_vectors = normalize(model.encode_text(queries))

    matrix = torch.cat(image_vectors, dim=0)
    print(f"Model: BEiT-3 | device: {device} | images: {len(valid_paths)}")
    for query, text_vector in zip(queries, text_vectors):
        scores = (matrix @ text_vector.cpu()).numpy()
        order = np.argsort(-scores)[: max(1, args.top_k)]
        print(f"\nQuery: {query!r}")
        print("rank\tscore\timage")
        for rank, index in enumerate(order, start=1):
            print(f"{rank}\t{scores[index]:.6f}\t{valid_paths[index]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
