from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND_DIR))

from src.color_retrieval.extraction import extract_metadata


def main():
    parser = argparse.ArgumentParser(description="Extract a 5x5 dominant-colour index for keyframe metadata")
    parser.add_argument("--metadata", required=True, help="FAISS metadata.csv path")
    parser.add_argument("--output", required=True, help="Output .npz path")
    parser.add_argument("--keyframes-root", default=None)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    result = extract_metadata(args.metadata, args.output, args.keyframes_root, args.workers)
    print(json.dumps({**result, "failures": len(result["failures"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
