"""One-off utility: zero-pad numeric keyframe filenames (e.g. 12.jpg -> 000012.jpg)
so they sort and join correctly with metadata that expects fixed-width IDs.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def pad_keyframe_filenames(root: Path, width: int = 6) -> None:
    for img in root.rglob("*.jpg"):
        stem = img.stem

        if stem.isdigit() and len(stem) < width:
            new_path = img.with_name(f"{int(stem):0{width}d}{img.suffix}")

            if new_path.exists():
                print(f"[SKIPPED] Target already exists: {new_path} (Source: {img})")
                continue

            img.rename(new_path)
            print(img, "->", new_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-pad numeric keyframe image filenames in place.")
    parser.add_argument("--root", default="data/processed/keyframes", help="Directory to scan recursively.")
    parser.add_argument("--width", type=int, default=6, help="Target zero-padded filename width.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[2] / root
    pad_keyframe_filenames(root, args.width)


if __name__ == "__main__":
    main()
