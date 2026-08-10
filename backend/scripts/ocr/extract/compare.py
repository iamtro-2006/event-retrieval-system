from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two per-video OCR JSON outputs (e.g. ocr/L21_V001.json vs ocr_test/L21_V001.json)."
    )
    parser.add_argument("baseline", type=Path, help="Path to the original result, e.g. ../data/ocr/L21/L21_V001.json")
    parser.add_argument("candidate", type=Path, help="Path to the new result, e.g. ../data/ocr_test/L21/L21_V001.json")
    parser.add_argument("--show-diffs", type=int, default=10, help="Max number of differing keyframes to print (default 10).")
    return parser.parse_args()


def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    base = load(args.baseline)
    cand = load(args.candidate)

    base_ids = set(base.keys())
    cand_ids = set(cand.keys())

    only_base = base_ids - cand_ids
    only_cand = cand_ids - base_ids
    common = base_ids & cand_ids

    n_diff_texts = 0
    n_same_texts = 0
    shown = 0

    for kf_id in sorted(common):
        base_texts = base[kf_id][1]
        cand_texts = cand[kf_id][1]
        if base_texts == cand_texts:
            n_same_texts += 1
            continue
        n_diff_texts += 1
        if shown < args.show_diffs:
            print(f"--- {kf_id} ---")
            print(f"  baseline ({len(base_texts)}): {base_texts}")
            print(f"  candidate ({len(cand_texts)}): {cand_texts}")
            shown += 1

    print("")
    print(f"baseline keyframes   : {len(base_ids)}")
    print(f"candidate keyframes  : {len(cand_ids)}")
    print(f"common keyframes     : {len(common)}")
    print(f"only in baseline     : {len(only_base)}")
    print(f"only in candidate    : {len(only_cand)}")
    print(f"identical texts      : {n_same_texts}")
    print(f"different texts      : {n_diff_texts}")
    if n_diff_texts > args.show_diffs:
        print(f"(showed first {args.show_diffs} diffs, use --show-diffs to see more)")


if __name__ == "__main__":
    main()
