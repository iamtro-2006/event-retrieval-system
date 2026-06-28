from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from src.index.retrieval_backend import create_retrieval_backend


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_queries(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_result_ids(df) -> set[str]:
    if df.empty:
        return set()
    ids = set()
    for _, row in df.iterrows():
        vid = str(row.get("video_id", ""))
        kid = str(row.get("keyframe_id", row.get("keyframe_id_int", "")))
        ids.add(f"{vid}/{kid}")
    return ids


def extract_result_ranking(df) -> list[str]:
    if df.empty:
        return []
    ids = []
    for _, row in df.iterrows():
        vid = str(row.get("video_id", ""))
        kid = str(row.get("keyframe_id", row.get("keyframe_id_int", "")))
        ids.append(f"{vid}/{kid}")
    return ids


def recall_at_k(faiss_ids: set[str], milvus_ids: set[str], k: int) -> float:
    if not faiss_ids:
        return 1.0
    top_faiss = set(list(faiss_ids)[:k])
    top_milvus = set(list(milvus_ids)[:k])
    if not top_faiss:
        return 1.0
    return len(top_faiss & top_milvus) / len(top_faiss)


def kendall_tau(rank_a: list[str], rank_b: list[str]) -> float:
    n = min(len(rank_a), len(rank_b))
    if n < 2:
        return 1.0
    a = rank_a[:n]
    b = rank_b[:n]
    common = list(set(a) & set(b))
    if len(common) < 2:
        return 1.0
    pos_a = {x: i for i, x in enumerate(a)}
    pos_b = {x: i for i, x in enumerate(b)}
    concordant = 0
    discordant = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            x, y = common[i], common[j]
            da = pos_a[x] - pos_a[y]
            db = pos_b[x] - pos_b[y]
            if da * db > 0:
                concordant += 1
            elif da * db < 0:
                discordant += 1
    total = concordant + discordant
    if total == 0:
        return 1.0
    return (concordant - discordant) / total


def run_comparison(
    faiss_cfg: dict, milvus_cfg: dict, queries: list[dict], top_k: int = 20
) -> list[dict]:
    print("[VERIFY] Loading FAISS backend...")
    faiss_backend = create_retrieval_backend(faiss_cfg)

    print("[VERIFY] Loading Milvus backend...")
    milvus_backend = create_retrieval_backend(milvus_cfg)

    results = []
    for i, q in enumerate(queries):
        query = q["query"]
        mode = q.get("mode", "semantic")
        q_top_k = q.get("top_k", top_k)

        print(f"[VERIFY] Query {i + 1}/{len(queries)}: '{query}' ({mode})")

        faiss_df, _ = faiss_backend.run_search(query, mode=mode, top_k=q_top_k)
        milvus_df, _ = milvus_backend.run_search(query, mode=mode, top_k=q_top_k)

        faiss_ids = extract_result_ids(faiss_df)
        milvus_ids = extract_result_ids(milvus_df)

        faiss_rank = extract_result_ranking(faiss_df)
        milvus_rank = extract_result_ranking(milvus_df)

        r20 = recall_at_k(faiss_ids, milvus_ids, 20)
        r50 = recall_at_k(faiss_ids, milvus_ids, min(50, len(faiss_ids)))
        kt = kendall_tau(faiss_rank, milvus_rank)

        result = {
            "query": query,
            "mode": mode,
            "faiss_count": len(faiss_ids),
            "milvus_count": len(milvus_ids),
            "recall_at_20": round(r20, 4),
            "recall_at_50": round(r50, 4),
            "kendall_tau": round(kt, 4),
        }
        results.append(result)
        print(f"  recall@20={r20:.4f}  recall@50={r50:.4f}  kendall_tau={kt:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Verify recall parity between FAISS and Milvus backends"
    )
    parser.add_argument("--faiss-config", type=str, default="configs/app.yaml.faiss")
    parser.add_argument("--milvus-config", type=str, default="configs/app.yaml")
    parser.add_argument("--queries", type=str, default="scripts/benchmark_queries.json")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    faiss_cfg = load_yaml(Path(args.faiss_config))
    faiss_cfg["backend"] = "faiss"
    milvus_cfg = load_yaml(Path(args.milvus_config))
    milvus_cfg["backend"] = "milvus"

    queries = load_queries(Path(args.queries))

    print(f"[VERIFY] Running {len(queries)} queries against both backends")
    results = run_comparison(faiss_cfg, milvus_cfg, queries, args.top_k)

    avg_r20 = np.mean([r["recall_at_20"] for r in results])
    avg_r50 = np.mean([r["recall_at_50"] for r in results])
    avg_kt = np.mean([r["kendall_tau"] for r in results])

    print("\n" + "=" * 60)
    print("RECALL PARITY REPORT")
    print("=" * 60)
    print(f"{'Query':<50} {'R@20':>6} {'R@50':>6} {'KT':>6}")
    print("-" * 60)
    for r in results:
        print(
            f"{r['query'][:50]:<50} {r['recall_at_20']:>6.4f} {r['recall_at_50']:>6.4f} {r['kendall_tau']:>6.4f}"
        )
    print("-" * 60)
    print(f"{'AVERAGE':<50} {avg_r20:>6.4f} {avg_r50:>6.4f} {avg_kt:>6.4f}")
    print("=" * 60)

    passed = avg_r20 >= 0.95 and avg_kt >= 0.9
    print(f"\nCUTOVER GATE: {'PASSED' if passed else 'FAILED'}")
    print("  Target: recall@20 >= 0.95, kendall_tau >= 0.90")
    print(f"  Actual: recall@20 = {avg_r20:.4f}, kendall_tau = {avg_kt:.4f}")

    if args.output:
        output_path = Path(args.output)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "results": results,
                    "summary": {
                        "avg_recall_at_20": round(float(avg_r20), 4),
                        "avg_recall_at_50": round(float(avg_r50), 4),
                        "avg_kendall_tau": round(float(avg_kt), 4),
                        "passed": passed,
                    },
                },
                f,
                indent=2,
            )
        print(f"\nReport saved to: {output_path}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
