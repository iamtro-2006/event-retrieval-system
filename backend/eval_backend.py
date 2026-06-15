from __future__ import annotations

import time
import statistics
from pathlib import Path
from collections import defaultdict

print("[BOOT] Starting eval_backend.py ...", flush=True)

from main import (
    retrieval_system,
    translate_query_if_needed,
)

print("[BOOT] Imported backend successfully", flush=True)


OUTPUT_PATH = Path("./evaluation_results.txt")

TOP_K = 120
CANDIDATE_MULTIPLIER = 3
DURATION_LIMIT = -1


SEMANTIC_QUERIES = [
    {
        "en": "woman in beige jacket, brown hair, clothes racks, shoes on shelf",
        "vi": "phụ nữ áo khoác màu be, tóc nâu, giá treo quần áo, giày trên kệ",
    },
    {
        "en": "man cooking in kitchen, vegetables on table, frying pan",
        "vi": "người đàn ông nấu ăn trong bếp, rau củ trên bàn, chảo chiên",
    },
]


TEMPORAL_QUERIES = [
    # 2 scenes
    {
        "scenes": 2,
        "en": "a woman in a beige jacket, clothes racks; shoes arranged on a shoe shelf",
        "vi": "một phụ nữ áo khoác màu be, các giá treo đồ; những đôi giày được xếp trên kệ giày",
    },
    {
        "scenes": 2,
        "en": "a man, refrigerator; then he takes out a bottle of water",
        "vi": "đàn ông, tủ lạnh; sau đó anh ấy lấy ra một chai nước",
    },

    # 3 scenes
    {
        "scenes": 3,
        "en": "a man enters a room; then he turns on the light; then he sits on a sofa",
        "vi": "một người đàn ông bước vào phòng; sau đó anh ấy bật đèn; sau đó anh ấy ngồi xuống ghế sofa",
    },
    {
        "scenes": 3,
        "en": "a woman takes a book from a shelf; then she opens it; then she starts reading",
        "vi": "một người phụ nữ lấy sách từ kệ, sau đó cô ấy mở sách, sau đó cô ấy bắt đầu đọc",
    },
 
    # 5 scenes
    {
        "scenes": 5,
        "en": "a woman walks along a street; then stops at a window display; then points at a dress; then talks to a friend; then enters the store",
        "vi": "một người phụ nữ đi dọc con phố; sau đó dừng trước tủ kính trưng bày; sau đó chỉ vào một chiếc váy; sau đó nói chuyện với bạn; sau đó bước vào cửa hàng",
    },
    {
        "scenes": 5,
        "en": "a man takes a camera, then points it at a building; then adjusts the lens; then takes a photo; then checks the image",
        "vi": "một người đàn ông cầm máy ảnh; sau đó hướng máy ảnh vào một tòa nhà; sau đó chỉnh ống kính; sau đó chụp ảnh; sau đó kiểm tra ảnh",
    },
]


def fmt_table(headers, rows):
    widths = [len(str(h)) for h in headers]

    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(str(value)))

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"

    lines = [sep]
    lines.append("| " + " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)) + " |")
    lines.append(sep)

    for row in rows:
        lines.append("| " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(row)) + " |")

    lines.append(sep)
    return "\n".join(lines)


def run_once(query, *, language, mode, translate, use_split=True):
    t0 = time.perf_counter()

    t_translate0 = time.perf_counter()
    search_query = translate_query_if_needed(query, translate)
    t_translate1 = time.perf_counter()

    resolved_mode = mode

    t_search0 = time.perf_counter()
    df, query_plan = retrieval_system.run_search(
        query=search_query,
        mode=resolved_mode,
        use_split=use_split,
        top_k=TOP_K,
        candidate_multiplier=CANDIDATE_MULTIPLIER,
        duration_limit=DURATION_LIMIT,
    )
    t_search1 = time.perf_counter()

    t1 = time.perf_counter()

    split_ms = 0.0  # split đã nằm trong retrieval_system.run_search()

    return {
        "language": language,
        "mode": mode,
        "query": query,
        "search_query": search_query,
        "total_ms": (t1 - t0) * 1000,
        "translate_ms": (t_translate1 - t_translate0) * 1000,
        "split_ms": split_ms,
        "search_ms": (t_search1 - t_search0) * 1000,
        "count": len(df),
        "events": query_plan.events,
        "event_queries": query_plan.event_queries,
        "sub_queries": query_plan.flat_queries,
        "num_sub_queries": len(query_plan.flat_queries),
        "num_events": len(query_plan.events),
    }

def warmup():
    logs = []
    logs.append("========== WARM-UP ==========")

    warmup_query = "person walking in a room, indoor scene"

    t0 = time.perf_counter()
    df, query_plan = retrieval_system.run_search(
        query=warmup_query,
        mode="semantic",
        use_split=False,
        top_k=TOP_K,
        candidate_multiplier=CANDIDATE_MULTIPLIER,
        duration_limit=DURATION_LIMIT,
    )
    t1 = time.perf_counter()

    logs.append("Warm-up mode  : semantic")
    logs.append(f"Warm-up query : {warmup_query}")
    logs.append(f"Events        : {query_plan.events}")
    logs.append(f"Sub queries   : {query_plan.flat_queries}")
    logs.append(f"Result count  : {len(df)}")
    logs.append(f"Warm-up time  : {(t1 - t0) * 1000:.2f} ms")
    logs.append("Warm-up is excluded from benchmark.")
    logs.append("")

    return logs

def summarize(rows):
    logs = []

    grouped = defaultdict(list)
    for row in rows:
        if row["mode"] == "semantic":
            key = (row["mode"], row["language"], "all")
        else:
            key = (row["mode"], row["language"], row["scenes"])
        grouped[key].append(row)

    summary_rows = []

    for key, items in sorted(grouped.items(), key=lambda x: str(x[0])):
        mode, language, scenes = key

        total_values = [x["total_ms"] for x in items]
        search_values = [x["search_ms"] for x in items]
        translate_values = [x["translate_ms"] for x in items]
        split_values = [x["split_ms"] for x in items]

        summary_rows.append([
            mode,
            language.upper(),
            scenes,
            len(items),
            f"{statistics.mean(total_values):.2f}",
            f"{statistics.mean(search_values):.2f}",
            f"{statistics.mean(translate_values):.2f}",
            f"{statistics.mean(split_values):.4f}",
            f"{min(total_values):.2f}",
            f"{max(total_values):.2f}",
        ])

    logs.append("========== SUMMARY ==========")
    logs.append(fmt_table(
        [
            "mode",
            "lang",
            "scenes",
            "n",
            "avg_total_ms",
            "avg_search_ms",
            "avg_translate_ms",
            "avg_split_ms",
            "min_total_ms",
            "max_total_ms",
        ],
        summary_rows,
    ))
    logs.append("")

    compare_rows = []

    summary_map = {}

    for key, items in grouped.items():
        total_values = [x["total_ms"] for x in items]
        search_values = [x["search_ms"] for x in items]
        translate_values = [x["translate_ms"] for x in items]

        summary_map[key] = {
            "total": statistics.mean(total_values),
            "search": statistics.mean(search_values),
            "translate": statistics.mean(translate_values),
        }

    semantic_en = summary_map.get(("semantic", "en", "all"))
    semantic_vi = summary_map.get(("semantic", "vi", "all"))

    if semantic_en and semantic_vi:
        compare_rows.append([
            "semantic",
            "all",
            f"{semantic_vi['total'] - semantic_en['total']:.2f}",
            f"{semantic_vi['search'] - semantic_en['search']:.2f}",
            f"{semantic_vi['translate'] - semantic_en['translate']:.2f}",
            f"{semantic_vi['total'] / semantic_en['total']:.2f}x",
        ])

    for scenes in [2, 3, 4, 5]:
        en = summary_map.get(("temporal", "en", scenes))
        vi = summary_map.get(("temporal", "vi", scenes))

        if not en or not vi:
            continue

        compare_rows.append([
            "temporal",
            scenes,
            f"{vi['total'] - en['total']:.2f}",
            f"{vi['search'] - en['search']:.2f}",
            f"{vi['translate'] - en['translate']:.2f}",
            f"{vi['total'] / en['total']:.2f}x",
        ])

    logs.append("========== VI VS EN ==========")
    logs.append(fmt_table(
        [
            "mode",
            "scenes",
            "vi_minus_en_total_ms",
            "vi_minus_en_search_ms",
            "vi_minus_en_translate_ms",
            "vi_over_en_total",
        ],
        compare_rows,
    ))
    logs.append("")

    return logs


def log_line(logs: list[str], line: str = ""):
    logs.append(line)
    print(line, flush=True)


def main():
    print("[MAIN] Start benchmark", flush=True)

    logs = []

    log_line(logs, "========== CONFIG ==========")
    log_line(logs, f"TOP_K                : {TOP_K}")
    log_line(logs, f"CANDIDATE_MULTIPLIER : {CANDIDATE_MULTIPLIER}")
    log_line(logs, f"DURATION_LIMIT       : {DURATION_LIMIT}")
    log_line(logs, f"OUTPUT_PATH          : {OUTPUT_PATH.resolve()}")
    log_line(logs, "")

    print("[WARMUP] Running semantic warm-up...", flush=True)
    logs.extend(warmup())
    print("[WARMUP] Done", flush=True)

    results = []

    log_line(logs, "")
    log_line(logs, "========== SEMANTIC SEARCH BENCHMARK ==========")

    for idx, item in enumerate(SEMANTIC_QUERIES, start=1):
        print(f"[SEM {idx:02d}] EN running...", flush=True)

        en_row = {
            "id": idx,
            "scenes": "semantic",
            **run_once(
                item["en"],
                language="en",
                mode="semantic",
                translate=False,
            ),
        }
        results.append(en_row)

        line = (
            f"[SEM {idx:02d}] EN done | "
            f"total={en_row['total_ms']:.2f} ms | "
            f"search={en_row['search_ms']:.2f} ms | "
            f"translate={en_row['translate_ms']:.2f} ms | "
            f"split={en_row['split_ms']:.4f} ms | "
            f"sub_queries={en_row['num_sub_queries']} | "
            f"count={en_row['count']}"
        )
        log_line(logs, line)

        print(f"[SEM {idx:02d}] VI running...", flush=True)

        vi_row = {
            "id": idx,
            "scenes": "semantic",
            **run_once(
                item["vi"],
                language="vi",
                mode="semantic",
                translate=True,
            ),
        }
        results.append(vi_row)

        line = (
            f"[SEM {idx:02d}] VI done | "
            f"total={vi_row['total_ms']:.2f} ms | "
            f"search={vi_row['search_ms']:.2f} ms | "
            f"translate={vi_row['translate_ms']:.2f} ms | "
            f"split={vi_row['split_ms']:.4f} ms | "
            f"sub_queries={vi_row['num_sub_queries']} | "
            f"count={vi_row['count']}"
        )
        log_line(logs, line)

    log_line(logs, "")
    log_line(logs, "========== TEMPORAL SEARCH BENCHMARK ==========")

    for idx, item in enumerate(TEMPORAL_QUERIES, start=1):
        scenes = item["scenes"]

        print(f"[TMP {idx:02d}] EN running... scenes={scenes}", flush=True)

        en_row = {
            "id": idx,
            "scenes": scenes,
            **run_once(
                item["en"],
                language="en",
                mode="temporal",
                translate=False,
            ),
        }
        results.append(en_row)

        line = (
            f"[TMP {idx:02d}] EN done | {scenes} scenes | "
            f"total={en_row['total_ms']:.2f} ms | "
            f"search={en_row['search_ms']:.2f} ms | "
            f"translate={en_row['translate_ms']:.2f} ms | "
            f"split={en_row['split_ms']:.4f} ms | "
            f"events={en_row['num_events']} | "
            f"count={en_row['count']}"
        )
        log_line(logs, line)

        print(f"[TMP {idx:02d}] VI running... scenes={scenes}", flush=True)

        vi_row = {
            "id": idx,
            "scenes": scenes,
            **run_once(
                item["vi"],
                language="vi",
                mode="temporal",
                translate=True,
            ),
        }
        results.append(vi_row)

        line = (
            f"[TMP {idx:02d}] VI done | {scenes} scenes | "
            f"total={vi_row['total_ms']:.2f} ms | "
            f"search={vi_row['search_ms']:.2f} ms | "
            f"translate={vi_row['translate_ms']:.2f} ms | "
            f"split={vi_row['split_ms']:.4f} ms | "
            f"events={vi_row['num_events']} | "
            f"count={vi_row['count']}"
        )
        log_line(logs, line)

    log_line(logs, "")
    print("[SUMMARY] Computing summary...", flush=True)
    logs.extend(summarize(results))
    print("[SUMMARY] Done", flush=True)

    log_line(logs, "========== QUERY SPLIT DETAILS ==========")

    for row in results:
        log_line(logs, "")
        log_line(
            logs,
            f"{row['mode'].upper()} | {row['language'].upper()} | "
            f"id={row['id']} | scenes={row['scenes']}",
        )
        log_line(logs, f"Original query: {row['query']}")
        log_line(logs, f"Search query  : {row['search_query']}")
        log_line(logs, f"Events        : {row['events']}")
        log_line(logs, f"Event queries : {row['event_queries']}")
        log_line(logs, f"Sub queries   : {row['sub_queries']}")

    text = "\n".join(logs)

    print("[SAVE] Writing evaluation file...", flush=True)
    OUTPUT_PATH.write_text(text, encoding="utf-8")
    print(f"[SAVE] Saved results to: {OUTPUT_PATH.resolve()}", flush=True)

if __name__ == "__main__":
    main()