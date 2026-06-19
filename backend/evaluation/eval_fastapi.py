from __future__ import annotations

import time
import statistics
from collections import defaultdict

import requests


API_URL = "http://127.0.0.1:8000/api/search"

TOP_K = 120
CANDIDATE_MULTIPLIER = 5
DURATION_LIMIT = -1
REPEAT = 5
TIMEOUT = 120


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


def post_search(payload: dict) -> dict:
    t0 = time.perf_counter()
    response = requests.post(API_URL, json=payload, timeout=TIMEOUT)
    t1 = time.perf_counter()

    round_trip_ms = (t1 - t0) * 1000

    try:
        data = response.json()
    except Exception:
        data = {}

    backend_ms = float(data.get("latency_ms", 0) or 0)

    return {
        "status_code": response.status_code,
        "round_trip_ms": round_trip_ms,
        "backend_ms": backend_ms,
        "network_overhead_ms": round_trip_ms - backend_ms,
        "count": data.get("count", 0),
        "events": data.get("events", []),
        "event_queries": data.get("event_queries", []),
        "sub_queries": data.get("sub_queries", []),
        "error": data.get("detail") if response.status_code != 200 else None,
    }


def make_payload(
    query: str,
    *,
    mode: str,
    use_translate: bool,
    use_split: bool,
) -> dict:
    return {
        "query": query,
        "top_k": TOP_K,
        "candidate_multiplier": CANDIDATE_MULTIPLIER,
        "use_split": use_split,
        "use_translate": use_translate,
        "search_mode": mode,
        "duration_limit": DURATION_LIMIT,
    }


def run_case(
    *,
    case_name: str,
    query: str,
    language: str,
    mode: str,
    scenes: int | str,
    use_translate: bool,
    use_split: bool,
) -> list[dict]:
    rows = []

    payload = make_payload(
        query,
        mode=mode,
        use_translate=use_translate,
        use_split=use_split,
    )

    for i in range(1, REPEAT + 1):
        result = post_search(payload)

        row = {
            "case_name": case_name,
            "repeat": i,
            "language": language,
            "mode": mode,
            "scenes": scenes,
            "use_split": use_split,
            "use_translate": use_translate,
            "query": query,
            **result,
        }

        rows.append(row)

        print(
            f"[{case_name}] run={i:02d} | "
            f"status={row['status_code']} | "
            f"round_trip={row['round_trip_ms']:.2f} ms | "
            f"backend={row['backend_ms']:.2f} ms | "
            f"overhead={row['network_overhead_ms']:.2f} ms | "
            f"count={row['count']}"
        )

    return rows


def summarize(rows: list[dict]) -> None:
    grouped = defaultdict(list)

    for row in rows:
        key = (
            row["mode"],
            row["language"],
            row["scenes"],
            row["use_split"],
            row["use_translate"],
        )
        grouped[key].append(row)

    print("\n========== SUMMARY ==========")

    header = (
        f"{'mode':<10} {'lang':<5} {'scenes':<8} {'split':<7} {'trans':<7} "
        f"{'n':<4} {'avg_rtt':<10} {'avg_backend':<12} {'avg_overhead':<13} "
        f"{'min_rtt':<10} {'max_rtt':<10}"
    )
    print(header)
    print("-" * len(header))

    for key, items in sorted(grouped.items(), key=lambda x: str(x[0])):
        mode, lang, scenes, use_split, use_translate = key

        rtt = [x["round_trip_ms"] for x in items]
        backend = [x["backend_ms"] for x in items]
        overhead = [x["network_overhead_ms"] for x in items]

        print(
            f"{mode:<10} {lang:<5} {str(scenes):<8} {str(use_split):<7} {str(use_translate):<7} "
            f"{len(items):<4} "
            f"{statistics.mean(rtt):<10.2f} "
            f"{statistics.mean(backend):<12.2f} "
            f"{statistics.mean(overhead):<13.2f} "
            f"{min(rtt):<10.2f} "
            f"{max(rtt):<10.2f}"
        )


def main():
    all_rows = []

    print("========== WARMUP ==========")
    warmup_payload = make_payload(
        "person walking in a room, indoor scene",
        mode="semantic",
        use_translate=False,
        use_split=True,
    )
    warmup = post_search(warmup_payload)
    print(
        f"warmup | status={warmup['status_code']} | "
        f"round_trip={warmup['round_trip_ms']:.2f} ms | "
        f"backend={warmup['backend_ms']:.2f} ms"
    )

    print("\n========== SEMANTIC ==========")

    for idx, item in enumerate(SEMANTIC_QUERIES, start=1):
        all_rows.extend(
            run_case(
                case_name=f"semantic_en_{idx}",
                query=item["en"],
                language="en",
                mode="semantic",
                scenes="all",
                use_translate=False,
                use_split=False,
            )
        )

        all_rows.extend(
            run_case(
                case_name=f"semantic_vi_{idx}",
                query=item["vi"],
                language="vi",
                mode="semantic",
                scenes="all",
                use_translate=True,
                use_split=False,
            )
        )

    print("\n========== TEMPORAL ==========")

    for idx, item in enumerate(TEMPORAL_QUERIES, start=1):
        scenes = item["scenes"]

        all_rows.extend(
            run_case(
                case_name=f"temporal_en_{idx}",
                query=item["en"],
                language="en",
                mode="temporal",
                scenes=scenes,
                use_translate=False,
                use_split=False,
            )
        )

        all_rows.extend(
            run_case(
                case_name=f"temporal_vi_{idx}",
                query=item["vi"],
                language="vi",
                mode="temporal",
                scenes=scenes,
                use_translate=True,
                use_split=False,
            )
        )

    summarize(all_rows)

    print("\n========== SPLIT DETAILS ==========")

    for row in all_rows[::REPEAT]:
        print()
        print(
            f"{row['mode'].upper()} | {row['language'].upper()} | "
            f"scenes={row['scenes']} | split={row['use_split']}"
        )
        print(f"Query         : {row['query']}")
        print(f"Events        : {row['events']}")
        print(f"Event queries : {row['event_queries']}")
        print(f"Sub queries   : {row['sub_queries']}")


if __name__ == "__main__":
    main()