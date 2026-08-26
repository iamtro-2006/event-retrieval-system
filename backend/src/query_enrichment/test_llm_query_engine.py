"""Live smoke test for the optional query-enrichment API.

Run from the repository root. The test reads backend/.env without printing
secrets and exercises the three JSON contracts used by the orchestrator.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from src.query_enrichment.llm_query_engine import LLMQueryEngine


def _load_backend_env() -> None:
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _load_backend_env()
    api_key = os.getenv("QUERY_ENRICHMENT_API_KEY", "")
    if not api_key:
        print("SKIP: QUERY_ENRICHMENT_API_KEY is not set")
        return 2

    engine = LLMQueryEngine(
        api_key=api_key,
        base_url=os.getenv("QUERY_ENRICHMENT_BASE_URL", "https://api.xah.io/v1"),
        model=os.getenv("QUERY_ENRICHMENT_MODEL", "claude-haiku-4.5"),
        timeout=20,
        max_retries=1,
        max_workers=2,
        max_output_tokens=512,
    )

    query = "một người đàn ông mở cửa rồi một phụ nữ bước vào và ngồi xuống ghế"
    print(f"MODEL: {os.getenv('QUERY_ENRICHMENT_MODEL', 'claude-haiku-4.5')}")
    print(f"INPUT: {query}")
    events = engine.split_temporal_events(query)
    print(f"TEMPORAL_EVENTS: {events}")
    for index, event in enumerate(events or [query], start=1):
        paraphrases = engine.paraphrase_semantic_query(event, max_subqueries=4)
        print(f"EVENT_{index}: {event}")
        print(f"EVENT_{index}_PARAPHRASES: {paraphrases}")
    focused = engine.generate_focused_queries(query, ["semantic", "ocr", "asr"])
    print(f"FOCUSED_QUERIES: {focused}")
    return 0 if events and focused else 1


if __name__ == "__main__":
    raise SystemExit(main())
