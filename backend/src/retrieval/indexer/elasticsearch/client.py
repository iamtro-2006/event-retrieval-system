from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

BACKEND_DIR = Path(__file__).resolve().parents[4]
try:
    from dotenv import load_dotenv

    load_dotenv(BACKEND_DIR / ".env")
except ImportError:
    pass


def _env_override(name: str) -> str | None:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else None


class ElasticsearchService:
    """Thin wrapper around the Elasticsearch client, scoped to a single index.

    Dùng chung cho cả ASR và OCR (trước đây là 2 bản copy-paste gần như
    giống hệt nhau, chỉ khác field dùng để full-text search).

    Connection settings resolve in this order, so the same YAML works for
    every teammate without editing tracked config:

        1. env var (ES_HOST / ES_PORT / ES_SCHEME) — per-machine override
        2. value from configs/*_extraction.yaml — the checked-in default

    Local docker (`backend/docker-compose.yaml`) is the default target, so a
    fresh clone with `docker compose up -d` needs no config edit at all.
    """

    def __init__(
        self,
        host: str,
        port: int,
        index_name: str,
        scheme: str = "http",
        search_field: str = "text",
    ) -> None:
        self.index_name = index_name
        self.search_field = search_field

        resolved_host = _env_override("ES_HOST") or str(host)
        resolved_scheme = _env_override("ES_SCHEME") or str(scheme or "http")
        resolved_port = int(_env_override("ES_PORT") or port)

        # A reverse proxy / ngrok tunnel usually terminates TLS on 443 and the
        # port is implicit, so only append it when it carries information.
        default_port = 443 if resolved_scheme == "https" else 80
        url = (
            f"{resolved_scheme}://{resolved_host}"
            if resolved_port == default_port
            else f"{resolved_scheme}://{resolved_host}:{resolved_port}"
        )

        self.url = url
        self.client = Elasticsearch(
            url,
            # Harmless against a plain local node; required when the same
            # config points at an ngrok tunnel.
            headers={"ngrok-skip-browser-warning": "true"},
        )

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        return self.client.ping()

    def health(self) -> dict[str, Any]:
        """Return a JSON-safe reachability report for this index.

        `ping()` alone is not enough for the API health endpoint: a node can
        answer `/` while the specific index was never created, which is the
        normal state before the first seed run.
        """
        try:
            reachable = bool(self.client.ping())
        except Exception as exc:  # noqa: BLE001 - report, never raise from health
            return {
                "url": self.url,
                "index": self.index_name,
                "reachable": False,
                "index_exists": False,
                "documents": 0,
                "error": f"{type(exc).__name__}: {exc}",
            }

        report: dict[str, Any] = {
            "url": self.url,
            "index": self.index_name,
            "reachable": reachable,
            "index_exists": False,
            "documents": 0,
            "error": None,
        }
        if not reachable:
            report["error"] = "node did not answer ping"
            return report

        try:
            report["index_exists"] = bool(self.index_exists())
            if report["index_exists"]:
                report["documents"] = int(self.count())
        except Exception as exc:  # noqa: BLE001
            report["error"] = f"{type(exc).__name__}: {exc}"
        return report

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------

    def index_exists(self) -> bool:
        return self.client.indices.exists(index=self.index_name)

    def delete_index(self) -> None:
        if self.index_exists():
            self.client.indices.delete(index=self.index_name)

    def create_index(self, mapping: dict[str, Any]) -> None:
        if self.index_exists():
            return

        self.client.indices.create(
            index=self.index_name,
            mappings=mapping,
            # A single-node local node (docker-compose.yaml) cannot allocate
            # replicas, which leaves every index yellow and makes bulk
            # indexing wait on unassigned shards. One replica only makes sense
            # on a real multi-node cluster.
            settings={"number_of_replicas": 0},
        )

    def refresh(self) -> None:
        """Make just-written documents visible to search/count.

        Elasticsearch only exposes new documents after the next refresh
        (1s by default). Seeding scripts and the health endpoint read counts
        immediately after writing, so without an explicit refresh they report
        zero documents for data that is already stored.
        """
        if self.index_exists():
            self.client.indices.refresh(index=self.index_name)

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def insert(self, document: dict[str, Any]) -> None:
        self.client.index(
            index=self.index_name,
            document=document,
        )

    def bulk_insert(self, documents: list[dict[str, Any]]) -> None:
        actions = [
            {"_index": self.index_name, "_source": doc}
            for doc in documents
        ]
        bulk(self.client, actions)
        # Refresh once per batch (not per document) so callers that count or
        # search right after seeding see the documents they just wrote.
        self.refresh()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, size: int = 10) -> list[dict]:
        from src.retrieval.retriever.common.video_filter import video_ids_context
        video_ids = video_ids_context.get()
        response = self.client.search(
            index=self.index_name,
            query={
                "bool": {
                    "filter": [{"terms": {"video_id": sorted(video_ids)}}] if video_ids else [],
                    # =========================================================
                    # TẦNG 1 — RECALL FILTER (bắt buộc)
                    # Chỉ cần khớp 50% số token là lọt vào tập kết quả.
                    # KHÔNG dùng operator="and" ở đây vì nó sẽ vô hiệu hoá
                    # minimum_should_match hoàn toàn (ES rule: minimum_should_match
                    # chỉ có tác dụng khi operator mặc định là "or").
                    # =========================================================
                    "must": {
                        "match": {
                            self.search_field: {
                                "query": query,
                                "operator": "or",
                                "minimum_should_match": "50%",
                            }
                        }
                    },

                    # =========================================================
                    # TẦNG 2 — RERANK theo nhiều "trường hợp", mỗi case là 1
                    # should clause độc lập, không loại kết quả, chỉ cộng điểm.
                    # Xếp theo độ ưu tiên giảm dần qua boost.
                    # =========================================================
                    "should": [

                        # --- Case A: khớp CỤM chính xác, ĐÚNG THỨ TỰ tuyệt đối ---
                        # Ưu tiên cao nhất — user gõ đúng y hệt cụm trong document.
                        {
                            "match_phrase": {
                                self.search_field: {
                                    "query": query,
                                    "slop": 0,
                                    "boost": 10.0,
                                }
                            }
                        },

                        # --- Case B: khớp CỤM gần đúng thứ tự (cho phép lệch nhẹ) ---
                        # Bắt các biến thể: chèn thêm từ ở giữa, đảo nhẹ vị trí.
                        # slop=3 nghĩa là tối đa 3 bước dịch chuyển để khớp lại thứ tự.
                        {
                            "match_phrase": {
                                self.search_field: {
                                    "query": query,
                                    "slop": 3,
                                    "boost": 6.0,
                                }
                            }
                        },

                        # --- Case C: khớp ĐỦ 100% token, KHÔNG cần đúng thứ tự ---
                        # Document chứa hết các từ khóa nhưng nằm rải rác.
                        {
                            "match": {
                                self.search_field: {
                                    "query": query,
                                    "operator": "and",
                                    "boost": 4.0,
                                }
                            }
                        },

                        # --- Case D: khớp chính xác >=85% token (siết hơn tầng recall) ---
                        # Cầu nối giữa "khớp hết" (case C) và "khớp 50%" (must).
                        {
                            "match": {
                                self.search_field: {
                                    "query": query,
                                    "operator": "or",
                                    "minimum_should_match": "85%",
                                    "boost": 2.5,
                                }
                            }
                        },

                        # --- Case E: FUZZY — bắt lỗi chính tả / dấu / OCR nhiễu ---
                        # Đặt boost thấp vì đây là fallback, không phải tín hiệu mạnh.
                        # prefix_length=2 để tránh fuzzy phá vỡ ký tự đầu của từ
                        # (giảm false-positive, vd. "chó" fuzzy không nên khớp "cho").
                        {
                            "match": {
                                self.search_field: {
                                    "query": query,
                                    "operator": "or",
                                    "fuzziness": "AUTO",
                                    "prefix_length": 2,
                                    "max_expansions": 20,
                                    "boost": 1.5,
                                }
                            }
                        },

                        # --- Case F: PROXIMITY — các từ khóa gần nhau trong văn bản ---
                        # Không yêu cầu đúng cụm, chỉ cần các token nằm gần nhau
                        # (khoảng cách <= slop). Hữu ích khi câu bị OCR/ASR cắt rời.
                        {
                            "match_phrase": {
                                self.search_field: {
                                    "query": query,
                                    "slop": 10,
                                    "boost": 1.0,
                                }
                            }
                        },
                    ],

                    # Không bắt buộc should nào phải match — chúng chỉ cộng điểm.
                    "minimum_should_match": 0,
                }
            },
            size=size,
        )
        return response["hits"]["hits"]

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------

    def count(self) -> int:
        response = self.client.count(index=self.index_name)
        return response["count"]
