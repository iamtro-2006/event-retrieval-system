from __future__ import annotations

from typing import Any

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk


class ElasticsearchService:
    """Thin wrapper around the Elasticsearch client, scoped to a single index.

    Dùng chung cho cả ASR và OCR (trước đây là 2 bản copy-paste gần như
    giống hệt nhau, chỉ khác field dùng để full-text search).
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
        self.client = Elasticsearch(
            f"{scheme}://{host}:{port}",
            headers={"ngrok-skip-browser-warning": "true"},
        ) # offline cân port
        # self.client = Elasticsearch(f"{scheme}://{host}") # online kh cần port vd elastic-search.tku.life

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        return self.client.ping()

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
        )

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

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, size: int = 10) -> list[dict]:
        response = self.client.search(
            index=self.index_name,
            query={
                "bool": {
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
