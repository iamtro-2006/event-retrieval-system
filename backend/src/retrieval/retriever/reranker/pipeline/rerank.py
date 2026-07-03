"""
pipeline/rerank.py — PLACEHOLDER.

Reranker hiện KHÔNG được dùng trong hệ thống (theo quyết định mới nhất).
File `models/internvl_reranker.py` chỉ được move nguyên khối sang đúng vị trí
theo REFACTOR_PLAN.md, KHÔNG tách load-model / batch-call như dự tính ban đầu
ở Prompt 5 / mục "Rủi ro cần lưu ý" — việc tách đó bị huỷ vì không còn nhu cầu.

`Orchestrator` (common/orchestrator.py) và `retrieval/system.py` KHÔNG import
gì từ package này. Nếu sau này cần bật lại reranker, implement `BaseRetriever`
(hoặc 1 interface rerank riêng) ở đây và wire vào factory.py + orchestrator.
"""

from __future__ import annotations


def build_reranker_pipeline(*_args, **_kwargs):
    """Placeholder — reranker chưa được kích hoạt lại trong hệ thống hiện tại."""
    raise NotImplementedError(
        "Reranker pipeline is a placeholder and is not wired into the system. "
        "See models/internvl_reranker.py for the (unused) original implementation."
    )
