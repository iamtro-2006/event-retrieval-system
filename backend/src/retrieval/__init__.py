# Cố ý KHÔNG re-export gì ở đây (khác trước: từng có `from .indexer import *`
# + `from .retriever import *`). Lý do bỏ:
#   - Không có nơi nào trong repo dùng top-level re-export này (grep xác
#     nhận mọi caller import thẳng từ submodule, giống convention đã ghi ở
#     `retriever/__init__.py`: "Package này CHƯA export gì ở đây; import
#     thẳng từ submodule cần dùng").
#   - `from .indexer import *` kéo theo `import faiss` (và transitively
#     elasticsearch/milvus client modules) ngay khi BẤT KỲ submodule nào
#     dưới `src.retrieval.*` được import — kể cả từ lớp API
#     (`src/api/main.py` chỉ cần `src.retrieval.system`), vì Python luôn
#     chạy `__init__.py` của package cha trước. Điều này chặn việc chạy
#     integration test HTTP layer với mock `RetrievalSystem` (không cần
#     faiss/model thật) — xem `tests_manual/test_api_integration.py`.
__all__ = []
