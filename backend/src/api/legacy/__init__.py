# Legacy API package.
#
# Toàn bộ code trong package này là API GỐC từ `main.py` (file lẻ) trước khi
# hệ thống được tách thành `src/api/routers/search.py` (nhánh search mới,
# xem README.md ở root). KHÔNG được đổi contract (path, request/response
# shape) của các module trong package này nếu không thật sự cần thiết —
# frontend cũ đang gọi thẳng vào các path này.
#
# Chỉ khác bản gốc ở 1 điểm bắt buộc: hệ thống retrieval (FAISS/OCR/ASR) giờ
# được build DUY NHẤT 1 lần trong `api/main.py` lifespan (theo quy tắc mới,
# xem `src/retrieval/system.py`), nên các hàm ở đây nhận `RetrievalSystem`/
# `Orchestrator` qua tham số thay vì tự gọi `build_system()` như bản main.py
# cũ (vốn build ngay ở top-level module).
