# src/api — layout

App FastAPI được build 1 lần trong `main.py` (lifespan) — xem docstring đầu
file đó cho chi tiết đầy đủ. Router chia làm 2 nhánh CÙNG tồn tại song song:

## Nhánh search MỚI (không đổi so với backend.zip gốc)
- `routers/search.py` + `schemas/search.py`
- Path: `/api/search/semantic`, `/temporal`, `/ocr`, `/asr`, `/auto`,
  `/advanced`, `/models`.
- Response strict-typed (`SearchResultItem`), mỗi mode 1 endpoint riêng.

## Nhánh API GỐC (`main.py` file lẻ cũ) — phục hồi lại, tách theo domain
- `legacy/` — helpers thuần (không phải router):
  - `paths.py` — load config YAML + resolve path (`LegacyPaths`).
  - `serializers.py` — `dict_to_result_FAST` + các hàm serialize kết quả
    (contract mà UI cũ đang render trực tiếp: `image_url`, `video_url`,
    `map_url`, `matched_sequence`, `temporal`, ...).
  - `dres_client.py` — gọi ra DRES evaluation server.
  - `translate.py` — dịch query vi->en trước khi search.
  - `deps.py` — FastAPI dependencies đọc lại `app.state` (retrieval system,
    config, paths) đã build 1 lần lúc lifespan.
- `schemas/legacy.py` — `SearchRequest`, `SimilaritySearchRequest`,
  `DresLoginRequest`, `DresSubmitRequest`.
- `routers/legacy_search.py` — `POST /api/search` (1 endpoint hợp nhất,
  chọn mode qua `search_mode`), `GET /api/frame-info`,
  `GET /api/surrounding-frames`, `POST /api/similarity-search`.
- `routers/health.py` — `GET /api/health`, `GET /api/config`.
- `routers/dres.py` — `POST /api/dres/login`, `POST /api/dres/submit`.
- `routers/speech.py` — `POST /api/speech/transcribe` (Faster-Whisper).

Contract (path + request/response shape) của nhánh này giữ NGUYÊN so với
`main.py` cũ. Khác biệt bắt buộc duy nhất: hệ thống retrieval (FAISS/OCR/ASR)
giờ được build đúng 1 lần trong `api/main.py` lifespan (quy tắc chung của
toàn backend, xem `src/retrieval/system.py`) — router đọc lại qua
`request.app.state` thay vì tự `build_system()` ở top-level module như bản
gốc. Field `model`/`faiss` trong `/api/health`, `/api/config` được suy ra từ
model MẶC ĐỊNH đang chạy (`orchestrator.index`) vì `configs/app.yaml` giờ hỗ
trợ nhiều model (không còn khoá `CFG["model"]`/`CFG["faiss"]` phẳng như bản
cũ) — tên field response không đổi, chỉ khác nguồn dữ liệu.
