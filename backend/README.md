# Backend

FastAPI phục vụ tìm kiếm video theo nội dung hình ảnh (semantic), chuỗi sự kiện (temporal), chữ trong khung hình (OCR), lời thoại (ASR), cùng fusion, color, multimodal và DRES. Ví dụ lệnh dưới đây bắt đầu ở gốc repo và chuyển vào `backend/`.

## Điểm vào và cấu hình

- `src/api/main.py` tạo ASGI `app`, nạp `CONFIG_PATH` (mặc định `configs/app.yaml`) trong lifespan, tạo `RetrievalSystem` riêng cho AIC/CAM và lưu vào `app.state`. `main.py` ở gốc backend re-export cùng app.
- `configs/app.yaml` định nghĩa dataset, model semantic đang bật, index, media roots và tính năng runtime. `configs/ocr_extraction.yaml` và `configs/asr_extraction.yaml` được nạp cho OCR/ASR lúc chạy; `embeddings.yaml`, `indexing.yaml`, `kf_extraction.yaml` phục vụ pipeline offline.
- Một model chỉ dùng được khi `enabled`, checkpoint và index tương ứng đều sẵn sàng. API có thể khởi động khi thiếu model/index, nhưng endpoint tìm kiếm của dataset không build được sẽ trả 503. OCR/ASR phụ thuộc Elasticsearch và có thể vắng mặt.

## Cài và chạy

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn src.api.main:app --reload --port 8000
```

`requirements.txt` là danh sách Python chung của backend; có gói PyTorch/CUDA, OCR và model nặng nên cần môi trường tương thích. PE-Core đã được pin trong cùng file; xem [docs/PE_CORE.md](docs/PE_CORE.md) về model và license. Relay God Mode có lệnh cài tối giản riêng trong [docs/GODMODE_RELAY.md](docs/GODMODE_RELAY.md). Frontend dùng npm.

`GET /api/health` kiểm tra trạng thái; `/docs` hiển thị OpenAPI. `docker-compose.yaml` khai báo Elasticsearch và Kibana cho chức năng cần Elasticsearch; không khởi chạy API.

## Bản đồ mã nguồn

| Vị trí | Chức năng |
| --- | --- |
| `src/api/main.py`, `src/api/routers/`, `src/api/schemas/`, `src/api/legacy/` | Lifespan, endpoint, schema và adapter contract giao diện. |
| `src/retrieval/system.py` | Facade `RetrievalSystem`, `build_system()` và điều phối lúc khởi động. |
| `src/retrieval/retriever/common/` | Parse query, chọn mode, phối hợp nguồn và scoring. |
| `src/retrieval/retriever/{semantic_search,temporal_search,ocr_search,asr_search}/` | Pipeline đọc/search từng nguồn. |
| `src/retrieval/index/`, `src/retrieval/indexer/` | Truy cập index khi search và xây index offline; có FAISS, Elasticsearch, Milvus. |
| `src/keyframe_extraction/`, `src/embedding_extraction/`, `src/ocr_extraction/` | Pipeline tạo dữ liệu từ video/keyframe. |
| `src/color_retrieval/`, `src/query_enrichment/`, `src/translation/` | Tìm màu, enrichment tùy chọn, dịch query. |
| `src/utils/` | Config, thiết bị, log, seed, video I/O. |
| `scripts/` | CLI cho keyframe, embedding, OCR/ASR, index, truy vấn và chẩn đoán. |

Luồng offline chính: video → keyframe → embedding → FAISS; OCR, ASR và color có index riêng. Lệnh và cấu trúc dữ liệu ở [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md).

## API đang hoạt động

- UI hiện dùng `POST /api/search`, cùng `/api/search/fusion`, `/api/search/multimodal`, `/api/search/color`, `/api/similarity-search`, `/api/frame-info`, `/api/surrounding-frames`, `/api/video-preview`, `/api/video-keyframes` và `/api/video-ids` trong `routers/legacy_search.py`.
- `routers/search.py` cung cấp `/api/search/semantic`, `/temporal`, `/ocr`, `/asr`, `/auto`, `/advanced` và `GET /api/search/models` với schema riêng.
- Router khác cung cấp `/api/health`, `/api/config`, `/api/dres/login`, `/api/dres/submit`, `/api/speech/transcribe`, `/api/translate`.

Router lấy hệ thống đã build từ `app.state`. Khi thay đổi response cho UI, đối chiếu `frontend/src/api/retrievalAPI.js` và `frontend/src/App.jsx`. Mỗi request cần chọn dataset phù hợp; index AIC và CAM độc lập.

## Tài liệu chuyên đề

- [ARCHITECTURE.md](ARCHITECTURE.md): giải thích sâu thiết kế/pipeline; đối chiếu code hiện tại trước khi sửa.
- [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md): chuẩn bị dữ liệu AIC/CAM và build index.
- [PE-Core](docs/PE_CORE.md) và [God Mode relay](docs/GODMODE_RELAY.md): hướng dẫn phần mở rộng.
