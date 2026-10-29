# Event Retrieval System

Ứng dụng tìm kiếm sự kiện trong video gồm API Python/FastAPI và giao diện React/Vite. Cấu hình hiện tại có hai dataset V3C và LHE, mỗi dataset dùng một index Qwen3-VL-Embedding-2B qua vLLM. Dữ liệu và index cần được chuẩn bị riêng.

## Bản đồ repo

| Vị trí | Vai trò |
| --- | --- |
| [backend/](backend/README.md) | API, truy vấn, trích xuất và tạo index. Python dependencies ở `backend/requirements.txt`. |
| [frontend/](frontend/README.md) | Giao diện và API client. JavaScript dependencies ở `frontend/package.json` và `frontend/package-lock.json`. |
| [backend/docs/PIPELINE_GUIDE.md](backend/docs/PIPELINE_GUIDE.md) | Chuẩn bị dữ liệu và index. |
| [backend/notebooks/README.md](backend/notebooks/README.md) | Notebook tạo keyframe, media, ASR/caption embeddings và host backend trên Kaggle/Drive. |

## Cấu trúc dữ liệu V3C và LHE

```text
data/
├── keyframes/{V3C,LHE}/<video_id>/<frame>.jpg
├── videos/{V3C,LHE}/<video_id>.*
├── map_keyframes/{V3C,LHE}/<video_id>.csv
├── embeddings/embeddings_qwen3_vl_embedding_2b_vllm/{V3C,LHE}/<video_id>/<frame>.npy
└── database/
    ├── faiss_qwen3_vl_embedding_2b_vllm/       # V3C
    └── faiss_qwen3_vl_embedding_2b_lhe_vllm/   # LHE
```

V3C dùng `video_id` năm chữ số, ví dụ `00001`; LHE dùng `LHE01`, `LHE02`, v.v. Notebook [host_server.ipynb](backend/notebooks/host_server.ipynb) cho chọn V3C hoặc LHE để tải embedding/map và build index trước khi host.

## Chạy UI và Codex bằng mock server

Chế độ mock dùng `backend/mock_server.py` ở cổng **8001**, độc lập với backend
retrieval thật ở cổng **8000**. Cài Codex CLI và đăng nhập trên máy chạy backend,
cài `backend/requirements-mock.txt`, rồi chạy frontend bằng `npm ci` và
`npm run dev:mock`. Không cần GPU/model/index cho chế độ này.

Codex chat và xem ảnh thật; search vẫn lấy mẫu ngẫu nhiên từ ảnh dataset. Agent
hiện dùng prompt + tool Python, chưa có skill nghiệp vụ hay adapter retrieval thật.
Xem [hướng dẫn mock/Codex](backend/docs/MOCK_SERVER.md) để dựng môi trường trên máy
mới, chuẩn bị ảnh, kiểm tra kết nối và hiểu các giới hạn.

## Chạy tại máy phát triển

1. Chuẩn bị dữ liệu/index theo `backend/docs/PIPELINE_GUIDE.md`, chỉnh `backend/configs/app.yaml` để khớp model và đường dẫn thật.
2. Trong `backend/`, cài môi trường Python từ `requirements.txt`, rồi chạy `python -m uvicorn src.api.main:app --reload --port 8000`. Danh sách này là snapshot lớn của môi trường backend, có gói GPU; xem README backend trước khi cài.
3. Trong `frontend/`, sao chép `.env.example` thành `.env`, đặt `VITE_API_BASE_URL`, chạy `npm ci` rồi `npm run dev`.

Kiểm tra `http://127.0.0.1:8000/api/health`; OpenAPI ở `/docs`. Khi chưa có model/index của một dataset, endpoint tìm kiếm dataset đó có thể trả 503.

## Luồng hệ thống

```text
video -> keyframe -> embedding / OCR / ASR -> FAISS / Elasticsearch / color index
query từ React -> frontend/src/api -> backend/src/api -> backend/src/retrieval
               -> kết quả + URL media -> giao diện
```

Runtime đăng ký V3C và LHE theo `backend/configs/app.yaml`, mặc định V3C. Frontend gọi API; URL ảnh, video, map có thể trỏ đến server media riêng qua các biến `VITE_*` trong `frontend/.env.example`.

## Đọc tiếp cho agent

- [README backend](backend/README.md): entrypoint, ranh giới module, API và cấu hình runtime.
- [README frontend](frontend/README.md): UI, API client, biến môi trường và lệnh kiểm tra.
- [Kiến trúc backend](backend/docs/ARCHITECTURE.md), [pipeline dữ liệu](backend/docs/PIPELINE_GUIDE.md), [PE-Core](backend/docs/PE_CORE.md), [God Mode relay](backend/docs/GODMODE_RELAY.md): tài liệu chuyên đề; đối chiếu code/config hiện tại khi sửa.
