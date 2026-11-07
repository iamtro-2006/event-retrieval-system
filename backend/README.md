# Backend

Để test UI/chat/tool mà không tải model, dùng [mock server + Codex](docs/MOCK_SERVER.md) trên cổng 8001. Điểm vào độc lập: `mock_server:app`. Dùng `requirements-mock.txt` và môi trường `.venv-mock`; không cần cài dependencies GPU. Đây chỉ là chế độ mock. Backend thật có agent gọi `RetrievalSystem.search_plan` và dùng skill `hierarchical-video-search`; xem [HIERARCHICAL_SEARCH.md](docs/HIERARCHICAL_SEARCH.md). Hướng dẫn clone/chạy trên máy khác, model/effort và kiểm thử nằm trong tài liệu trên.

FastAPI phục vụ tìm kiếm video theo nội dung hình ảnh (semantic), chuỗi sự kiện (temporal), chữ trong khung hình (OCR), lời thoại (ASR), cùng fusion, color, multimodal và DRES. Ví dụ lệnh dưới đây bắt đầu ở gốc repo và chuyển vào `backend/`.

## Điểm vào và cấu hình

- `src/api/main.py` tạo ASGI `app`, nạp `CONFIG_PATH` (mặc định `configs/app.yaml`) trong lifespan, tạo `RetrievalSystem` riêng cho V3C và LHE rồi lưu vào `app.state`. `main.py` ở gốc backend re-export cùng app.
- `configs/app.yaml` định nghĩa dataset, model semantic đang bật, index, media roots và tính năng runtime. `configs/ocr_extraction.yaml` và `configs/asr_extraction.yaml` được nạp cho OCR/ASR lúc chạy; `embeddings.yaml`, `indexing.yaml`, `kf_extraction.yaml` phục vụ pipeline offline.
- Media local được xếp theo loại và dataset: `../data/keyframes/{V3C,LHE}/`, `../data/videos/{V3C,LHE}/`, `../data/map_keyframes/{V3C,LHE}/`. API phục vụ các root này qua `/keyframes`, `/videos`, `/mapkeyframes` khi thư mục tồn tại lúc khởi động. Frontend có thể trỏ mỗi loại media sang host riêng.
- Một model chỉ dùng được khi `enabled`, checkpoint và index tương ứng đều sẵn sàng. API có thể khởi động khi thiếu model/index, nhưng endpoint tìm kiếm của dataset không build được sẽ trả 503. OCR/ASR phụ thuộc Elasticsearch và có thể vắng mặt.
- `Qwen/Qwen3-VL-Embedding-2B` là model semantic ảnh/text duy nhất của cả V3C và LHE, dùng vLLM qua server riêng. Mỗi dataset có FAISS index 2048 chiều riêng; xem [README gốc](../README.md#cấu-trúc-dữ-liệu-v3c-và-lhe). OCR dùng pipeline khác.

## Cài và chạy

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn src.api.main:app --reload --port 8000
```

`requirements.txt` là danh sách Python chung của backend; có gói PyTorch/CUDA, OCR và model nặng nên cần môi trường tương thích. PE-Core đã được pin trong cùng file; xem [docs/PE_CORE.md](docs/PE_CORE.md) về model và license. Relay God Mode có lệnh cài tối giản riêng trong [docs/GODMODE_RELAY.md](docs/GODMODE_RELAY.md). Frontend dùng npm.

vLLM không nằm trong `requirements.txt`: chạy server bằng môi trường Linux/Python 3.12 riêng với `requirements-vllm.txt`. Backend chỉ cần `httpx` đã có trong requirements để gọi server. Nhờ đó vLLM tự chọn bộ PyTorch/CUDA phù hợp mà không ép thay các pin của backend. Cách chạy và chọn engine ở [PIPELINE_GUIDE.md](docs/PIPELINE_GUIDE.md).

`GET /api/health` kiểm tra trạng thái; `/docs` hiển thị OpenAPI. `docker-compose.yaml` khai báo Elasticsearch và Kibana cho chức năng cần Elasticsearch; không khởi chạy API.

## Search scope và agent thật

Đọc [AGENTS.md](../AGENTS.md) và [hợp đồng search/scope](docs/HIERARCHICAL_SEARCH.md) trước khi sửa. Agent thật dùng `/api/agent/chat` và ba tool `list_capabilities`, `register_plans`, `run_next_plan`. `scope` độc lập với `mode`: dataset, videos, pool_frames, pool_videos. Giới hạn video của session là giới hạn ngoài cùng; parent pool chỉ ghi nhận nhánh. Không search global top-k rồi mới lọc.

`pool_frames` hiện chỉ hỗ trợ semantic/hierarchical. Mapping gần biên khi thiếu keyframe đại diện là chủ đích, không phải lỗi cần loại bỏ.

## Bản đồ mã nguồn

| Vị trí | Chức năng |
| --- | --- |
| `src/api/main.py`, `src/api/routers/`, `src/api/schemas/`, `src/api/legacy/` | Lifespan, endpoint, schema và adapter contract giao diện. |
| `src/retrieval/search_scope.py`, `src/retrieval/plan_search.py` | Giải scope và dispatch pipeline plan dùng chung, không chứa HTTP serialization. |
| `src/retrieval/retriever/common/candidate_filter.py` | Giới hạn video/frame theo request; không xử lý pool ID. |
| `src/retrieval/system.py` | Facade `RetrievalSystem`, `build_system()` và điều phối lúc khởi động. |
| `src/retrieval/retriever/common/` | Parse query, chọn mode, phối hợp nguồn và scoring. |
| `src/retrieval/retriever/{semantic_search,temporal_search,ocr_search,asr_search}/` | Pipeline đọc/search từng nguồn. |
| `src/retrieval/index/`, `src/retrieval/indexer/` | Truy cập index khi search và xây index offline; có FAISS, Elasticsearch, Milvus. |
| `src/keyframe_extraction/`, `src/embedding_extraction/`, `src/ocr_extraction/`, `src/asr_extraction/` | Pipeline tạo dữ liệu từ video/keyframe. |
| `src/color_retrieval/`, `src/query_enrichment/`, `src/translation/` | Tìm màu, enrichment tùy chọn, dịch query. |
| `src/utils/` | Config, thiết bị, log, seed, video I/O. |
| `scripts/` | CLI cho keyframe, embedding, OCR/ASR, index, truy vấn và chẩn đoán. |

Luồng offline chính: video → keyframe → embedding → FAISS; OCR, ASR và color có index riêng. Lệnh và cấu trúc dữ liệu ở [PIPELINE_GUIDE.md](docs/PIPELINE_GUIDE.md).

## API đang hoạt động

- UI hiện dùng `POST /api/search`, cùng `/api/search/fusion`, `/api/search/multimodal`, `/api/search/color`, `/api/similarity-search`, `/api/frame-info`, `/api/surrounding-frames`, `/api/video-preview`, `/api/video-keyframes` và `/api/video-ids` trong `routers/legacy_search.py`.
- `routers/search.py` cung cấp `/api/search/semantic`, `/temporal`, `/ocr`, `/asr`, `/auto`, `/advanced` và `GET /api/search/models` với schema riêng.
- `routers/plans.py` cung cấp `/api/search/plan`, `/api/search/caption`, `/api/search/hierarchical`, và các endpoint agent/session; contract scope áp dụng cho plan pipeline.
- Router khác cung cấp `/api/health`, `/api/config`, `/api/dres/login`, `/api/dres/submit`, `/api/speech/transcribe`, `/api/translate`.

Router lấy hệ thống đã build từ `app.state`. Khi thay đổi response cho UI, đối chiếu `frontend/src/api/retrievalAPI.js` và `frontend/src/App.jsx`. Cấu hình hiện tại dùng `v3c` và `lhe`, mặc định `v3c`.

## Tài liệu chuyên đề

- [HIERARCHICAL_SEARCH.md](docs/HIERARCHICAL_SEARCH.md): contract scope/mode, pool, mapping, ownership và regression checks.
- [ARCHITECTURE.md](docs/ARCHITECTURE.md): giải thích sâu thiết kế/pipeline; đối chiếu code hiện tại trước khi sửa.
- [PIPELINE_GUIDE.md](docs/PIPELINE_GUIDE.md): chuẩn bị dữ liệu và build index.
- [PE-Core](docs/PE_CORE.md) và [God Mode relay](docs/GODMODE_RELAY.md): hướng dẫn phần mở rộng.

## VLM keyframes

Luồng VLM hiện hành là bản hợp nhất V3C, dùng duy nhất
[`notebooks/vlm_keyframes.ipynb`](notebooks/vlm_keyframes.ipynb):
OmniShotCut clean_shot → lấy mẫu 3 FPS → GPU quality + PE-Core → P3 → dedup
shot kề nhau → JPEG/map. Code thuật toán ở `src/keyframe_extraction/`;
không còn notebook V3C thử nghiệm hoặc selector VLM cũ.

Sau sửa source, chạy `python scripts/keyframe_extraction/build_vlm_keyframes.py`
từ backend để đồng bộ notebook. Kiểm tra:
`python -m unittest tests.test_vlm_keyframes tests.test_vlm_remote_jobs tests.test_video_io -q`.
Chi tiết Kaggle, input và upload ở [notebooks/README.md](notebooks/README.md).

## Metadata cho trình phát video

`GET /api/video-keyframes` lấy ảnh và thời gian từ metadata FAISS của đúng dataset,
trả thêm `image_url`, `image_rel_path`, `fps` và chỉ số keyframe để FE dựng dải ảnh.
`keyframe_id` là chỉ số ảnh sparse; `frame_idx` là frame thật trong video. Helper
contract ở `src/api/legacy/video_keyframes.py`; serializer media hiện có vẫn tạo
đường dẫn tương đối theo root dataset. Không đọc CSV ở FE và không sửa thuật toán
retrieval. Mock có cùng contract, dùng CSV dưới `MOCK_DATA_ROOT` khi có mapping.
