# Chuẩn bị dữ liệu và index cho V3C/LHE

Hai dataset vật lý của API là `v3c` và `lhe`. CAM là tên nhánh cũ, không dùng trong cấu hình runtime. Mỗi dataset chỉ bật một model semantic: `Qwen/Qwen3-VL-Embedding-2B`, engine `vllm`. PE-Core trong notebook VLM dùng để **chọn keyframe**, không phải embedding cho FAISS retrieval.

## Cây dữ liệu

```text
data/
├── videos/{V3C,LHE}/<video_id>.*
├── keyframes/{V3C,LHE}/<video_id>/<frame>.jpg
├── map_keyframes/{V3C,LHE}/<video_id>.csv
├── embeddings/embeddings_qwen3_vl_embedding_2b_vllm/{V3C,LHE}/<video_id>/<frame>.npy
└── database/
    ├── faiss_qwen3_vl_embedding_2b_vllm/       # V3C
    └── faiss_qwen3_vl_embedding_2b_lhe_vllm/   # LHE
```

`vlm_keyframes.ipynb` tạo keyframe, map và scene của LHE. PKL PE-Core mà notebook tạo ra không thay cho file Qwen `.npy`. Dùng `extract_embedding.ipynb` để tạo vector Qwen vLLM cho từng ảnh keyframe; đặt output vào đúng thư mục dataset ở trên. Không trộn vector của V3C và LHE.

## Build FAISS và vector cache

Chạy từ thư mục `backend/` sau khi có embedding `.npy` và map CSV:

```bash
python -m scripts.retrieval.run --task build-index --dataset v3c
python -m scripts.retrieval.run --task build-vector-cache --config configs/app.yaml --dataset v3c --dtype float32
python -m scripts.retrieval.run --task build-index --dataset lhe
python -m scripts.retrieval.run --task build-vector-cache --config configs/app.yaml --dataset lhe --dtype float32
```

`configs/indexing.yaml` giới hạn input V3C ở source group `V3C`, LHE ở `LHE`. Nếu dataset chưa có vector, lệnh build báo lỗi thay vì tạo index trộn hoặc rỗng. `configs/app.yaml` khai báo `index_dir` chưa có hậu tố engine; API thêm `_vllm` và nạp đúng index riêng cho từng dataset. V3C vẫn là dataset mặc định.

## Chạy dịch vụ

Backend dùng vLLM server riêng cho Qwen; xem `requirements-vllm.txt`. `host_server.ipynb` có lựa chọn `DATASET=V3C` hoặc `DATASET=LHE`, tải ZIP embedding/map tương ứng rồi build index và cache. Khi chọn LHE cần điền Drive folder ID embedding/map LHE ở cell cấu hình. Nếu thiếu index LHE, API vẫn có thể phục vụ V3C; request tìm kiếm LHE trả 503 cho đến khi index và vLLM sẵn sàng.

Frontend đọc danh sách dataset từ `/api/config`, cho phép chọn V3C hoặc LHE. URL media có dạng `/keyframes/LHE/...`, `/videos/LHE/...`, `/mapkeyframes/LHE/...` hoặc URL từ host media riêng. OCR, ASR và color là các nguồn tùy chọn với index tách riêng theo dataset; cấu hình hiện tại chưa bật OCR/ASR.
