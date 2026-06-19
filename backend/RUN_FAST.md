# Backend HNSW tối ưu

## Cấu hình đã áp dụng

- Giữ nguyên schema và endpoint API hiện có.
- Đọc index `data/database/faiss_hnsw_clip_vitl16_siglip_256/keyframes.faiss`.
- HNSW runtime: `efSearch=64`; index cũ của bạn đã build với `efConstruction=200` nên không cần build lại.
- `M` không thể suy ra từ `efConstruction`; backend đọc trực tiếp index đã lưu nên vẫn dùng đúng graph/M đã build.
- Một batch encode text cho toàn bộ query/sub-query.
- Một batch HNSW search cho mỗi nhóm query; không encode lại từng event.
- Temporal search reconstruct vector một lần từ FAISS vào RAM rồi dùng matrix multiplication; không đọc `.npy` trong request.
- Metadata được lập lookup O(1) cho frame và lookup theo video.
- Whisper lazy-load, không làm chậm startup retrieval nếu chưa dùng speech.

## Cài đặt khuyến nghị

Python 3.11 được khuyến nghị. Tạo môi trường mới:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
```

Cài PyTorch trước bằng lệnh được sinh tại trang **PyTorch – Get Started**, chọn đúng GPU và CUDA wheel. Sau đó:

```bash
pip install -r requirements-fast.txt
```

Kiểm tra CUDA:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Kiểm tra FAISS/HNSW:

```bash
python -c "import faiss; i=faiss.read_index('data/database/faiss_hnsw_clip_vitl16_siglip_256/keyframes.faiss'); print(type(i), i.ntotal, i.d, getattr(i, 'hnsw', None).efSearch)"
```

## Chạy server

Chạy từ thư mục backend, không dùng `--reload` khi benchmark hoặc production vì reloader tạo process phụ và load model/index hai lần:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Chỉ dùng **1 worker** vì mỗi worker sẽ giữ một bản ViT-L và vector cache. Tăng throughput bằng batch search bên trong request và FAISS threads, không nhân worker cho đến khi đã đo đủ RAM/VRAM.

## Tuning tốc độ/độ chính xác

Trong `configs/app.yaml`:

- `faiss.ef_search: 64`: điểm cân bằng tốt. Thử 32 để nhanh hơn; 96/128 để tăng recall.
- `faiss.threads: 8`: thường tốt trên desktop. Đo 4/8/12; quá nhiều thread có thể chậm do contention.
- `faiss.cache_index_vectors: true`: bắt buộc để temporal nhanh. RAM xấp xỉ `N × dimension × 4 bytes`.
- `model.precision: amp`: dùng FP16 autocast trên CUDA; CPU tự chuyển FP32.
- `model.compile: false`: bật `true` chỉ khi PyTorch/GPU ổn định; lần đầu warm-up lâu hơn và không phải GPU nào cũng lợi.
- `candidate_multiplier`: giảm từ 3 xuống 2 nếu cần latency thấp; tăng lên 4–5 nếu ưu tiên recall temporal.

## Warm-up sau startup

Gửi một semantic query và một temporal query mẫu trước benchmark để warm CUDA kernels, tokenizer, filesystem và HNSW cache. Không benchmark request đầu tiên.

## Build lại HNSW (khi cần)

`configs/indexing.yaml` hiện dùng:

```yaml
index:
  type: hnsw
  metric: cosine
  hnsw_m: 32
  ef_construction: 200
  ef_search: 64
```

Build bằng CLI/pipeline cũ của dự án. `efConstruction` chỉ ảnh hưởng lúc build; `efSearch` được backend gán lại lúc load và có thể chỉnh mà không build lại.
