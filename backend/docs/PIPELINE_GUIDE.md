# Hướng dẫn chuẩn bị dữ liệu và chạy retrieval

Tài liệu này tóm tắt luồng chính cho hai dataset AIC và CAM. Chạy các lệnh Python bên dưới từ thư mục `backend/`. Thay `aic` bằng `cam` khi cần. Đường dẫn trong config được tính tương đối từ `backend/`.

## 1. Thư mục dữ liệu tham khảo

```text
data/
├── AIC/                              # CAM có cấu trúc tương tự
│   ├── videos/<split>/<video_id>.*   # video gốc, ví dụ L21/L21_V001.mp4
│   ├── keyframes/<split>/<video_id>/<frame>.jpg
│   ├── map_keyframes/<split>/<video_id>.csv
│   ├── embeddings/<model>/<split>/<video_id>/<frame>.npy
│   ├── ocr/<split>/<video_id>.json
│   ├── asr/<split>/<video_id>.json
│   └── database/<index_name>/        # FAISS index + metadata (+ vector cache)
└── CAM/                              # thư mục độc lập, không dùng chung index AIC
```

`<split>` là tên phần dữ liệu, ví dụ `L21` hoặc `L22`. Giữ nguyên `video_id` và tên frame giữa keyframes, embeddings, map, OCR/ASR để metadata trỏ đúng ảnh/video.

## 2. Video → keyframes và map

Chạy bộ trích keyframe cho từng video. `--output-dir` phải là gốc dataset để sinh `keyframes/` và `map_keyframes/` bên dưới:

```powershell
python scripts/keyframe_extraction/run.py --method distribution --video ..\data\AIC\videos\L21\L21_V001.mp4 --output-dir ..\data\AIC
```

Kết quả chính:

```text
data/AIC/keyframes/L21/L21_V001/*.jpg
data/AIC/map_keyframes/L21/L21_V001.csv
```

`--method` nhận `distribution` (mặc định) hoặc `kfeavi`. Thêm `--overwrite` để tạo lại kết quả đã có.

## 3. Keyframes → embeddings

Script đọc các thư mục keyframe và ghi vector `.npy`. Cấu hình mẫu là `configs/embeddings.yaml`; trước khi chạy, chỉnh `input_keyframes_root`, `output_embeddings_root` và bật đúng model trong YAML cho dataset/model cần tạo. PE-Core yêu cầu package/weights riêng; SigLIP2 dùng preset khai báo trong config.

```powershell
python scripts/embedding_extraction/run.py --config configs/embeddings.yaml --model siglip2_so400m_384
```

Mỗi model ghi ra một thư mục riêng dưới `output_embeddings_root`, tên dạng `embeddings_<model-key>`. Ví dụ output mong muốn cho CAM:

```text
data/CAM/embeddings/embeddings_siglip2_so400m_384/<split>/<video_id>/<frame>.npy
```

Muốn chạy AIC, dùng config trỏ tới `data/AIC/keyframes` và output nằm dưới `data/AIC/embeddings`. Không dùng chung vector giữa các dataset hoặc giữa các model khác nhau.

## 4. Embeddings → FAISS semantic index

`configs/indexing.yaml` chọn dataset bằng `--dataset`; kiểm tra ba đường dẫn `keyframes_root`, `embeddings_root`, `map_keyframes_root` cùng `index.output_dir` trong dataset đó khớp với đầu ra các bước trước. `index.output_dir` cần khớp `semantic_models.<model_key>.index_dir` trong `configs/app.yaml`.

```powershell
python scripts/retrieval/run.py --task build-index --config configs/indexing.yaml --dataset aic
python scripts/retrieval/run.py --task build-index --config configs/indexing.yaml --dataset cam
```

Index gồm `keyframes.faiss` và `metadata.csv`. Vector cache `vectors_fp32.npy` là tùy chọn theo runtime config; build bằng `--task build-vector-cache` khi cần và trỏ config tới đúng FAISS index.

## 5. Các modality khác

### OCR

OCR đọc keyframes, xuất một JSON cho mỗi video. Trước khi chạy, đặt `extraction.input_keyframes_root` và `extraction.output_root` trong `configs/ocr_extraction.yaml` cho dataset đang xử lý (ví dụ `../data/AIC/keyframes` và `../data/AIC/ocr`). Sau đó:

```powershell
python scripts/ocr/extract/run.py --config configs/ocr_extraction.yaml
python scripts/ocr/index/run.py --dataset aic
```

Indexing script chọn index theo dataset: AIC `ocr_aic`, CAM `ocr_cam`. OCR JSON nằm dưới `<dataset-root>/ocr/<split>/<video_id>.json`.

### ASR

Đặt transcript theo từng video dưới `<dataset-root>/asr/<split>/<video_id>.json`. Mỗi file là danh sách segment có `start`, `end`, `transcript` (có thể kèm `video_id`), ví dụ:

```json
[{"start": 1.2, "end": 4.8, "transcript": "Lời thoại trong đoạn này."}]
```

Sau khi transcript đã sẵn sàng, tạo index tương ứng:

```powershell
python scripts/asr/index/run.py --dataset aic
```

CAM dùng `--dataset cam`; các index là `asr_aic` và `asr_cam`. Repo có endpoint speech transcription và script mock, nhưng không có một CLI ASR batch extraction tổng quát trong luồng này.

### Color

Sau khi FAISS index có `metadata.csv`, tạo color index độc lập:

```powershell
python scripts/color/extract.py --metadata ..\data\AIC\database\faiss_hnsw_pe_core_l14_336\metadata.csv --output ..\data\AIC\database\color_5x5.npz --keyframes-root ..\data\AIC\keyframes
```

## 6. Khởi động API và frontend

API nạp model/index theo `configs/app.yaml`; các model runtime đang bật phải có index đúng đường dẫn và được tạo từ cùng model/preprocessing. Elasticsearch cần truy cập được nếu dùng OCR/ASR.

Terminal 1, tại `backend/`:

```powershell
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Terminal 2, tại `frontend/`:

```powershell
npm run dev
```

Mở URL Vite hiển thị trên terminal. Kiểm tra API tại `http://127.0.0.1:8000/api/health`; tài liệu endpoint ở `/docs`.

## Lưu ý về đường dẫn hiện tại

- `configs/embeddings.yaml` là config mẫu trỏ vào CAM; khi xử lý dataset khác phải đổi roots trước khi chạy.
- Đối chiếu `configs/indexing.yaml` với `configs/app.yaml` trước khi build: index output phải đúng `index_dir` mà API dùng. Hai file hiện có một số đường dẫn không đồng nhất giữa AIC/CAM; đừng khởi động API với index vừa build nếu chúng khác nhau.
- Script keyframe sinh tên thư mục `map_keyframes`; kiểm tra `map_keyframes_root` của CAM trong `configs/app.yaml` khớp thư mục thực tế (hiện khai báo `map-keyframes`).
- Lệnh indexing Elasticsearch tạo index và ingest các JSON đang có. Chạy lại có thể đưa dữ liệu trùng vào index nếu không xóa/reset index trước.
