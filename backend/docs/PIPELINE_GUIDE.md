# Hướng dẫn chuẩn bị dữ liệu và chạy retrieval

Tài liệu này tóm tắt luồng chính cho hai dataset AIC và CAM. Chạy các lệnh Python bên dưới từ thư mục `backend/`. Thay `aic` bằng `cam` khi cần. Đường dẫn trong config được tính tương đối từ `backend/`.

## 1. Thư mục dữ liệu tham khảo

```text
data/
├── keyframes/AIC/<split>/<video_id>/<frame>.jpg  # CAM nằm cạnh AIC
├── videos/AIC/<split>/<video_id>.*               # video gốc
├── mapkeyframes/AIC/<split>/<video_id>.csv
└── AIC/                                          # dữ liệu xử lý và index riêng
    ├── embeddings/<model>/<split>/<video_id>/<frame>.npy
    ├── ocr/<split>/<video_id>.json
    ├── asr/<split>/<video_id>.json
    └── database/<index_name>/                    # FAISS + metadata
```

`<split>` là tên phần dữ liệu, ví dụ `L21` hoặc `L22`. Giữ nguyên `video_id` và tên frame giữa keyframes, embeddings, map, OCR/ASR để metadata trỏ đúng ảnh/video.

## 2. Video → keyframes và map

Runtime và URL media dùng ba root `data/keyframes/<DATASET>/`, `data/videos/<DATASET>/`, `data/mapkeyframes/<DATASET>/`. Script trích keyframe hiện ghi `keyframes/` và `map_keyframes/` dưới `--output-dir`, nên chạy vào vùng staging rồi đưa hai thư mục kết quả vào root tương ứng:

```powershell
python scripts/keyframe_extraction/run.py --method distribution --video ..\data\videos\AIC\L21\L21_V001.mp4 --output-dir ..\data\staging\AIC
```

Kết quả chính:

```text
data/staging/AIC/keyframes/L21/L21_V001/*.jpg
data/staging/AIC/map_keyframes/L21/L21_V001.csv
```

Đặt nội dung `staging/AIC/keyframes/` vào `data/keyframes/AIC/` và `staging/AIC/map_keyframes/` vào `data/mapkeyframes/AIC/` trước khi build index hoặc chạy API. `--method` nhận `distribution` (mặc định), `kfeavi` hoặc `vlm` (pipeline embedding PE-Core, tên CLI cũ). Thêm `--overwrite` để tạo lại kết quả đã có.

## 3. Keyframes → embeddings

Script đọc các thư mục keyframe và ghi vector `.npy`. Cấu hình mẫu là `configs/embeddings.yaml`; trước khi chạy, chỉnh `input_keyframes_root`, `output_embeddings_root` và bật đúng model trong YAML cho dataset/model cần tạo. PE-Core yêu cầu package/weights riêng; SigLIP2 dùng preset khai báo trong config.

```powershell
python scripts/embedding_extraction/run.py --config configs/embeddings.yaml --model siglip2_so400m_384
```

Mỗi model ghi ra một thư mục riêng dưới `output_embeddings_root`, tên dạng `embeddings_<model-key>`. Ví dụ output mong muốn cho CAM:

```text
data/CAM/embeddings/embeddings_siglip2_so400m_384/<split>/<video_id>/<frame>.npy
```

Muốn chạy AIC, dùng config trỏ tới `data/keyframes/AIC` và output nằm dưới `data/AIC/embeddings`. Không dùng chung vector giữa các dataset hoặc giữa các model khác nhau.

Qwen3-VL-Embedding-2B dùng adapter ảnh/text trong registry chung. Đặt `input_keyframes_root` và `output_embeddings_root` cho AIC hoặc CAM trước khi trích. `engine: auto` ưu tiên server vLLM đang phục vụ đúng model; nếu server vắng thì chọn SentenceTransformer lúc khởi tạo. Đặt `QWEN_EMBEDDING_ENGINE=vllm` để bắt buộc vLLM (lỗi nếu không có server), hoặc `sentence_transformer` cho máy chỉ chạy fallback. `QWEN_VLLM_URL` mặc định là `http://127.0.0.1:8001/v1`. Backend giữ engine đã chọn suốt vòng đời; server vLLM ngắt giữa chừng sẽ báo lỗi, không tự chuyển engine giữa request. Vector 2048 chiều được lưu ở `embeddings_qwen3_vl_embedding_2b_<engine>`; không trộn hai engine. Việc trích embedding cần keyframe local; URL host dùng cho trình duyệt không thay thế đường dẫn này.

vLLM được pin **0.32.0** trong `requirements-vllm.txt`. Trên Kaggle/Linux, cài nó trong môi trường Python 3.12 riêng; không cài cùng `requirements.txt` vì vLLM có bộ PyTorch/CUDA phụ thuộc riêng. Hai môi trường giao tiếp bằng HTTP trên máy local. Ví dụ terminal vLLM:

```bash
python3.12 -m venv .venv-vllm
.venv-vllm/bin/pip install uv
.venv-vllm/bin/uv pip install --python .venv-vllm/bin/python -r requirements-vllm.txt --torch-backend=auto
.venv-vllm/bin/vllm serve Qwen/Qwen3-VL-Embedding-2B --runner pooling --max-model-len 8192 --host 127.0.0.1 --port 8001
```

Nếu chỉ dùng SentenceTransformer, cài `requirements.txt` cho backend rồi chọn `QWEN_EMBEDDING_ENGINE=sentence_transformer`; không cần cài hay chạy vLLM. `batch_size` trong `embeddings.yaml` là số ảnh pipeline gửi đồng thời; vLLM tự gộp các HTTP request thành batch theo scheduler. Chỉnh giá trị này sau khi đo throughput và bộ nhớ GPU.

```powershell
python scripts/embedding_extraction/run.py --config configs/embeddings.yaml --model qwen3_vl_embedding_2b
```

## 4. Embeddings → FAISS semantic index

`configs/indexing.yaml` chọn dataset bằng `--dataset`; kiểm tra ba đường dẫn `keyframes_root`, `embeddings_root`, `map_keyframes_root` cùng `index.output_dir` trong dataset đó khớp với đầu ra các bước trước. `index.output_dir` cần khớp `semantic_models.<model_key>.index_dir` trong `configs/app.yaml`.

```powershell
python scripts/retrieval/run.py --task build-index --config configs/indexing.yaml --dataset aic
python scripts/retrieval/run.py --task build-index --config configs/indexing.yaml --dataset cam
```

Để tạo index Qwen mà không đổi cấu hình build mặc định của PE-Core/SigLIP2, truyền đúng root embedding và thư mục index riêng của dataset. Ví dụ AIC:

```powershell
python scripts/retrieval/run.py --task build-index --dataset aic --embeddings-root ../data/AIC/embeddings/embeddings_qwen3_vl_embedding_2b_vllm --output-dir ../data/AIC/database/faiss_qwen3_vl_embedding_2b_vllm
```

Với CAM, thay `--dataset cam` và hai đường dẫn `AIC` bằng `CAM`. Nếu dùng fallback, thay hậu tố `_vllm` bằng `_sentence_transformer` ở cả embedding và index. API ghép hậu tố engine vào `semantic_models.qwen3-vl-embedding-2b.index_dir` trong `configs/app.yaml` khi khởi động. Qwen chỉ load thành công sau khi tạo index cùng engine; không dùng lại index/vector của model hoặc engine khác.

Index gồm `keyframes.faiss` và `metadata.csv`. Vector cache `vectors_fp32.npy` là tùy chọn theo runtime config; build bằng `--task build-vector-cache` khi cần và trỏ config tới đúng FAISS index.

## 5. Các modality khác

### OCR

OCR đọc keyframes, xuất một JSON cho mỗi video. Trước khi chạy, đặt `extraction.input_keyframes_root` và `extraction.output_root` trong `configs/ocr_extraction.yaml` cho dataset đang xử lý (ví dụ `../data/keyframes/AIC` và `../data/AIC/ocr`). Sau đó:

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
python scripts/color/extract.py --metadata ..\data\AIC\database\faiss_hnsw_pe_core_l14_336\metadata.csv --output ..\data\AIC\database\color_5x5.npz --keyframes-root ..\data\keyframes\AIC
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

- `configs/embeddings.yaml` mặc định đọc `data/keyframes/AIC`; khi xử lý dataset khác phải đổi roots trước khi chạy.
- Đối chiếu `configs/indexing.yaml` với `configs/app.yaml` trước khi build: index output phải đúng `index_dir` mà API dùng. Hai file hiện có một số đường dẫn không đồng nhất giữa AIC/CAM; đừng khởi động API với index vừa build nếu chúng khác nhau.
- Script keyframe sinh thư mục `map_keyframes` trong output staging; cấu hình runtime/index dùng root public `data/mapkeyframes/<DATASET>/`.
- Lệnh indexing Elasticsearch tạo index và ingest các JSON đang có. Chạy lại có thể đưa dữ liệu trùng vào index nếu không xóa/reset index trước.
