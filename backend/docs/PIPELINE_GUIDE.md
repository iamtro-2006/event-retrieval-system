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

## Trích xuất ASR

Từ `backend/`, chạy Faster-Whisper trên video trong `../data/videos/{V3C,LHE}/`:

```bash
python -m scripts.asr.extract.run --dataset v3c
python -m scripts.asr.extract.run --dataset lhe --video-id LHE01
```

Lệnh dùng model đa ngôn ngữ `Systran/faster-whisper-large-v3` với `compute_type: int8` theo `configs/asr_extraction.yaml`. `language: null` để Faster-Whisper tự nhận diện ngôn ngữ của từng video. Model được tải một lần rồi xử lý lần lượt các video chưa có kết quả; cần CUDA/cuDNN tương thích với Faster-Whisper để chạy cấu hình GPU. Mỗi video tạo `../data/asr/<dataset>/<video_id>.json`, gồm các đoạn `{video_id, start, end, transcript}` mà ASR indexer đọc trực tiếp. Dùng `--overwrite` để tạo lại file đã có.

Sau khi tạo JSON, lập chỉ mục theo dataset:

```bash
python -m scripts.asr.index.run --dataset v3c
python -m scripts.asr.index.run --dataset lhe
```

Trên Kaggle, `notebooks/extract_asr.ipynb` tải video V3C từ Drive theo dải ID và ghi JSON lên Drive theo lô 10 video. Notebook gọi cùng script với `--input-dir` và `--output-dir` để dùng thư mục tạm cho từng lô; model cache được giữ lại giữa các lô. Môi trường Kaggle chỉ cài `faster-whisper` khi cần, không tạo venv hay cài toàn bộ requirements của backend.

## Elasticsearch cho OCR và ASR

OCR và ASR là hai nguồn tùy chọn dùng Elasticsearch làm text database. Trên `vbs-dev` cả hai đang tắt (`ocr: null`, `asr: null` trong `configs/app.yaml`) vì chưa có data; bật lại bằng cách trỏ tới config tương ứng:

```yaml
ocr:
  config_path: configs/ocr_extraction.yaml
asr:
  config_path: configs/asr_extraction.yaml
```

### Chạy Elasticsearch trên máy mình

Không cần ai host hộ. ES 8.18 chạy bằng Docker, cấu hình nằm ở `backend/docker-compose.yaml`:

```bash
cd backend
docker compose up -d elastic          # chỉ ES, nhẹ (~1.2GB RAM)
# hoặc kèm Kibana để xem index bằng UI:
docker compose up -d
```

Kiểm tra: `curl http://localhost:9200/_cluster/health` trả `"status":"green"`.

Mặc định `configs/ocr_extraction.yaml` và `configs/asr_extraction.yaml` đã trỏ `localhost:9200`, nên sau `docker compose up` là dùng được ngay, không sửa file nào. Muốn trỏ máy khác thì override bằng biến môi trường, đừng sửa YAML đã commit:

```bash
ES_HOST=192.168.1.50 ES_PORT=9200 python -m scripts.ocr.index.run --dataset v3c
```

Dữ liệu ES nằm trong Docker volume `backend_esdata`, giữ lại qua các lần restart. `docker compose down -v` sẽ xoá sạch và phải seed lại.

### Seed dữ liệu

Sau khi có JSON từ bước extract (`data/ocr/<DATASET>/<video_id>.json` và `data/asr/<DATASET>/<video_id>.json`), chạy từ `backend/`:

```bash
python -m scripts.ocr.index.run --dataset v3c
python -m scripts.asr.index.run --dataset v3c
```

Dùng `python -m` (không phải `python scripts/...`) để `src` nằm trong `sys.path`. Index name lấy từ `datasets.available.<dataset>.{ocr,asr}_index` trong `configs/app.yaml` — hiện là `ocr_v3c`/`asr_v3c` và `ocr_lhe`/`asr_lhe`, nên mỗi dataset tách index riêng.

Script in `Total documents: <n>` khi xong. Nếu in `0` trong khi vừa index thì đó là lỗi refresh, không phải thiếu data.

Chạy lại lần hai sẽ **cộng dồn**, không thay thế: script chỉ tạo index nếu chưa có, nên document cũ vẫn nằm đó và số đếm tăng gấp đôi. Thêm `--recreate` để xoá index trước khi seed:

```bash
python -m scripts.ocr.index.run --dataset v3c --recreate
```

### Kiểm tra đã seed đúng

```bash
curl "http://localhost:9200/_cat/indices?v"
```

Hoặc search tay, dùng cùng index với API:

```bash
python -m scripts.ocr.search.run --dataset v3c
python -m scripts.asr.search.run --dataset v3c
```

Script in ra URL, index name và số document trước khi vào vòng nhập query. Nếu `documents: 0`, seed chưa chạy. Nếu `reachable: false`, ES chưa lên.

### Trạng thái qua API

`GET /api/health` và `GET /api/config` trả khối `ocr`/`asr` gồm `configured`, `available`, `url`, `index`, `reachable`, `index_exists`, `documents`, `error`. Dùng để phân biệt ba trạng thái: chưa bật (`configured: false`), bật nhưng index rỗng (`documents: 0`), và ES chết (`reachable: false`).

## Chạy dịch vụ

Backend dùng vLLM server riêng cho Qwen; xem `requirements-vllm.txt`. `host_server.ipynb` có lựa chọn `DATASET=V3C` hoặc `DATASET=LHE`, tải ZIP embedding/map tương ứng rồi build index và cache. Khi chọn LHE cần điền Drive folder ID embedding/map LHE ở cell cấu hình. Nếu thiếu index LHE, API vẫn có thể phục vụ V3C; request tìm kiếm LHE trả 503 cho đến khi index và vLLM sẵn sàng.

Frontend đọc danh sách dataset từ `/api/config`, cho phép chọn V3C hoặc LHE. URL media có dạng `/keyframes/LHE/...`, `/videos/LHE/...`, `/mapkeyframes/LHE/...` hoặc URL từ host media riêng. OCR, ASR và color là các nguồn tùy chọn với index tách riêng theo dataset; xem mục Elasticsearch ở trên để bật và seed.
