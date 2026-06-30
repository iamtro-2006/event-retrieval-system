# Backend - Event Retrieval System

Backend là thành phần phục vụ hệ thống truy xuất sự kiện từ video bằng cách kết hợp:
- trích xuất keyframe và embedding từ video,
- xây dựng indexing/vector database cho tìm kiếm ngữ nghĩa và truy xuất tương tự,
- hỗ trợ tìm kiếm semantic, temporal, OCR, ASR và các luồng truy xuất mở rộng.

Mục tiêu chính của backend là nhận query từ người dùng, tìm các keyframe liên quan trong kho dữ liệu video, rồi trả về metadata, timestamp, đường dẫn ảnh, video và thông tin ngữ cảnh để frontend hiển thị.

> Ghi chú kiến trúc: hiện tại hệ thống mặc định dùng FAISS làm backend indexing, nhưng pipeline và API được thiết kế để có thể thay bằng Milvus, Qdrant, Elastic hoặc các hệ thống vector database khác trong tương lai. Vì vậy, khi nói chung chung, nên dùng thuật ngữ indexing hoặc vector database thay vì cố định FAISS.

---

## 1. Tổng quan hệ thống

Hệ thống backend được thiết kế theo một pipeline dữ liệu rõ ràng:

1. Nhận video đầu vào từ thư mục dữ liệu gốc.
2. Trích xuất cảnh (scene) và keyframe từ video.
3. Tạo embedding cho từng keyframe bằng mô hình hình ảnh.
4. Xây dựng indexing/vector database và metadata để phục vụ tìm kiếm nhanh.
5. Cung cấp API cho frontend thực hiện truy vấn và nhận kết quả.

Backend hiện có hai lớp chính:
- Lớp xử lý dữ liệu và index: dùng để tạo keyframe, embedding và indexing layer.
- Lớp dịch vụ truy xuất: nhận query, thực hiện search, sắp xếp kết quả và trả về payload phù hợp cho UI.

---

## 2. Cấu trúc thư mục backend

```text
backend/
├── main.py                  # FastAPI entrypoint chính
├── mock_api.py              # API mock phục vụ thử nghiệm/UI
├── requirements.txt         # Dependencies đầy đủ
├── requirements-fast.txt    # Phiên bản nhẹ hơn cho nhanh cài đặt
├── RUN_FAST.md              # Hướng dẫn chạy nhanh
├── configs/                 # File cấu hình cho toàn bộ pipeline
│   ├── app.yaml
│   ├── embeddings.yaml
│   ├── indexing.yaml
│   └── kf_extraction.yaml
├── data/                    # Dữ liệu đầu vào/đầu ra của hệ thống
│   ├── raw/                 # Video gốc
│   ├── processed/           # Keyframe, embeddings, map keyframes
│   └── database/            # indexing data, metadata, vector cache
├── evaluation/              # Script đánh giá và kết quả
├── external/                # Thư viện bên ngoài như TransNetV2
├── logs/                    # Log hệ thống
├── scripts/                 # Entry point cho các pipeline
│   ├── keyframes/
│   ├── embeddings/
│   └── retrieval/
├── src/                     # Mã nguồn chính của backend
│   ├── asr/                 # Module ASR
│   ├── embeddings/          # Mô hình embedding và pipeline encode
│   ├── keyframes/           # Scene detection, keyframe selection
│   ├── logic/               # Query parsing, scoring, temporal logic
│   ├── ocr/                 # Module OCR
│   ├── reranker/            # Reranking bằng mô hình nâng cao
│   ├── retrieval/           # Indexing và retrieval engine
│   ├── utils/               # Logger, config, video I/O, seed
│   └── ui/                  # Thành phần UI/helper nếu có
└── weights/                 # Weight mô hình như TransNetV2
```

### Vai trò của các thư mục quan trọng
- configs/: lưu toàn bộ cấu hình cho runtime, embedding, index, keyframe extraction.
- data/: là nơi chứa dữ liệu thực thi của pipeline.
- scripts/: chạy các pipeline riêng lẻ mà không cần sửa code.
- src/: chứa logic nghiệp vụ chính.
- weights/: chứa trọng số mô hình.

---

## 3. Cách cài đặt

### Yêu cầu
- Python 3.10 hoặc 3.11 được khuyến nghị.
- Nếu dùng GPU, nên cài đúng version PyTorch tương thích với CUDA.

### Bước 1: Tạo môi trường ảo

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```

### Bước 2: Cài đặt dependency cơ bản

```bash
python -m pip install -U pip setuptools wheel
pip install -r requirements-fast.txt
```

> Lưu ý: PyTorch nên được cài riêng trước, vì phiên bản wheel phụ thuộc vào hệ điều hành, GPU và CUDA. Nếu dùng GPU, hãy chọn đúng command cài từ trang PyTorch chính thức.

### Bước 3: Kiểm tra môi trường

```bash
python -c "import torch; print(torch.__version__)"
python -c "import faiss; print('faiss ok')"
```

### Bước 4: Chuẩn bị file môi trường (nếu cần)

Nếu hệ thống dùng token Hugging Face hoặc biến môi trường bổ sung, có thể tạo file .env trong thư mục backend với nội dung tương ứng.

---

## 4. Cách chạy backend

### Chạy API chính

Từ thư mục backend:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Chạy API mock

```bash
uvicorn mock_api:app --host 0.0.0.0 --port 8001
```

### Endpoint quan trọng

Sau khi chạy, backend cung cấp các endpoint như:
- /api/health: kiểm tra trạng thái hệ thống.
- /api/config: lấy cấu hình công khai cho frontend.
- /api/search: thực hiện query search.
- /api/similarity-search: tìm các frame tương tự.

Các endpoint cụ thể có thể thay đổi theo version, nên nên kiểm tra trực tiếp trong file main.py khi cần sử dụng chi tiết.

---

## 5. Pipeline tổng thể của backend

### 5.1 Pipeline trích xuất keyframe

Module chính: src/keyframes/

Luồng:
1. Đọc danh sách video từ thư mục input.
2. Chuyển video sang định dạng phù hợp nếu cần.
3. Dùng TransNetV2 phát hiện scene.
4. Tạo embedding cho các frame trong video.
5. Chọn keyframe đại diện cho từng scene.
6. Lưu map keyframe, ảnh keyframe và metadata liên quan.

Entry point:
```bash
python scripts/keyframes/run.py
```

### 5.2 Pipeline trích xuất embedding

Module chính: src/embeddings/

Luồng:
1. Scan thư mục chứa ảnh keyframe.
2. Load mô hình embedding.
3. Encode từng batch hình ảnh.
4. Lưu kết quả embedding dưới dạng file có thể dùng cho index.

Entry point:
```bash
python scripts/embeddings/run.py
```

### 5.3 Pipeline xây dựng indexing/vector database

Module chính: src/retrieval/

Luồng:
1. Thu thập các embedding đã tạo.
2. Kết hợp embedding với metadata như video_id, frame_id, path ảnh.
3. Tạo indexing backend theo metric cosine; hiện tại mặc định là FAISS, nhưng có thể thay bằng backend khác.
4. Lưu file index và file metadata phù hợp với backend đang dùng.
5. Tuỳ chọn build vector cache để tối ưu temporal search.

Entry point:
```bash
python scripts/retrieval/run.py --task build-index
```

### 5.4 Pipeline truy xuất dữ liệu

Module chính: src/retrieval/ và src/logic/

Luồng:
1. Nhận query đầu vào từ frontend.
2. Phân tích query thành các câu truy vấn con (semantic hoặc temporal).
3. Encode query bằng mô hình embedding text.
4. Thực hiện tìm kiếm trên FAISS index.
5. Nếu cần, chạy temporal search hoặc reranking.
6. Trả về danh sách frame, timestamp, score và đường dẫn media.

Các chế độ tìm kiếm chính:
- semantic: tìm theo ngữ nghĩa của query.
- temporal: tìm theo chuỗi sự kiện theo thời gian.
- ocr/asr: các module sẵn có cho xử lý ngữ liệu và văn bản trong media.
- auto: chọn luồng phù hợp tùy query.

---

## 6. Vai trò các module chính

### src/embeddings/
- load_clip_model: nạp mô hình embedding hình ảnh/text.
- encode_video_frames: encode frame từ video.
- extract_embeddings: pipeline chuẩn hóa dữ liệu embedding.

### src/keyframes/
- detector.py: phát hiện scene bằng TransNetV2.
- selector.py: chọn keyframe phù hợp cho mỗi scene.
- extract_keyframes.py: pipeline tổng hợp để tạo keyframe từ video.

### src/retrieval/
- faiss_index.py: tạo metadata, build matrix, build index FAISS.
- retrieval_system.py: engine tìm kiếm chính, xử lý cache vector, query plan và search.
- build_faiss.py: pipeline tạo chỉ mục.
- build_vector_cache.py: tạo vector cache phục vụ temporal search.

### src/logic/
- query_parser.py: tách query thành các mục con.
- temporal_search.py: tìm kiếm theo đồ thị thời gian / chuỗi sự kiện.
- scoring.py: tính điểm và rerank kết quả.

### src/utils/
- config.py: đọc cấu hình YAML.
- logger.py: cấu hình logging.
- video_io.py: đọc/chuẩn hóa video.
- device.py: chọn device CPU/GPU.

### src/reranker/
- module dùng cho reranking chất lượng cao sau khi lấy candidate ban đầu.

### src/asr/ và src/ocr/
- dành cho xử lý speech và OCR, có thể tích hợp vào pipeline truy xuất sâu hơn.

---

## 7. File cấu hình quan trọng

### configs/app.yaml
- cấu hình runtime của API.
- đường dẫn tới index FAISS, metadata, thư mục keyframe/video.
- cấu hình model, search, speech, debug.

### configs/embeddings.yaml
- cấu hình cho quá trình tạo embedding.
- đặt input/output directory và thông số batch size, model.

### configs/indexing.yaml
- cấu hình cho việc build FAISS index và vector cache.
- có các thông số như metric, type, hnsw_m, ef_construction, ef_search.

### configs/kf_extraction.yaml
- cấu hình cho quá trình phát hiện scene và chọn keyframe.
- điều chỉnh threshold, batch size, save options và đường dẫn output.

---

## 8. Dữ liệu đầu vào/đầu ra

### Thư mục đầu vào
- data/raw/videos: video gốc cần xử lý.

### Thư mục đầu ra
- data/processed/keyframes: ảnh keyframe đã chọn.
- data/processed/map_keyframes: file mapping giữa video và keyframe.
- data/processed/embeddings: embedding được lưu.
- data/database: index FAISS và metadata.

### File nặng / phụ trợ
- weights/transnetv2-pytorch-weights.pth: trọng số cho scene detector.
- external/TransNetV2: mã nguồn hỗ trợ scene detection.

---

## 9. Ghi chú vận hành

- Nếu chưa tạo index FAISS thì API sẽ không thể thực hiện retrieval đầy đủ.
- Nếu muốn dùng temporal search hiệu quả, nên build vector cache riêng sau khi build index.
- Khi chạy benchmark hoặc production, nên tránh dùng --reload để giảm chi phí load model nhiều lần.
- Nếu dùng GPU, nên kiểm tra lại device và precision trong config để tránh hiệu năng thấp.

---

## 10. Tóm tắt nhanh

Backend hiện tại hoạt động như một hệ thống truy xuất sự kiện video end-to-end với các bước chính:

```text
Video -> Scene Detection -> Keyframe Selection -> Embedding Extraction -> FAISS Index -> Query Search -> Ranked Results
```

Nó phù hợp cho các ứng dụng như:
- tìm kiếm video/keyframe bằng ngữ nghĩa,
- truy xuất sự kiện theo thời gian,
- phục vụ UI hiện thị kết quả theo thumbnail và timestamp.
