# Backend Architecture — Event Retrieval System

**Tài liệu handover kỹ thuật chi tiết.** README.md ở root là bản tóm tắt
"đọc trong 5 phút"; tài liệu này là bản đầy đủ để một team mới có thể tiếp
quản, sửa bug, và mở rộng hệ thống mà không cần hỏi lại người viết ban đầu.

> Quy ước đọc: mọi đường dẫn trong tài liệu này là tương đối so với thư mục
> `backend/` (chứa `main.py`). Tên hàm/class được viết `code_style` và trỏ
> thẳng tới file thật trong repo tại thời điểm viết tài liệu (có thể lệch nếu
> code đã đổi — xem mục 11 "Cách giữ tài liệu này không bị stale").

---

## Mục lục

1. [Tổng quan hệ thống & tech stack](#1-tổng-quan-hệ-thống--tech-stack)
2. [Sơ đồ luồng dữ liệu end-to-end](#2-sơ-đồ-luồng-dữ-liệu-end-to-end)
3. [Offline pipelines (build dữ liệu / index)](#3-offline-pipelines-build-dữ-liệu--index)
4. [Lõi retrieval — cấu trúc & nguyên tắc](#4-lõi-retrieval--cấu-trúc--nguyên-tắc)
5. [Chi tiết từng search block](#5-chi-tiết-từng-search-block)
6. [Indexer — ghi dữ liệu vào các store](#6-indexer--ghi-dữ-liệu-vào-các-store)
7. [Tầng API (`main.py`)](#7-tầng-api-mainpy)
8. [Translation subsystem](#8-translation-subsystem)
9. [Cấu hình (configs/)](#9-cấu-hình-configs)
10. [Vấn đề đã biết / nợ kỹ thuật](#10-vấn-đề-đã-biết--nợ-kỹ-thuật)
11. [Cách giữ tài liệu này không bị stale](#11-cách-giữ-tài-liệu-này-không-bị-stale)
12. [Hướng dẫn thêm 1 search block mới (chi tiết mã giả)](#12-hướng-dẫn-thêm-1-search-block-mới-chi-tiết-mã-giả)

---

## 1. Tổng quan hệ thống & tech stack

Hệ thống retrieval video theo 4 kiểu truy vấn trên cùng 1 tập keyframe:

| Kiểu search | Ý nghĩa | Backend lưu trữ |
|---|---|---|
| `semantic` | Tìm keyframe theo nội dung hình ảnh (text→ảnh, CLIP) | FAISS |
| `temporal` | Tìm 1 **chuỗi** sự kiện theo đúng thứ tự thời gian trong 1 video | FAISS (DP trên ma trận điểm) |
| `ocr` | Tìm theo chữ xuất hiện trên màn hình | Elasticsearch |
| `asr` | Tìm theo lời thoại (giọng nói) | Elasticsearch |
| `auto` | Tự chọn `semantic` hoặc `temporal` dựa trên câu query | — |

### 1.1 Tech stack thực tế (trích từ import trong code, không phải suy đoán)

| Layer | Thư viện / công nghệ | Vai trò |
|---|---|---|
| Web framework | **FastAPI** + Pydantic (`BaseModel`) | REST API, validate request |
| Vector search | **FAISS** (`faiss`), HNSW index | ANN search trên embedding CLIP |
| Embedding model | **OpenCLIP** (`open_clip`) — `ViT-L-16-SigLIP-256`, pretrained `webli` | encode text/ảnh thành vector |
| Deep learning runtime | **PyTorch** (`torch`, `torchvision`) | chạy CLIP, TransNet, reranker |
| Numeric tăng tốc | **NumPy**, **SciPy**, **scikit-learn** (`sklearn`), **Numba** (`@njit`) | xử lý ma trận, DP tốc độ C |
| Dữ liệu bảng | **pandas** | toàn bộ kết quả search là `pd.DataFrame` |
| Full-text search | **Elasticsearch** (`elasticsearch` client) | index/search OCR & ASR |
| Vector DB thay thế | **Milvus** (`pymilvus` qua `src/retrieval/indexer/milvus`) | backend ghi index thay thế cho FAISS (hiện có indexer, chưa thấy retriever tương ứng — xem mục 10) |
| OCR | **PaddleOCR** (detect) + **VietOCR** (`vietocr`, nhận dạng ký tự tiếng Việt) | trích chữ trên khung hình |
| ASR (speech-to-text) | **faster-whisper** (`faster_whisper`) | model Whisper tối ưu CTranslate2, dùng cho cả pipeline offline lẫn endpoint `/api/speech/transcribe` (query bằng giọng nói) |
| Scene/keyframe detection | **TransNetV2** (custom, load qua `torch.hub`-style loader trong `detector.py`) + thuật toán cluster riêng | phát hiện chuyển cảnh, chọn keyframe đại diện |
| Video I/O | **decord**, **ffmpeg-python**, **OpenCV** (`cv2`) | đọc frame video hiệu quả |
| Rerank (tuỳ chọn, CHƯA wire) | **InternVL** qua `transformers` | rerank kết quả bằng VLM — xem mục 10 |
| Dịch câu truy vấn | LibreTranslate (HTTP server) hoặc **Hy-MT2** GGUF model qua `llama_cpp` (chạy local, CPU/GPU) | dịch query VI→EN trước khi encode CLIP (CLIP train trên tiếng Anh) |
| Config | YAML (`pyyaml`) + `python-dotenv` | mỗi subsystem 1 file config riêng |
| Nộp bài đánh giá | **DRES** (Distributed Retrieval Evaluation Server, qua `requests`) | tích hợp thi đấu dạng competition (AI Challenge / VBS) |

> Không tìm thấy `requirements.txt` trong bản zip được cung cấp — danh sách
> trên được suy ra trực tiếp từ `import` trong mã nguồn. **Việc đầu tiên team
> kế tiếp nên làm là `pip freeze` môi trường đang chạy production và thêm
> `requirements.txt`/`pyproject.toml` vào repo**, vì hiện không có cách nào
> khác để biết version chính xác của từng thư viện.

### 1.2 Nguyên tắc kiến trúc xuyên suốt (bắt buộc phải hiểu trước khi sửa code)

1. **`system.py` là ranh giới duy nhất** giữa tầng API (`main.py`) và toàn bộ
   `src/retrieval/`. `main.py` không bao giờ import trực tiếp từ
   `retriever/<x>_search/` hay `indexer/`.
2. **Đọc (`retriever/`) và Ghi (`indexer/`) tách biệt tuyệt đối.** Không file
   nào trong `retriever/` được phép insert/bulk_insert vào index.
3. **4 backend search (semantic/temporal/ocr/asr) có cùng 1 hình dạng**: mỗi
   block có `factory.py` (khởi tạo dependency) + `pipeline/search.py` (thuật
   toán thuần + đúng 1 class `SearchPipeline` với method `.search(...)`).
   `Orchestrator` gọi cả 4 theo cùng 1 cách.
4. **Model/index chỉ được load 1 lần** lúc khởi động (`build_system` trong
   `system.py`, gọi từ FastAPI lifespan trong `main.py`), không load lại mỗi
   request.
5. **OCR và ASR là subsystem optional**: nếu Elasticsearch không sẵn sàng lúc
   khởi động, `system.py` bắt exception, log cảnh báo, và trả `None` cho
   pipeline đó — server vẫn chạy được với `semantic`/`temporal`, chỉ có
   `mode="ocr"`/`"asr"` sẽ raise `RuntimeError` khi gọi.
6. **Mọi thuật toán là hàm thuần (pure function)**; class chỉ điều phối
   (orchestrate), không chứa logic tính toán bên trong nó — giúp test từng
   hàm độc lập không cần mock index/model.

---

## 2. Sơ đồ luồng dữ liệu end-to-end

```
┌─────────────┐   ┌────────────────────┐   ┌─────────────────────┐
│   video/*    │──▶│ keyframe_extraction │──▶│ embedding_extraction │
│ (raw videos) │   │ (TransNetV2 + DP    │   │ (OpenCLIP encode     │
│              │   │  cluster + histo    │   │  từng keyframe ->    │
│              │   │  dedup)             │   │  vector .npy)        │
└─────────────┘   └────────────────────┘   └──────────┬───────────┘
                                                        │
                        ┌───────────────────────────────┼───────────────────────┐
                        ▼                               ▼                       ▼
              ┌──────────────────┐          ┌──────────────────┐    ┌────────────────────┐
              │ indexer/faiss     │          │ ocr_extraction    │    │ (audio track from   │
              │ build_pipeline.py │          │ (PaddleOCR detect │    │  video, tách bằng    │
              │ -> keyframes.faiss│          │  + VietOCR nhận   │    │  ffmpeg)             │
              │    + metadata.csv │          │  dạng)            │    └──────────┬──────────┘
              └────────┬──────────┘          └────────┬──────────┘               │
                       │                              ▼                          ▼
                       │                   ┌─────────────────────┐   ┌──────────────────────┐
                       │                   │ indexer/elasticsearch│   │ faster-whisper ASR    │
                       │                   │   /ocr (bulk index)  │   │ (scripts/asr) ->      │
                       │                   └─────────────────────┘   │ indexer/elasticsearch │
                       │                                              │   /asr (bulk index)   │
                       │                                              └──────────┬────────────┘
                       ▼                                                         ▼
        ┌───────────────────────────────────────────────────────────────────────────┐
        │                      RUNTIME — main.py (FastAPI, load 1 lần)                │
        │                                                                             │
        │   build_system(config) -> RetrievalSystem                                   │
        │      ├── ClipFaissIndex  (load faiss index + metadata.csv + model CLIP)     │
        │      ├── semantic_search.SearchPipeline(index)                              │
        │      ├── temporal_search.SearchPipeline(index)                              │
        │      ├── ocr_search.SearchPipeline(OCRRepository)      [optional]           │
        │      └── asr_search.SearchPipeline(ASRRepository)      [optional]           │
        │                                                                             │
        │   POST /api/search --> Orchestrator.run_search(query, mode, ...)            │
        │      -> chọn 1 trong 4 SearchPipeline.search(...) theo `mode`               │
        │      -> trả pd.DataFrame + QueryPlan                                        │
        │      -> main.py serialize DataFrame -> JSON (dict_to_result_FAST, ...)      │
        └───────────────────────────────────────────────────────────────────────────┘
```

**Điểm mấu chốt cần nhớ**: pha *offline* (keyframe → embedding → index) và
pha *online* (API search) hoàn toàn tách rời qua 2 artefact trung gian:
`keyframes.faiss` + `metadata.csv` (cho semantic/temporal) và các index
Elasticsearch `ocr-*`/`asr-*` (cho ocr/asr). Muốn cập nhật dữ liệu mới, chạy
lại `scripts/` offline rồi restart API — API không tự động re-index.

---

## 3. Offline pipelines (build dữ liệu / index)

Tất cả nằm dưới `scripts/<tên>/run.py`, chỉ là CLI mỏng gọi vào class
pipeline tương ứng trong `src/<tên>/pipeline/`. Không có business logic
trong `scripts/`.

### 3.1 Keyframe extraction — `src/keyframe_extraction/`

**Input:** 1 file video (`.mp4`...).
**Output:** danh sách file ảnh keyframe (`data/processed/keyframes/<video_id>/*.jpg`)
+ 1 file `map_keyframes/<video_id>.csv` map `keyframe_id -> frame_idx, timestamp_sec, fps`.

Thuật toán (mã giả, xem `models/detector.py` + `models/clustering.py` +
`models/selector.py`):

```
detect_scenes(video):
    frames = decode_video(video)                 # decord/ffmpeg
    predictions = TransNetV2(frames)              # xác suất "frame là ranh giới scene"
    scenes = predictions_to_scenes(predictions, threshold)
    scenes = split_large_scenes(scenes, max_gap)  # scene quá dài -> chia nhỏ thêm

for each scene:
    features = CNN_features(frames_in_scene)      # đặc trưng thị giác thô, dùng cho cluster
    centroids = init_centroids(features)           # k-means init
    cluster_ids = select_cluster_keyframes(features, min_frames=3)
    candidate_indexes = 1 frame đại diện / cluster

    # loại các keyframe gần giống nhau về mặt histogram màu
    final_indexes = histogram_dedup(video_path, candidate_indexes, threshold, min_bins)

save_keyframe_images(final_indexes, video_path, output_dir, image_quality)
save_keyframe_map(final_indexes, video_path, csv_path)
```

Điều phối bởi `KeyframeExtractionPipeline` (`pipeline/extract_keyframes.py`),
config ở `configs/kf_extraction.yaml`. Chạy: `python scripts/keyframe_extraction/run.py`.

### 3.2 Embedding extraction — `src/embedding_extraction/`

**Input:** thư mục keyframe ảnh của 1/nhiều video.
**Output:** 1 file `.npy` vector/keyframe (cùng tên, khác thư mục gốc), dùng
làm input cho `indexer/faiss`.

```
model = load_clip_model(model_name, pretrained, precision, device)  # OpenCLIP
loader = build_frame_loader(image_dir, batch_size, num_workers)     # torch DataLoader

for batch in iter_frame_batches(loader):
    tensors = preprocess(batch.images)          # resize/normalize theo CLIP
    with autocast(precision):
        vectors = model.encode_image(tensors)
    vectors = l2_normalize(vectors)              # bắt buộc để dùng cosine qua inner-product FAISS
    save_embedding(vectors, output_path)
```

Điều phối bởi `ExtractEmbeddingPipeline`, config `configs/embeddings.yaml`.
Model phải **cùng `model_name`/`pretrained`** với model dùng lúc runtime
(`configs/app.yaml -> model.*`) — lệch model là nguyên nhân phổ biến nhất
khiến semantic search trả kết quả vô nghĩa dù index build thành công.

### 3.3 OCR extraction — `src/ocr_extraction/`

**Input:** keyframe ảnh. **Output:** danh sách text/box theo từng keyframe,
được `indexer/elasticsearch/ocr` bulk-insert.

```
for image in ImageDataset(keyframe_dir):        # models/dataset.py
    boxes = paddleocr_detect(image)              # models/engine.py: chỉ DETECT vùng chữ
    crops = [crop_box(image, b) for b in boxes]  # models/vietocr_engine.py
    texts = vietocr_recognize(crops)             # nhận dạng ký tự tiếng Việt trên từng crop
    yield {video_id, keyframe_id, texts, boxes}
```

Lý do tách 2 model: PaddleOCR detect tốt vùng chữ đa ngôn ngữ nhưng recognizer
mặc định kém tiếng Việt có dấu; VietOCR chuyên recognize tiếng Việt, cần
PaddleOCR cấp toạ độ box trước. Điều phối bởi `ExtractOCRPipeline`, config
`configs/ocr_extraction.yaml`.

### 3.4 ASR extraction — `scripts/asr/`

**Input:** audio tách từ video (ffmpeg). **Output:** các segment thoại
`{video_id, start_time, end_time, text}`, được `indexer/elasticsearch/asr`
bulk-insert. Dùng chung model `faster_whisper.WhisperModel` với endpoint
runtime `/api/speech/transcribe` (config `speech.*` trong `app.yaml`:
`model_size`, `device`, `compute_type`, `lazy_load`).

### 3.5 Bảng tổng hợp offline pipeline

| Pipeline | Input | Output | Config | Class điều phối |
|---|---|---|---|---|
| Keyframe extraction | video | ảnh keyframe + map CSV | `configs/kf_extraction.yaml` | `KeyframeExtractionPipeline` |
| Embedding extraction | ảnh keyframe | vector `.npy` | `configs/embeddings.yaml` | `ExtractEmbeddingPipeline` |
| FAISS indexing | vector `.npy` + map CSV | `keyframes.faiss` + `metadata.csv` | `configs/indexing.yaml` | `BuildFaissIndexPipeline` |
| OCR extraction + index | ảnh keyframe | ES index `ocr-*` | `configs/ocr_extraction.yaml` + `configs/ocr.yaml` | `ExtractOCRPipeline` + `IndexPipeline` (ES) |
| ASR extraction + index | audio | ES index `asr-*` | `configs/asr_extraction.yaml` + `configs/asr.yaml` | script `scripts/asr` + `IndexPipeline` (ES) |

---

## 4. Lõi retrieval — cấu trúc & nguyên tắc

```
src/retrieval/
├── system.py            # FACADE — điểm vào duy nhất từ main.py
├── base/
│   ├── base_retriever.py   # interface BaseRetriever + dataclass Hit chuẩn hoá
│   └── base_indexer.py     # interface BaseIndexer (create_index/insert/bulk_insert/count)
├── index/
│   ├── clip_faiss_index.py # ClipFaissIndex — HẠ TẦNG dùng chung (không có search())
│   └── factory.py           # build_clip_faiss_index(cfg)
├── indexer/              # GHI — xem mục 6
│   ├── faiss/
│   ├── elasticsearch/{ocr,asr}/
│   └── milvus/
└── retriever/             # ĐỌC — xem mục 5
    ├── semantic_search/
    ├── temporal_search/
    ├── ocr_search/
    ├── asr_search/
    ├── reranker/           # chưa wire — mục 10
    └── common/
        ├── orchestrator.py   # chọn mode + dispatch + enrich OCR/ASR
        ├── query_parser.py   # split_query() — tách câu theo liên từ thời gian
        └── scoring.py        # rerank_multi_query() — hợp nhất điểm nhiều sub-query
```

### 4.1 `ClipFaissIndex` — hạ tầng dùng chung (`index/clip_faiss_index.py`)

Đây là lớp thay thế `SemanticIndex` cũ. **Chỉ chịu trách nhiệm hạ tầng**, không
có method `search()` nghiệp vụ:

- Load FAISS index (`.faiss`) + metadata (`metadata.csv`) + model OpenCLIP.
- `encode_text(query) / encode_texts(queries) / encode_image(path)` — encode
  qua CLIP, có normalize L2 nếu `normalize: true`.
- Quản lý **vector cache** để phục vụ các thuật toán cần đọc lại chính vector
  đã index (vd. temporal search cần similarity giữa các keyframe liên tiếp
  trong cùng video) — 2 chế độ: `ram` (load hết vào RAM qua
  `_reconstruct_index_vectors`) hoặc `memmap` (đọc từ file `.npy` qua
  `_load_vector_memmap`, tiết kiệm RAM, đánh đổi I/O).
- `search_lock` (`RLock`): FAISS index không thread-safe khi vừa search vừa
  bị truy cập đồng thời ở mức C-level trên một số cấu hình; mọi lệnh gọi
  `index.search(...)` phải giữ lock này (xem `faiss_search()` trong
  `semantic_search/pipeline/search.py`).
- `metadata_row_for_ocr_hit(video_id, keyframe_id)` và
  `metadata_rows_for_asr_hit(video_id, start, end)`: cầu nối để
  `Orchestrator` "làm giàu" (enrich) kết quả thô từ Elasticsearch bằng
  metadata FAISS (đường dẫn ảnh, fps, timestamp...) — đây là lý do OCR/ASR
  trả về kết quả **có cùng hình dạng cột** với semantic/temporal dù dữ liệu
  gốc nằm ở Elasticsearch.
- Hằng số `METADATA_DISPLAY_COLUMNS` = tập cột chuẩn mọi kết quả search phải
  có, dùng để đồng bộ format giữa 4 loại search.

### 4.2 `Orchestrator` (`retriever/common/orchestrator.py`)

Là "bộ não" chọn mode + dispatch. Không tự chứa thuật toán search — chỉ biết
gọi đúng pipeline và (với OCR/ASR) join dữ liệu về FAISS metadata.

**`QueryPlan`** — cấu trúc hoá 1 câu query thành các "event" (sự kiện theo
thời gian) × "sub-query" (biến thể ngữ nghĩa trong 1 event):

```
build_query_plan(query, mode, use_split):
    events_text = split_temporal_events(query)      # tách theo . ; -> mỗi phần là 1 "event"
    for text in events_text:
        parts = split_semantic_queries(text) if use_split else [text]
        # split theo dấu phẩy -> các cách diễn đạt khác nhau của CÙNG 1 event
        events.append(parts)
    return QueryPlan(query, mode, use_split, events)
```

Ví dụ: `"a man opens the door, walks in; then he sits down"`
→ 2 events: `["a man opens the door, walks in", "a man opens the door", "walks in"]`
và `["then he sits down", "then he sits down"]` (sau khi lọc liên từ).
`event_queries` lấy phần tử đầu mỗi event (dùng cho temporal DP theo thứ tự);
`flat_queries` gộp phẳng tất cả sub-query (dùng cho semantic multi-query).

**`run_search(query, mode, ...)`** — luồng chính:

```
plan = build_query_plan(query, mode, use_split)
if plan rỗng: return DataFrame rỗng

effective_mode = mode
if mode == "auto":
    effective_mode = "temporal" if len(plan.events) > 1 else "semantic"

dispatch effective_mode:
  semantic  -> semantic_search.search(plan.events, top_k, candidate_k)
  temporal  -> temporal_search.search(plan.events, top_k, candidate_k, duration_limit)
  ocr       -> self.ocr_search(plan.query, top_k)      # dùng RAW query, không split
  asr       -> self.asr_search(plan.query, top_k)      # dùng RAW query, không split
```

> **Vì sao OCR/ASR dùng `plan.query` thô** thay vì `plan.events`/`flat_queries`?
> Elasticsearch tự làm tokenization/fuzzy-matching/BM25 riêng, việc split theo
> literal `and/then/,` (vốn thiết kế cho ngữ nghĩa CLIP) sẽ làm mất ngữ cảnh
> câu đối với full-text search.

**Enrich OCR/ASR** (`_enrich_ocr_hits`, `_enrich_asr_hits`): xem mục 5.3–5.4,
đây là phần phức tạp nhất trong orchestrator và là nơi dễ phá vỡ hành vi nhất
nếu sửa nhầm.

### 4.3 `base_retriever.Hit` / `base_indexer.BaseIndexer`

- `Hit(id, score, metadata)`: dataclass chuẩn hoá, **mục tiêu tương lai** là
  mọi retriever trả về `list[Hit]`. Hiện tại **chỉ là interface tham chiếu** —
  `semantic_search`/`temporal_search` trả `pd.DataFrame` (không phải
  `list[Hit]`), `ocr_search`/`asr_search` trả `list[TypedDict]` riêng
  (`OCRHit`, `ASRHit`). Đây là nợ kỹ thuật đã được ghi chú rõ trong code —
  xem mục 10.
- `BaseIndexer`: interface bắt buộc mọi backend ghi (`FaissIndexer`,
  `ElasticsearchService`/`OCRRepository`, `MilvusRepository`...) phải hiện
  thực `create_index / insert / bulk_insert / count`.

---

## 5. Chi tiết từng search block

### 5.1 Semantic search (`retriever/semantic_search/`)

**Input:** `events: list[list[str]]` (từ `QueryPlan`), `top_k`, `candidate_k`.
**Output:** `pd.DataFrame`, mỗi dòng 1 keyframe, sắp theo `alignment_score` giảm dần.

Thuật toán (`pipeline/search.py`):

```
queries = clean_queries(flatten(events))                # dedupe, strip, casefold-compare
embeddings = index.encode_texts(queries)                 # (n_queries, dim), CLIP text encoder

scores, indices = faiss_search(index, lock, embeddings, candidate_k)  # (n_queries, candidate_k) mỗi loại

# aggregate_multi_query — toàn bộ vectorized bằng NumPy, KHÔNG loop Python trên từng candidate:
for mỗi keyframe id xuất hiện trong candidate pool của >=1 query:
    avg_score   = trung bình score qua các query match được id đó
    max_score   = score cao nhất
    matched     = số query khác nhau match được id đó
    coverage    = matched / tổng_số_query
    alignment   = 0.8 * avg_score + 0.2 * coverage        # trade-off: điểm cao NHƯNG cũng
                                                            # nên match được nhiều cách diễn đạt
sort theo alignment giảm dần, lấy top_k
```

`SearchPipeline.search(events, top_k, candidate_k)` là entrypoint mà
`Orchestrator` gọi. Ngoài ra còn `similarity_search_by_image(image_path,
top_k)` dùng cho endpoint `/api/similarity-search` (tìm keyframe tương tự 1
keyframe cho trước — "more like this").

Cột output chính: tất cả cột trong `metadata.csv` (dataset, video_id,
keyframe_id, keyframe_path, frame_idx, timestamp_sec, fps...) + `avg_score`,
`max_score`, `matched_queries`, `coverage_score`, `alignment_score`,
`retrieval_score` (= alignment_score, dùng cho FE), `display_rank`, `rank`.

### 5.2 Temporal search (`retriever/temporal_search/`)

**Bài toán**: cho một chuỗi N sự kiện theo thứ tự (`event_queries` từ
`QueryPlan`), tìm trong mỗi video 1 chuỗi N keyframe **giữ đúng thứ tự thời
gian** sao cho tổng độ tương đồng ngữ nghĩa là lớn nhất — bài toán "Longest
Increasing Subsequence có trọng số", giải bằng **Dynamic Programming**, được
biên dịch JIT bằng **Numba** (`@njit(fastmath=True, nogil=True, cache=True)`)
để đạt tốc độ gần C, bỏ qua GIL Python.

Mã giả DP (input `S`: ma trận similarity `(m sự kiện) x (n keyframe trong 1 video)`,
đã sort theo thời gian):

```
dp[0][j] = S[0][j]                          # sự kiện đầu tiên có thể là bất kỳ frame nào
for qi in 1..m-1:
    running_max = -inf
    for j in 0..n-1:
        running_max = max(running_max, dp[qi-1][j-1])   # tốt nhất trong số các frame TRƯỚC j
        if running_max khả dụng:
            dp[qi][j] = running_max + S[qi][j]           # bắt buộc: frame của sự kiện qi phải
                                                           # đứng SAU frame của sự kiện qi-1
best = argmax_j dp[m-1][j]
backtrack theo parents[][] để lấy path (list[int] độ dài m — 1 index/sự kiện)
```

Đây chính xác là DP "strict-order" O(m·n) cho 1 video, chạy vector hoá theo
cột `j` trong vòng lặp ngoài `qi` (không so mọi cặp `(qi,j)` bằng vòng lặp
Python thuần — đó là lý do cần Numba thay vì NumPy thuần cho phần này, vì đây
là bài toán tuần tự có phụ thuộc "chạy tích luỹ" (running max) chứ không
vectorize được bằng broadcast).

**Ràng buộc `duration_limit`**: nếu > 0, giới hạn cửa sổ thời gian của cả
chuỗi (frame cuối - frame đầu <= duration_limit giây) — dùng
`np.searchsorted` trên mảng timestamp đã sort để cắt window trước khi chạy DP,
tránh chuỗi trải dài toàn bộ video dài hàng giờ.

**Đa ứng viên/video** (`_temporal_topk_dp`): sau khi tìm được 1 chuỗi tốt
nhất, "khoá" các ô đã dùng (`= -inf`) rồi chạy lại DP để tìm chuỗi tốt thứ 2,
lặp tối đa `max_sequences * 4` lần, lọc trùng lặp bằng `_overlap` (tỉ lệ giao
nhau giữa 2 tập index) và `_time_iou` (Intersection-over-Union theo thời
gian) để loại các chuỗi gần như trùng nhau (`overlap_threshold`).

**Nguồn `S` (similarity matrix)**: xây từ `multi_query_search` (tái sử dụng
hàm thuần của `semantic_search`) để lấy candidate keyframe/video cho từng
event, rồi encode lại thành ma trận điểm đầy đủ per-video trước khi chạy DP —
xem phần còn lại của `pipeline/search.py` (hàm build ma trận `S` per video,
nằm sau đoạn DP trong cùng file) và `pipeline/frame_context.py` (lấy vector
theo `video_id` từ vector cache của `ClipFaissIndex`, đúng vai trò
"pure function nhận state đã khởi tạo sẵn" theo quy tắc mục 3.2 của README).

Output: `pd.DataFrame`, mỗi dòng là 1 chuỗi khớp (1 video có thể có nhiều
dòng nếu nhiều chuỗi ứng viên), có cột `matched_sequence` (danh sách keyframe
trong chuỗi, dùng để FE vẽ dải "temporal sequence") — **ASR search sau này
tái sử dụng đúng cột `matched_sequence` này** để hiển thị giống hệt UI
temporal (xem 5.4).

### 5.3 OCR search (`retriever/ocr_search/`)

**Input:** `query: str` (raw, không split), `top_k`.
**Output nội bộ** (`pipeline/search.py`, trước khi Orchestrator enrich):
`list[OCRHit]` với `OCRHit = {score, dataset, video_id, keyframe_id, texts}`.

```
SearchPipeline.search(query, top_k):
    raw_hits = OCRRepository.search(text=query, top_k=top_k)   # ES full-text (BM25) trên field "texts"
    return [OCRHit(score=hit._score, ...source...) for hit in raw_hits]
```

**Enrich** (`Orchestrator._enrich_ocr_hits`, gọi với `oversample_factor=3`
tức lấy `top_k*3` hit thô trước khi enrich, để bù các hit bị loại):

```
max_score = max(score của mọi hit)                       # để normalize về [0,1]
for hit in hits:
    meta = index.metadata_row_for_ocr_hit(hit.video_id, hit.keyframe_id)
    if meta is None: continue      # OCR index có entry nhưng FAISS không có -> DROP, không hiện lỗi
    row = {cột METADATA_DISPLAY_COLUMNS từ meta}
    row.update(ocr_score=raw, score=normalized, retrieval_score=normalized,
               matched_texts=hit.texts, search_mode="ocr")
sort theo retrieval_score giảm dần, cắt top_k, gán display_rank/rank
```

### 5.4 ASR search (`retriever/asr_search/`)

**Input:** `query: str` (raw), `top_k`.
**Output nội bộ:** `list[ASRHit]` với `ASRHit = {score, dataset, video_id,
start_time, end_time, text}` — mỗi hit là **1 đoạn thoại** (segment), không
phải 1 keyframe.

**Điểm khác biệt quan trọng với OCR**: 1 segment thoại trải dài qua **nhiều**
keyframe. Để FE hiển thị nhất quán (tái sử dụng đúng component
`TemporalSequence.jsx` vốn render mọi row có `matched_sequence` khác rỗng),
`_enrich_asr_hits` biến mỗi segment thành **1 dòng dạng temporal-search hit**:

```
for hit in hits (dừng khi đã nhận đủ top_k segment hợp lệ):
    frames = index.metadata_rows_for_asr_hit(hit.video_id, hit.start_time, hit.end_time)
    if not frames: continue                     # segment không map được keyframe nào -> DROP
    frames = frames[:max_frames_per_hit]         # cap an toàn (mặc định 50)

    matched_sequence = [
        {cột METADATA_DISPLAY_COLUMNS từ frame, score=normalized,
         candidate_rank=i+1, sub_query=transcript_text}
        for i, frame in enumerate(frames)
    ]
    anchor = frames[len(frames)//2]              # keyframe GIỮA segment làm ảnh đại diện (cover)
    row = {cột từ anchor,
           asr_score, score=normalized, retrieval_score=normalized,
           matched_texts=[transcript_text],
           temporal_start_time, temporal_end_time, temporal_duration_sec,
           search_mode="asr", matched_sequence=matched_sequence}
```

`top_k` ở đây giới hạn **số segment**, không giới hạn tổng số frame trả về
(1 segment có thể mở rộng ra hàng chục dòng con trong `matched_sequence`).

### 5.5 Reranker (`retriever/reranker/`) — CHƯA wire vào hệ thống

`models/internvl_reranker.py` (dùng `transformers`, model InternVL — VLM đa
phương thức) + `pipeline/rerank.py`. Mục đích: sau khi có top-K candidate từ
semantic/temporal, dùng VLM để "đọc hiểu" lại ảnh + câu query rồi rerank chính
xác hơn similarity CLIP thuần. **Đã viết code nhưng chưa được gọi từ
`orchestrator.py` hay `system.py`** — xem ghi chú tường minh ngay trong
`system.py`. **Không tự ý wire khi không được yêu cầu** (rủi ro thay đổi hành
vi mặc định + tốn tài nguyên GPU đáng kể mỗi request).

---

## 6. Indexer — ghi dữ liệu vào các store

### 6.1 FAISS (`indexer/faiss/`)

- `index_builder.py`: đọc toàn bộ file `.npy` embedding
  (`collect_embedding_files`), parse tên file để lấy `dataset/video_id/keyframe_id`
  (`parse_keyframe_stem`, `parse_embedding_path`), join với `map_keyframes/*.csv`
  (`load_map_row`, `normalize_map_df`) để lấy `frame_idx/timestamp_sec/fps`,
  build ma trận `(N, dim)` + list metadata dict song song
  (`build_matrix_and_metadata`), `normalize_matrix` (L2), rồi
  `create_faiss_index` (HNSW, tham số từ `configs/indexing.yaml`) và
  `save_faiss_index` (ghi `.faiss` + `metadata.csv`).
- `vector_cache.py`: script CLI riêng để **xuất sẵn** vector gốc dạng
  `float32` memmap (`vectors_fp32.npy`) song song với index FAISS — phục vụ
  chế độ `vector_cache_mode: memmap` của `ClipFaissIndex` (đọc lại vector gốc
  nhanh mà không cần `index.reconstruct()` từng cái một, vốn chậm với HNSW).
- Điều phối: `BuildFaissIndexPipeline` (`build_pipeline.py`), chạy qua
  `scripts/embedding_extraction` → `scripts` build index (thứ tự: extract
  keyframe → extract embedding → build index → build vector cache).

### 6.2 Elasticsearch — OCR & ASR (`indexer/elasticsearch/{ocr,asr}/`)

Cấu trúc y hệt nhau cho `ocr/` và `asr/` (đúng theo quy tắc README gốc mục 4):

| File | Vai trò |
|---|---|
| `client.py` (dùng chung, ở `elasticsearch/`) | Kết nối tới cluster ES (host/port/auth từ config) |
| `schemas.py` | `OCRDocument`/`ASRDocument` (Pydantic) — định nghĩa mapping field của index |
| `repository.py` | `OCRRepository`/`ASRRepository` — CRUD + `search(text, top_k)` (BM25 full-text) |
| `factory.py` | `load_ocr_config()/load_asr_config()`, `build_elasticsearch_service()`, `build_ocr_repository()`, `build_ocr_index_pipeline()` — nơi DUY NHẤT khởi tạo kết nối ES cho subsystem này |
| `indexing_pipeline.py` | `IndexPipeline` — nhận output của `ocr_extraction`/ASR script, gọi `repository.bulk_insert(...)` |

`repository` implement `BaseIndexer` (`create_index/insert/bulk_insert/count`)
— retriever (`ocr_search`/`asr_search`) chỉ gọi `repository.search(...)`
(method đọc, không nằm trong `BaseIndexer`), không bao giờ gọi
`insert`/`bulk_insert`.

### 6.3 Milvus (`indexer/milvus/`)

`client.py` + `factory.py` + `repository.py` — backend ghi thay thế cho FAISS
(vector DB đầy đủ tính năng hơn: filter theo metadata, scale-out...). **Hiện
tại chưa có `retriever/` hay `index/` tương ứng đọc từ Milvus** — nghĩa là
nhánh này mới dừng ở "có thể ghi" nhưng runtime search vẫn 100% qua FAISS.
Nếu muốn dùng Milvus cho search thật, cần viết thêm 1 `retrieval/index/`
tương đương `ClipFaissIndex` (hoặc mở rộng nó) + retriever tương ứng, theo
đúng quy trình ở mục 12.

---

## 7. Tầng API (`main.py`)

`main.py` là **entrypoint FastAPI duy nhất** (808 dòng, single file — xem
mục 10 về việc tách route). Khởi tạo `RetrievalSystem` 1 lần (lifespan) qua
`build_system(SYSTEM_CONFIG)`, mount static files (`keyframes`, `videos`,
`map-keyframes`).

### 7.1 Danh sách endpoint

| Method | Path | Việc chính | Ghi chú |
|---|---|---|---|
| GET | `/api/health` | health check | |
| GET | `/api/config` | trả config public cho FE (search modes khả dụng, top_k mặc định...) | FE dùng để biết `available_modes` (semantic/temporal/ocr/asr/auto) — OCR/ASR sẽ KHÔNG xuất hiện ở đây nếu pipeline optional bị vô hiệu hoá lúc khởi động |
| POST | `/api/search` | endpoint search chính | body = `SearchRequest`, xem 7.2 |
| GET | `/api/frame-info` | lấy metadata 1 frame theo `video_id + keyframe_id` | dùng `find_metadata_row` |
| GET | `/api/surrounding-frames` | lấy các frame lân cận 1 frame (trước/sau `radius`) | dùng cho UI xem ngữ cảnh quanh 1 kết quả; giới hạn `ui.max_surrounding_radius` trong `app.yaml` |
| POST | `/api/similarity-search` | "more like this": tìm ảnh tương tự 1 keyframe cho trước | gọi `semantic_search.SearchPipeline.similarity_search_by_image` |
| POST | `/api/speech/transcribe` | nhận audio upload, trả text (faster-whisper) | dùng để **nhập query bằng giọng nói**, KHÔNG phải ASR search trên video; model load lazy qua `get_speech_model()` |
| POST | `/api/dres/login` | đăng nhập DRES | phục vụ nộp bài thi (VBS/AI Challenge) |
| POST | `/api/dres/submit` | nộp 1 kết quả (video_id + frame_id + timestamp) lên DRES | |

### 7.2 `SearchRequest` (Pydantic schema)

```python
class SearchRequest(BaseModel):
    query: str
    top_k: int | None = None
    candidate_multiplier: int | None = None
    use_split: bool | None = None
    use_translate: bool | None = None
    search_mode: SearchMode | None = "semantic"   # "semantic"|"temporal"|"ocr"|"asr"|"auto"
    duration_limit: float | None = -1
```

Luồng xử lý `/api/search` (mã giả, `search_api`):

```
payload -> áp dụng default từ configs/app.yaml (top_k, candidate_multiplier...)
query = translate_query_if_needed(payload.query, payload.use_translate)
        # nếu use_translate=True: gọi translator (Libre/Hy-MT2) dịch VI->EN
        # trước khi encode CLIP (model CLIP train chủ yếu trên tiếng Anh)
df, plan = retrieval_system.<search_mode tương ứng>(query, top_k, ...)
          # ví dụ search_mode="temporal" -> retrieval_system.search_temporal(...)
results = [dict_to_result_FAST(row) for row in df.to_dict("records")]
          # serialize từng field: format keyframe_id, resolve keyframe_path
          # thành URL tĩnh, resolve video_path, serialize matched_sequence
          # (đệ quy dùng lại chính hàm này cho từng frame con), json_safe (NaN -> null)
return {results, query_plan: plan, ...}
```

Điểm quan trọng: `main.py` **không tự viết logic search** — nó chỉ gọi 1
trong các hàm `search_*` phẳng của `RetrievalSystem` (xem mục 4), rồi lo phần
"trình bày" (serialize DataFrame → JSON, resolve path tương đối → URL tĩnh
FastAPI phục vụ qua `StaticFiles`).

### 7.3 Các hàm tiện ích serialize đáng chú ý

- `dict_to_result_FAST`: hàm hot-path — chạy trên MỌI row của MỌI response
  search, nên được viết tối ưu (tên có hậu tố `_FAST` là chủ đích). Bất kỳ
  thay đổi nào ở đây cần đo lại latency trước khi merge.
- `serialize_matched_sequence`: đệ quy lại `dict_to_result_FAST` cho từng
  frame trong `matched_sequence` — đây là điểm chung khiến cả temporal lẫn
  ASR search dùng lại được đúng 1 UI component ở FE.
- `safe_float/safe_int/json_safe`: chống lỗi serialize khi DataFrame có
  `NaN`/`None`/numpy scalar (JSON chuẩn không có NaN).
- `clean_external_url`: chuẩn hoá URL DRES.
- `find_metadata_row`: tra cứu trực tiếp vào `metadata.csv` (qua `ClipFaissIndex`)
  theo `(video_id, keyframe_id)` — dùng cho `/api/frame-info` và
  `/api/surrounding-frames`, độc lập với pha search.

### 7.4 `src/api/` — scaffold rỗng

`routers/`, `schemas/`, `utils/` hiện chỉ có `__init__.py` trống. Đây là nơi
dự kiến sẽ chuyển route/schema/util ra khỏi `main.py` trong tương lai (xem
đề xuất refactor ở mục 10). Cấu trúc thư mục đã được chuẩn bị sẵn theo đúng
convention FastAPI phổ biến (routers theo domain, schemas Pydantic tách
riêng, utils dùng chung).

---

## 8. Translation subsystem (`src/translation/`)

Vì OpenCLIP text encoder hoạt động tốt nhất với tiếng Anh, mọi câu query
tiếng Việt cần dịch trước khi encode. `BaseTranslator` (ABC) + 2 hiện thực:

| Translator | Cơ chế | Khi nên dùng |
|---|---|---|
| `LibreTranslator` | gọi HTTP tới 1 LibreTranslate server (`base_url`, `api_key`, `timeout` từ `configs/app.yaml -> translate.libre`) | có sẵn server LibreTranslate (self-host hoặc public); độ trễ phụ thuộc network |
| `HyMT2Translator` | load model GGUF `Hy-MT2-1.8B-2Bit` local qua `llama_cpp` (`n_gpu_layers` cấu hình được, 0 = CPU) | muốn chạy hoàn toàn offline / không phụ thuộc network, đánh đổi RAM/VRAM và thời gian load model |

Chọn qua `translate_agent: libre|hymt2` trong `app.yaml`, dựng bằng
`get_translator(cfg, backend_dir)` (`factory.py`). `main.py` gọi
`translate_query_if_needed(query, use_translate)` — nếu FE gửi
`use_translate=false` hoặc `translate.enabled_default=false`, bỏ qua bước
dịch (query được dùng nguyên văn — hữu ích khi query vốn đã là tiếng Anh
hoặc muốn test CLIP trực tiếp).

---

## 9. Cấu hình (`configs/`)

| File | Subsystem | Trường quan trọng |
|---|---|---|
| `app.yaml` | Runtime chính (main.py + system.py) | `faiss.*` (path, `ef_search`, cache mode), `model.*` (phải khớp model lúc train embedding), `search.*` (`default_top_k`, `max_top_k`, `candidate_multiplier`), `ui.*` (surrounding radius), `translate_agent` + `translate.*`, `speech.*` (Whisper cho `/api/speech/transcribe`), `debug.profile` |
| `ocr_extraction.yaml` | offline OCR pipeline | tham số PaddleOCR/VietOCR |
| `asr_extraction.yaml` | offline ASR pipeline | tham số faster-whisper cho batch transcribe |
| `embeddings.yaml` | offline embedding pipeline | model CLIP, batch size, device |
| `indexing.yaml` | build FAISS index | tham số HNSW (M, efConstruction...), đường dẫn output |
| `kf_extraction.yaml` | offline keyframe pipeline | threshold TransNetV2, `max_gap`, `histogram threshold` |

> **Không tìm thấy `configs/ocr.yaml`/`configs/asr.yaml`** (dùng bởi
> `load_ocr_config`/`load_asr_config` cho phần **search-time Elasticsearch
> connection**, khác với `ocr_extraction.yaml`/`asr_extraction.yaml` là
> config cho pha **offline extraction**) trong bản zip được cung cấp. Đây là
> file cấu hình kết nối ES (host, index name, port...) cần thiết để
> `build_system()` khởi tạo `ocr_search_pipeline`/`asr_search_pipeline` —
> **nếu thiếu, OCR/ASR search sẽ tự tắt (log warning) chứ không crash**, theo
> đúng thiết kế "optional subsystem" ở mục 4. Team kế nhiệm cần xác nhận file
> này có tồn tại trong môi trường triển khai thật hay đã bị .gitignore.

---

## 10. Vấn đề đã biết / nợ kỹ thuật

Được tổng hợp từ ghi chú tường minh (comment) trong chính source code —
không phải suy đoán — cộng với quan sát cấu trúc thư mục:

1. **OCR/ASR chưa trả về `list[Hit]` chuẩn.** `BaseRetriever.search()` quy
   định trả `list[Hit]`, nhưng `ocr_search`/`asr_search` trả `list[OCRHit]`/
   `list[ASRHit]` (TypedDict riêng), buộc `Orchestrator` phải biết chi tiết
   nội bộ của từng loại (`_enrich_ocr_hits`, `_enrich_asr_hits`) thay vì xử
   lý đồng nhất qua interface chung. **Không nhân bản thêm pattern này khi
   thêm block mới** — nếu thêm 1 loại search mới, ưu tiên trả `list[Hit]`
   ngay từ đầu.
2. **Reranker chưa được wire.** Code đã có (`retriever/reranker/`) nhưng
   không nằm trong luồng `run_search`. Muốn bật cần: (a) quyết định bật cho
   mode nào (chỉ semantic? cả temporal?), (b) đo chi phí GPU/latency, (c)
   thêm tham số bật/tắt qua config thay vì hard-code luôn bật.
3. **Milvus indexer mồ côi.** Có `indexer/milvus/` nhưng không có
   `retriever`/`index` nào đọc từ đó — hoặc là nhánh thử nghiệm dang dở, hoặc
   dự định thay thế FAISS trong tương lai. Cần làm rõ ý định trước khi
   xoá/tiếp tục.
4. **`main.py` là 1 file 808 dòng ôm toàn bộ route + schema + serialize
   helper.** `src/api/{routers,schemas,utils}` đã có sẵn thư mục trống — nên
   tách dần theo domain (search, dres, frame-info, speech) khi có thời gian,
   **nhưng không phải ưu tiên khẩn** vì `main.py` vẫn tuân thủ đúng ranh giới
   quan trọng nhất (chỉ import qua `system.py`).
5. **Thiếu `requirements.txt`/lockfile** trong bản giao — xem mục 1.1.
6. **`configs/ocr.yaml`/`configs/asr.yaml` (config kết nối ES lúc search)
   không có trong bản zip** — xem mục 9. Cần xác nhận với hạ tầng
   triển khai thật.
7. **README con lỗi thời**: `src/README.md`, `src/utils/README.md` mô tả tên
   thư mục cũ (`embeddings/`, `keyframes/`, `logic/`) — đã ghi chú trong
   README gốc, nhắc lại ở đây để tránh nhầm khi đọc các file đó.
8. **Không có test suite tự động** được tìm thấy trong bản zip (không có
   thư mục `tests/`). Việc thêm test là ưu tiên cao khi mở rộng thêm search
   mode, vì `Orchestrator`/`aggregate_multi_query`/DP temporal đều là logic
   tính toán thuần rất dễ unit-test (input/output rõ ràng, không cần mock
   hạ tầng) nhưng hiện chưa có test nào bảo vệ.

---

## 11. Cách giữ tài liệu này không bị stale

Tài liệu này được viết bằng cách đọc trực tiếp toàn bộ `src/` và `main.py`
tại 1 thời điểm — nó **sẽ lệch dần theo thời gian** giống cách README con cũ
đã lệch. Quy ước cho team tương lai:

- Mọi PR thêm 1 search block mới / đổi contract của `Orchestrator`/`system.py`
  **phải** cập nhật tương ứng mục 4, 5, 12 của file này trong cùng PR.
- Mọi thay đổi schema `SearchRequest`/endpoint mới trong `main.py` phải cập
  nhật bảng ở mục 7.
- README.md (root) chỉ nên chứa bản tóm tắt + link sang file này — không
  lặp lại chi tiết thuật toán, để tránh 2 nguồn dễ lệch nhau.
- Nếu phát hiện mục nào trong tài liệu **đã sai** so với code hiện tại, ưu
  tiên sửa tài liệu ngay khi phát hiện (comment `# xem ARCHITECTURE.md mục
  X — cần cập nhật` là đủ nếu chưa có thời gian viết lại ngay).

---

## 12. Hướng dẫn thêm 1 search block mới (chi tiết mã giả)

Áp dụng nguyên văn quy tắc mục 3 của README gốc, kèm mã giả cụ thể hơn cho
từng bước để một dev mới có thể làm theo không cần hỏi lại.

### Bước 1 — Tạo thư mục block

```
src/retrieval/retriever/<ten>_search/
├── __init__.py
├── factory.py
└── pipeline/
    ├── __init__.py
    └── search.py
```

### Bước 2 — `pipeline/search.py`

```python
# 1) Các hàm thuần triển khai thuật toán — KHÔNG khởi tạo gì, chỉ nhận state
def score_candidates(state, query_embedding, top_k) -> pd.DataFrame:
    ...  # thuật toán thật

# 2) Đúng 1 class entrypoint, cùng hình dạng semantic/ocr/asr
class SearchPipeline:
    def __init__(self, dependency) -> None:
        self.dependency = dependency  # index / repository đã build sẵn

    def search(self, query, top_k=10, **kwargs) -> pd.DataFrame:  # hoặc list[Hit]
        prepared = self.dependency.encode_or_fetch(query)   # bước (1): chuẩn bị input
        return score_candidates(self.dependency, prepared, top_k)  # bước (2): gọi hàm thuần
```

### Bước 3 — `factory.py`

```python
def build_<ten>_search_pipeline(cfg=None, config_path="configs/<ten>.yaml") -> SearchPipeline:
    cfg = cfg or load_yaml(config_path)
    dependency = init_dependency_from_cfg(cfg)   # NƠI DUY NHẤT mở connection/load model/đọc file
    return SearchPipeline(dependency)
```

### Bước 4 — Đăng ký vào `orchestrator.py`

```python
SearchMode = Literal["semantic", "temporal", "ocr", "asr", "<ten>", "auto"]

class Orchestrator:
    def __init__(self, ..., <ten>_search_pipeline=None):
        self.<ten>_search_pipeline = <ten>_search_pipeline

    def run_search(self, query, mode, ...):
        ...
        if effective_mode == "<ten>":
            return self.<ten>_search_pipeline.search(plan.query_or_events, top_k), plan
```

Nếu output shape khác `pd.DataFrame` chuẩn (giống OCR/ASR), viết thêm
`_enrich_<ten>_hits(...)` theo đúng mẫu `_enrich_ocr_hits`/`_enrich_asr_hits`
— nhưng **cân nhắc trả `list[Hit]` chuẩn ngay từ đầu** để không lặp lại nợ kỹ
thuật ở mục 10.1.

### Bước 5 — `system.py`

```python
def _build_<ten>_pipeline_or_none(cfg):
    <ten>_cfg = cfg.get("<ten>")
    if not <ten>_cfg:
        return None
    try:
        return build_<ten>_search_pipeline(config_path=<ten>_cfg.get("config_path", "configs/<ten>.yaml"))
    except Exception as exc:
        print(f"[system] <ten> pipeline init failed, disabling: {exc}")
        return None

# trong build_system():
<ten>_pipeline = _build_<ten>_pipeline_or_none(config)
orchestrator = Orchestrator(..., <ten>_search_pipeline=<ten>_pipeline)

# thêm hàm phẳng trong RetrievalSystem:
def search_<ten>(self, query, top_k=10) -> tuple[pd.DataFrame, QueryPlan]:
    return self._orchestrator.run_search(query, mode="<ten>", top_k=top_k)
```

### Bước 6 — `configs/<ten>.yaml` + `main.py`

Thêm file config nếu cần. Trong `main.py`: mở rộng `SearchMode`/`available_modes`
ở `/api/config`; **không** viết logic search trong `main.py` — điểm gọi vẫn
luôn là `retrieval_system.search_<ten>(...)`.

### Checklist (đồng bộ với README gốc mục 3.4)

- [ ] `factory.py` + `pipeline/search.py` đúng khuôn
- [ ] Hàm thuật toán thuần tách khỏi `SearchPipeline`
- [ ] `SearchPipeline.search(...)` là entrypoint duy nhất
- [ ] Output là `list[Hit]` chuẩn nếu có thể (ưu tiên hơn DataFrame/TypedDict riêng)
- [ ] Không viết logic ghi/insert trong `retriever/`
- [ ] Đăng ký `SearchMode`, dispatch, enrich (nếu cần) trong `orchestrator.py`
- [ ] Hàm phẳng `search_<ten>` trong `system.py`
- [ ] `main.py` chỉ gọi qua `RetrievalSystem`, không import thẳng block
- [ ] Config riêng ở `configs/<ten>.yaml`, load qua factory, subsystem optional (try/except khi build)
- [ ] Cập nhật `available_modes` trong `/api/config`
- [ ] **Cập nhật mục 4/5/12 của `ARCHITECTURE.md` này trong cùng PR**
