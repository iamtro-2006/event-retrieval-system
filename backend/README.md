# Backend — Event Retrieval System

## Chạy keyframe extraction

Pipeline hỗ trợ hai strategy trong `configs/kf_extraction.yaml`:

```yaml
keyframe:
  strategy: p3  # hoặc legacy_lmske
```

- `p3`: TransNetV2 → adaptive candidate sampling → quality filter →
  `MobileCLIP2-S4/dfndr2b` → K-medoids/AUCC → common/unique/global MMR → dedup.
- `legacy_lmske`: giữ pipeline LMSKE cũ và encoder
  `ViT-L-16-SigLIP-256/webli`.

Input mặc định nằm tại `../data/raw/videos`, output được ghi trực tiếp vào
`../data/processed`. Weight TransNetV2 phải có tại
`weights/transnetv2-pytorch-weights.pth`; OpenCLIP sẽ tải/cache model trong lần
chạy đầu nếu máy chưa có.

```powershell
cd backend
python scripts\keyframe_extraction\run.py --config configs\kf_extraction.yaml
```

Chạy thử một nhóm video mà không cần sao chép dữ liệu hoặc đổi config:

```powershell
python scripts\keyframe_extraction\run.py --config configs\kf_extraction.yaml --video-ids L21_V001 L21_V002 L22_V001 L22_V002
```

Output tương thích với embedding/OCR/retrieval:

```text
data/processed/
├── keyframes/<group>/<video_id>/000000.jpg
├── map_keyframes/<group>/<video_id>.csv
└── diagnostics/<group>/<video_id>/
    ├── candidates.csv
    ├── shots.csv
    ├── selected_keyframes.csv
    ├── dedup_dropped.csv
    ├── metrics.json
    └── run_config.json
```

Chạy unit test:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

Backend FastAPI cho hệ thống truy vấn/retrieval keyframe video theo nội dung
(semantic/CLIP), theo trình tự thời gian (temporal), theo chữ viết trên màn
hình (OCR) và theo lời thoại (ASR).

> 📘 **Tài liệu này là bản tóm tắt.** Bản đầy đủ — tech stack chi tiết, sơ đồ
> luồng dữ liệu end-to-end, mã giả thuật toán cho từng pipeline (keyframe
> extraction, embedding, temporal DP, OCR/ASR enrich...), danh sách đầy đủ
> endpoint, nợ kỹ thuật đã biết, và hướng dẫn handover cho team kế tiếp —
> nằm ở [`ARCHITECTURE.md`](./ARCHITECTURE.md). Đọc file này để có bức tranh
> tổng quan, rồi sang `ARCHITECTURE.md` khi cần đi sâu vào 1 phần cụ thể
> hoặc trước khi thêm 1 search block mới.

---

## 1. Cây thư mục thực tế

```text
backend/
├── main.py                     # FastAPI app — entrypoint DUY NHẤT của API
├── mock_api.py                 # server mock để FE dev không cần backend thật
├── requirements.txt             # ⚠ không có trong bản zip hiện tại — xem ARCHITECTURE.md mục 1.1
├── configs/                    # 1 file YAML config / 1 subsystem
│   ├── app.yaml                 # config chính: model, faiss, search, ui, paths...
│   ├── ocr.yaml                 # config Elasticsearch cho OCR
│   ├── asr.yaml                 # config Elasticsearch cho ASR
│   ├── embeddings.yaml          # config pipeline trích embedding
│   ├── indexing.yaml            # config build index
│   └── kf_extraction.yaml       # config trích keyframe
│
├── scripts/                    # CLI/offline: gọi vào src/, không chứa logic
│   ├── keyframe_extraction/run.py
│   ├── embedding_extraction/run.py
│   ├── ocr/{index,search}/run.py
│   ├── asr/{index,search}/run.py
│   └── retrieval/{run.py, query.py}
│
└── src/                         # toàn bộ business logic
    ├── keyframe_extraction/     # video -> scene detect -> chọn keyframe
    │   ├── models/               # detector, clustering, selector (thuật toán thuần)
    │   └── pipeline/             # extract_keyframes.py — điều phối models/
    │
    ├── embedding_extraction/    # keyframe -> embedding (OpenCLIP/SigLIP...)
    │   ├── models/               # embedder, encoder, frame_loader
    │   └── pipeline/             # extract_embeddings.py — điều phối models/
    │
    ├── ocr_extraction/          # keyframe -> text (PaddleOCR detect + VietOCR nhận dạng)
    │   ├── models/               # dataset, engine (PaddleOCR), vietocr_engine
    │   └── pipeline/             # extract_ocr.py — điều phối models/
    │
    ├── translation/              # dịch query VI->EN trước khi encode CLIP
    │   ├── base_translator.py    # interface BaseTranslator
    │   ├── libre_translator.py   # backend: LibreTranslate server (HTTP)
    │   ├── hy_mt2_translator.py  # backend: model GGUF local (llama_cpp)
    │   └── factory.py            # get_translator(cfg) chọn theo `translate_agent`
    │
    ├── retrieval/                # lõi retrieval — xem chi tiết mục 3
    │   ├── system.py             # FACADE DUY NHẤT giữa api/ và retriever/*
    │   ├── base/                 # BaseRetriever, BaseIndexer (interface chung)
    │   ├── index/                # ClipFaissIndex (FAISS + model CLIP) + factory
    │   ├── indexer/               # các backend GHI index: faiss/ elasticsearch/ milvus/
    │   └── retriever/             # các backend ĐỌC/search: mỗi loại search 1 block
    │       ├── semantic_search/
    │       ├── temporal_search/
    │       ├── ocr_search/
    │       ├── asr_search/
    │       ├── reranker/
    │       └── common/            # orchestrator.py, query_parser.py, scoring.py
    │
    ├── api/                      # SCAFFOLD, hiện chưa dùng — xem mục 4
    │   ├── routers/  schemas/  utils/
    │
    └── utils/                    # tiện ích dùng chung: config, logger, device, seed, video_io
```

> Ghi chú: một vài README con trong repo (`src/README.md`, `src/utils/README.md`)
> mô tả tên thư mục cũ (`embeddings/`, `keyframes/`, `logic/`) — tên thật hiện tại
> là `embedding_extraction/`, `keyframe_extraction/`, và phần "logic" nằm trong
> `retrieval/retriever/common/`. File này (README gốc ở root backend) và
> `ARCHITECTURE.md` mô tả đúng cấu trúc hiện hành, nên ưu tiên đọc 2 file này
> trước. `src/ocr_extraction/` và `src/translation/` là 2 module tồn tại thật
> trong code nhưng chưa được README con nào nhắc tới trước đây — chi tiết ở
> `ARCHITECTURE.md` mục 3.3 và mục 8.

---

## 2. Luồng chạy tổng quát

```text
video
 └─ keyframe_extraction  → chọn keyframe đại diện cho từng scene
     └─ embedding_extraction → encode keyframe thành vector (CLIP/SigLIP)
         └─ retrieval.indexer.*  → ghi vector/metadata vào FAISS / Elasticsearch / Milvus
             └─ retrieval.retriever.* → nhận query, search trên index đã build
                 └─ retrieval.system.RetrievalSystem → 1 hàm search_* / mode
                     └─ main.py (FastAPI) → serialize kết quả → trả JSON cho FE
```

`scripts/` là lớp CLI mỏng gọi vào `src/*` để chạy offline (build index, test
search…) — không có business logic trong `scripts/`.

---

## 3. Quy tắc chung khi tạo một "block" trong `src/retrieval/retriever/`

Đây là phần quan trọng nhất khi mở rộng backend: mỗi loại search
(`semantic_search`, `temporal_search`, `ocr_search`, `asr_search`) được viết
theo **đúng 1 khuôn mẫu** để `Orchestrator` gọi thống nhất. Khi thêm một loại
search mới, hãy tuân thủ chính xác các quy tắc sau.

### 3.1 Cấu trúc thư mục bắt buộc của 1 block search

```text
retriever/<ten>_search/
├── __init__.py
├── factory.py                 # build_<ten>_search_pipeline(...) -> SearchPipeline
└── pipeline/
    ├── __init__.py
    └── search.py               # các hàm thuật toán thuần + class SearchPipeline
```

### 3.2 Nguyên tắc tách trách nhiệm (rất quan trọng)

- **`factory.py`**: là nơi DUY NHẤT được phép khởi tạo dependency (mở kết nối
  Elasticsearch, load `ClipFaissIndex`, đọc config YAML…) rồi trả về một
  `SearchPipeline` đã sẵn sàng dùng. Mọi block factory đều có cùng chữ ký dạng
  `build_<ten>_search_pipeline(...) -> SearchPipeline`.
- **`pipeline/search.py`**:
  - Chứa các **hàm thuần** (pure function) triển khai thuật toán search —
    không được tự khởi tạo index/model/kết nối, không tự gọi `encode(...)`
    của model. Các hàm này chỉ nhận state đã được khởi tạo sẵn (index, lock,
    metadata, embedding đã encode…) làm tham số và trả về `pd.DataFrame`.
  - Có đúng **1 class `SearchPipeline`** ở cuối file, đóng vai trò
    **entrypoint duy nhất** của cả block. Class này chỉ điều phối 2 bước:
    (1) gọi model để encode/chuẩn bị input, (2) gọi các hàm thuật toán thuần
    ở trên — không chứa thuật toán bên trong class.
  - Bắt buộc expose method `.search(...)` để `Orchestrator` gọi đồng nhất
    (`self.semantic_search.search(...)`, `self.ocr_search_pipeline.search(...)`…).
- **Không** có class `SemanticSearch`/`OCRSearch`… riêng ngoài `SearchPipeline`
  — nếu thấy cấu trúc cũ dạng `models/xxx_search.py` thì đó là pattern đã bị
  loại bỏ, không tái tạo lại.
- **Đọc vs Ghi tách biệt hoàn toàn**: block trong `retriever/` chỉ ĐỌC. Việc
  ghi/ingest dữ liệu vào index nằm ở `src/retrieval/indexer/<backend>/` và kế
  thừa `BaseIndexer` (`create_index`, `insert`, `bulk_insert`, `count`).
  Không được viết logic ghi index bên trong `retriever/`.
- **Interface chung**: nếu có thể, retriever nên tương thích với
  `BaseRetriever.search(query, top_k) -> list[Hit]` (`src/retrieval/base/base_retriever.py`).
  Hiện tại `ocr_search`/`asr_search` trả về DataFrame riêng (chưa chuẩn hoá về
  `Hit`) — đây là nợ kỹ thuật đã được ghi chú rõ trong `orchestrator.py`,
  không nên nhân bản thêm khi thêm block mới; ưu tiên chuẩn hoá dần về `Hit`.

### 3.3 Điểm nối vào hệ thống — theo đúng thứ tự

1. **`src/retrieval/retriever/<ten>_search/`**: viết block theo khuôn 3.1–3.2.
2. **`src/retrieval/retriever/common/orchestrator.py`**: thêm nhánh cho
   `SearchMode` mới (mở rộng `Literal[...]`), inject pipeline mới vào
   `Orchestrator.__init__`, xử lý enrich kết quả nếu shape trả về khác
   `pd.DataFrame` chuẩn (xem cách `_enrich_ocr_hits`/`_enrich_asr_hits` đang
   làm).
3. **`src/retrieval/system.py`**: đây là **facade DUY NHẤT** giữa `api/` (tức
   `main.py`) và `retriever/*`. Thêm 1 hàm `search_<ten>(...)` phẳng trong
   `RetrievalSystem`, gọi `self._orchestrator.run_search(mode="<ten>", ...)`.
   **`main.py` không được import trực tiếp từ `retriever/<ten>_search/`** —
   chỉ được import từ `src.retrieval.system`.
4. **`configs/<ten>.yaml`** (nếu block cần config riêng, giống `ocr.yaml`/
   `asr.yaml`): thêm file config mới, build bằng hàm `load_<ten>_config`
   tương tự `load_ocr_config`.
5. **`main.py`**: nếu cần expose ra API, mở rộng `SearchRequest.search_mode`
   và danh sách `available_modes` trong `/api/config`; điểm gọi search vẫn
   luôn là `orchestrator.run_search(...)` — không tự viết logic search trong
   `main.py`.
6. Nếu search mới cần một backend indexing riêng (vd. một loại database mới):
   viết thêm ở `src/retrieval/indexer/<backend>/`, kế thừa `BaseIndexer`,
   theo đúng convention của `indexer/elasticsearch/` (có `client.py`,
   `factory.py`, `repository.py`, `schemas.py`, `indexing_pipeline.py`) hoặc
   `indexer/faiss/` (`index_builder.py`, `build_pipeline.py`, `vector_cache.py`).

### 3.4 Checklist khi thêm 1 search block mới

- [ ] Thư mục đúng khuôn: `factory.py` + `pipeline/search.py` (+ `pipeline/__init__.py`)
- [ ] `pipeline/search.py`: hàm thuật toán thuần tách khỏi `SearchPipeline`
- [ ] `SearchPipeline.search(...)` là entrypoint duy nhất, không có class thuật toán rời
- [ ] `factory.build_<ten>_search_pipeline(...)` là nơi duy nhất khởi tạo dependency
- [ ] Không viết logic ghi/insert index trong `retriever/`
- [ ] Đăng ký mode mới trong `orchestrator.py` (`SearchMode`, dispatch, enrich nếu cần)
- [ ] Thêm hàm phẳng `search_<ten>` trong `retrieval/system.py`
- [ ] `main.py` chỉ gọi qua `orchestrator`/`RetrievalSystem`, không import thẳng block
- [ ] Config riêng (nếu có) đặt ở `configs/<ten>.yaml`, load qua factory
- [ ] Cập nhật `available_modes` trong `/api/config` nếu expose ra FE

---

## 4. Các phần khác

- **`src/retrieval/indexer/`**: nơi ghi index, tách theo backend
  (`elasticsearch/`, `faiss/`, `milvus/`), mỗi backend implement `BaseIndexer`.
  `elasticsearch/asr/` và `elasticsearch/ocr/` là 2 ví dụ song song, cùng có
  `client.py` (kết nối), `repository.py` (CRUD), `schemas.py` (mapping index),
  `factory.py` (build repository từ config), `indexing_pipeline.py` (ingest).
- **`src/retrieval/retriever/reranker/`**: bước rerank tuỳ chọn
  (`models/internvl_reranker.py` + `pipeline/rerank.py`), **chưa** được wire
  vào `system.py`/`orchestrator.py` (ghi chú rõ trong `system.py`) — không tự
  ý wire khi không được yêu cầu, tránh phá vỡ hành vi mặc định.
- **`src/api/`**: hiện là **scaffold rỗng** (`routers/`, `schemas/`, `utils/`
  chỉ có `__init__.py` trống). Toàn bộ route thật sự đang định nghĩa trực tiếp
  trong `main.py` ở root. Nếu tách route ra khỏi `main.py` trong tương lai,
  đây là nơi nên chuyển vào, theo đúng tên thư mục đã có sẵn.
- **`main.py`**: entrypoint FastAPI duy nhất — khởi tạo `RetrievalSystem` qua
  `build_system(SYSTEM_CONFIG)`, mount static files (`keyframes`, `videos`,
  `map-keyframes`), định nghĩa các endpoint `/api/search`, `/api/health`,
  `/api/config`, `/api/frame-info`, `/api/surrounding-frames`,
  `/api/similarity-search`, `/api/speech/transcribe`, và tích hợp nộp bài
  DRES (`/api/dres/login`, `/api/dres/submit`). Toàn bộ logic serialize
  DataFrame -> JSON response (`dict_to_result_FAST`, `serialize_matched_sequence`…)
  nằm ở đây, không nằm trong `src/`.
- **`src/ocr_extraction/`**: pipeline offline trích chữ trên khung hình —
  PaddleOCR để phát hiện vùng chữ (`models/engine.py`), VietOCR để nhận dạng
  ký tự tiếng Việt trên từng vùng đã crop (`models/vietocr_engine.py`), điều
  phối bởi `ExtractOCRPipeline`. Output được `indexer/elasticsearch/ocr`
  bulk-index — xem `ARCHITECTURE.md` mục 3.3.
- **`src/translation/`**: dịch câu query tiếng Việt sang tiếng Anh trước khi
  encode CLIP (model CLIP train chủ yếu trên tiếng Anh), 2 backend hoán đổi
  được qua config (`LibreTranslate` server hoặc model GGUF local `Hy-MT2`
  qua `llama_cpp`) — xem `ARCHITECTURE.md` mục 8.
- **`src/utils/`**: các tiện ích nền tảng dùng chung
  (`config.py`, `logger.py`, `device.py`, `seed.py`, `video_io.py`) — module
  ổn định, các module khác phụ thuộc vào nó nên hạn chế thay đổi breaking.

---

## 5. Nguyên tắc chung cho toàn backend

- Mỗi module trong `src/` là 1 thành phần độc lập: input/output/config/dependency rõ ràng.
- Không module nào phụ thuộc vào cấu trúc dữ liệu đặc thù của frontend.
- Tách bạch tuyệt đối: `indexer/` = ghi, `retriever/` = đọc.
- `system.py` là ranh giới duy nhất giữa `api` (main.py) và toàn bộ `retrieval/`.
- Mọi thuật toán nên là hàm thuần, class chỉ đóng vai trò điều phối/entrypoint.
- Config tách theo file YAML riêng cho từng subsystem trong `configs/`.

---

## 6. Đọc tiếp

Đây là bản tóm tắt cấu trúc + quy tắc. Để hiểu **vì sao** hệ thống được thiết
kế như vậy, **thuật toán chạy cụ thể ra sao** (temporal DP, aggregate đa
query, enrich OCR/ASR...), danh sách đầy đủ endpoint/schema, và những vấn đề
đã biết cần lưu ý khi tiếp quản dự án, xem **[`ARCHITECTURE.md`](./ARCHITECTURE.md)**.
