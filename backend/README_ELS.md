# README_ELS — Milvus + Elasticsearch Retrieval Infrastructure

This document describes the two datastore backends that power the retrieval
system, plus the fusion layer that combines them. It is the source of truth
for the **current** (post-Milvus, post-fusion) architecture and supersedes the
FAISS-centric description in `ARCHITECTURE.md`.

All paths are relative to `backend/`.

---

## 1. Full-system diagram

```
                       RAW ARTEFACTS (offline)
┌─────────────────────────────────┬───────────────────────────────────────────┐
│   Keyframe .npy vectors         │   OCR JSONs                 ASR JSONs      │
│   + metadata.csv                │   Qwen3-VL VLM (default)    faster-whisper │
│                                 │   or PaddleOCR + VietOCR   (external*)     │
└────────────────┬────────────────┴──────────────┬────────────────┬───────────┘
                 │                               │                │
                 ▼                               └───────┬────────┘
        [ Milvus Indexer ]                                ▼
        BuildMilvusIndexPipeline              [ Text Ingestion Pipeline ]
      ┌────────────┬───────────────┐            multimodal_text.IndexPipeline
      ▼            ▼               ▼                      │
  Milvus DB   vectors_fp32.npy  metadata.csv               ▼
  keyframes     (memmap cache)  (pandas)          Elasticsearch
  (HNSW ANN)       ▲                              ├─ multimodal_text (unified)
                   │                              ├─ ocr_db          (legacy)
                   │                              └─ asr_db          (legacy)
                   │ (temporal DP reads memmap)
═══════════════════╪═══════════════════════════════════════════════════════════
                   │                   RUNTIME / API
                   ▼
              User Query
                   │
        ┌──────────┴───────────┐
        ▼ (translated, vi→en)  ▼ (raw query)
   [ Semantic Arm ]        [ Text Arm ]
     ClipMilvusIndex          text_search
       (Milvus)            (ES multi_match)
        │                        │
        └───────────┬────────────┘
                    ▼
            [ Fusion Layer ]  retriever/common/fusion.py
              - min-max score normalization (per-mode, over candidate pool)
              - join on (video_id, keyframe_id_int)
              - weighted sum (semantic 0.6 / text 0.4) + RRF consensus bonus
                    │
                    ▼
            Ranked keyframes  (same result shape as every other mode)
```

**Other runtime modes** (not in the fusion path above):

| Mode | Backend | Notes |
|---|---|---|
| `semantic` | Milvus (CLIP) | single-mode CLIP text→image |
| `temporal` | memmap `.npy` (DP) | multi-event sequence, reads `vectors_fp32.npy` |
| `ocr` | ES `ocr_db` | standalone on-screen text |
| `asr` | ES `asr_db` | standalone speech segment (returns `matched_sequence`) |
| `text` | ES `multimodal_text` | unified OCR+ASR multi_match |
| `fusion` | Milvus + ES | this document's focus |
| `auto` | — | picks `temporal` (multi-event) or `semantic` |

\* The bulk per-video ASR extraction (`extract_asr`) that produces
`data/processed/asr/*.json` is **external to this repo** — the backend only
indexes/searches those JSONs. `faster_whisper` is imported in `main.py` for the
`/api/transcribe` endpoint (voice query), not for bulk transcript extraction.

---

## 2. Milvus — vector / ANN search (replaces FAISS)

### 2.1 Three-pillar design

`ClipMilvusIndex` keeps the same *interface* as the old `ClipFaissIndex` but
splits the storage responsibilities into three pillars:

| Pillar | Storage | Purpose |
|---|---|---|
| 1. ANN search | **Milvus** (`Collection`) | fast, scalable semantic vector search |
| 2. Vector cache | **`vectors_fp32.npy`** (`np.memmap`) | temporal-search DP matrix (RAM-backed, no network round-trips) |
| 3. Metadata | **`metadata.csv` → pandas** | O(1) OCR/ASR/text hit enrichment (no per-hit Milvus query) |

### 2.2 Key files

- `src/retrieval/index/clip_milvus_index.py` — `ClipMilvusIndex` + `MilvusSearchAdapter`
- `src/retrieval/indexer/milvus/build_pipeline.py` — `BuildMilvusIndexPipeline` (ingestion)
- `src/retrieval/index/factory.py` — `build_clip_milvus_index(...)`, `build_clip_index(config, backend)`
- `src/retrieval/index/constants.py` — shared `METADATA_DISPLAY_COLUMNS`, `resolve_device`

### 2.3 Collection schema

Milvus holds **only ANN data** (filtering/metadata resolution stays in pandas):

| Field | Type | Role |
|---|---|---|
| `row_id` | `INT64` (primary key, `auto_id=False`) | positional row index == memmap row == `metadata.csv` row |
| `embedding` | `FLOAT_VECTOR` (dim = CLIP dim) | L2-normalized CLIP embedding |

Index: **HNSW**, metric **IP** (inner product on normalized vectors == cosine).
The positional `row_id` PK means `MilvusSearchAdapter.search()` returns FAISS-
shaped positional indices directly — no secondary id→row mapping, and no
`keyframe_id` uniqueness pitfall.

### 2.4 Adapter contract

`MilvusSearchAdapter` exposes the FAISS-compatible surface the semantic/
temporal pipelines already call:

```python
index.search(embeddings, k) -> (scores, indices)   # float32 (n,k), int64 (n,k)
index.ntotal                                        # = collection.num_entities
index.d                                             # = embedding dim
```

`ClipMilvusIndex` additionally exposes `search_lock`, `metadata`,
`metadata_records`, `_row_by_video_frame`, `_rows_by_video`, `index_vectors`,
`allow_npy_fallback`, `cache_info`, `encode_texts()`, `encode_image()`,
`metadata_row_for_ocr_hit()`, `metadata_rows_for_asr_hit()` — identical to the
FAISS version.

### 2.5 Config (`configs/app.yaml`)

```yaml
milvus:
  host: localhost
  port: 19530
  collection_name: keyframes
  metric_type: IP            # IP on L2-normalized vectors == cosine
  search_params: {ef: 128}   # HNSW query params
  metadata_path: .../metadata.csv
  vector_cache_mode: memmap
  vector_cache_dtype: float32
  vector_cache_path: .../vectors_fp32.npy
  allow_npy_fallback: false
```

### 2.6 Ingestion

```
python -m scripts.retrieval.run --task build-milvus-index
```

`BuildMilvusIndexPipeline` reads `configs/indexing.yaml` (`milvus:` + `index:`
blocks), then: build matrix+metadata → normalize → create collection → HNSW
index → batch insert (`row_id = i`) → write `metadata.csv` + `vectors_fp32.npy`
in **the same row order**, guaranteeing `row_id == metadata row == memmap row`.

### 2.7 Dual backend

FAISS is retained as a fallback (`src/retrieval/index/clip_faiss_index.py`).
`system.py` selects via `config["backend"]` (`"milvus"` default, `"faiss"`
fallback) → `build_clip_index(...)`. The runtime default is Milvus.

### 2.8 Known issue

`pymilvus` 3.0 deprecates the ORM-style APIs used here (`Collection`,
`connections.connect`, `utility.*`, `Collection.search/insert`). They still
work on 3.0 but "will be removed in 3.1" — a migration to `MilvusClient` is
the follow-up.

---

## 3. Elasticsearch — text search

### 3.1 Indexes

| Index | Backed by | Used by |
|---|---|---|
| `multimodal_text` | OCR+ASR fused per keyframe | `text` and `fusion` modes |
| `ocr_db` | per-keyframe on-screen text | `ocr` mode |
| `asr_db` | per-segment transcripts | `asr` mode |

### 3.2 Key files

- `src/retrieval/indexer/elasticsearch/client.py` — `ElasticsearchService` (+ `multi_match_search`)
- `src/retrieval/indexer/elasticsearch/multimodal_text/` — `schemas.py`, `repository.py`, `factory.py`, `indexing_pipeline.py`
- `src/retrieval/indexer/elasticsearch/ocr/`, `asr/` — legacy standalone repositories + indexing pipelines
- `src/retrieval/retriever/text_search/` — `SearchPipeline` → `TextHit`
- `scripts/text/index/run.py`, `scripts/ocr/index/run.py`, `scripts/asr/index/run.py`

### 3.3 `multimodal_text` document

```python
class MultimodalTextDocument:
    dataset: str
    video_id: str
    keyframe_id: str          # zero-padded display id ("000123")
    timestamp_sec: float
    ocr_text: str             # joined OCR lines for that keyframe
    asr_text: str             # FULL transcript of the covering speech segment
```

Search is a boosted `multi_match` over `ocr_text^3` + `asr_text^1`
(`best_fields`, `fuzziness: AUTO`) with highlighting → `matched_ocr` /
`matched_asr`.

### 3.4 Fusion ingestion (`IndexPipeline`)

```
python -m scripts.text.index.run        # configs/text.yaml
```

For each video: read `metadata.csv` (keyframe → `timestamp_sec`), the OCR JSON
and the ASR JSON, then:

1. `ocr_text = " ".join(texts)` for keyframes present in the OCR JSON.
2. `asr_text = transcript of the segment where start ≤ timestamp_sec ≤ end`
   (the **full** segment transcript, per keyframe).
3. Index a keyframe only when `ocr_text` **or** `asr_text` is non-empty.

Config (`configs/text.yaml`):

```yaml
elasticsearch: {host: localhost, port: 9200, scheme: http, index: multimodal_text}
dataset:
  ocr_root: data/processed/ocr
  asr_root: data/processed/asr
  metadata_path: .../metadata.csv
```

### 3.5 Version pin

The docker image is Elasticsearch **8.11.3**. The Python client must be **8.x**
(`elasticsearch==8.14.0`) — a 9.x client sends `compatible-with=9`, which the
8.11 server rejects (`media_type_header_exception`).

---

## 4. Fusion layer — semantic + text

`mode="fusion"` combines the two arms into one ranked list.

### 4.1 Key files

- `src/retrieval/retriever/common/fusion.py` — `min_max_normalize`, `reciprocal_rank_fusion`, `fuse`
- `src/retrieval/retriever/common/orchestrator.py` — `fusion_search(...)`, dispatch in `run_search`

### 4.2 Algorithm (`fuse`)

1. **Retrieve** both arms concurrently (`ThreadPoolExecutor(2)`):
   - semantic: `semantic_search.search(plan.events, candidate_k, candidate_k)`
   - text: `text_search(raw_query, candidate_k, oversample_factor=1)`
   (`candidate_k = top_k * candidate_multiplier` — oversample for a real score distribution.)
2. **Normalize** each arm's *raw* score over its own pool with `min_max_normalize`
   (semantic → `retrieval_score`, text → `es_score` BM25). Degenerate pool → 1.0.
3. **Join** on `(video_id, keyframe_id_int)`; a keyframe in both arms carries
   semantic metadata + text provenance (`matched_texts`, `matched_ocr`,
   `matched_asr`, `es_score`).
4. **Combine**: `fused = Σ_m w_m·s'_m + rrf_weight · RRF(d)`, with `RRF(d) = Σ_m 1/(k+rank_m)`
   (consensus bonus; a missing arm contributes 0). Weights renormalize when an arm is empty.
5. **Rank**, `head(top_k)`, set `retrieval_score = fused`, and record
   `fusion_components[arm] = {score, raw_score, normalized, rank}` in each row.

### 4.3 Translation is per-arm

`main.py` translates the query for the **semantic arm** (CLIP is English-
oriented) and passes the **raw** query to the **text arm** (ES matches literal
OCR/ASR text). `should_translate` still skips `ocr`/`asr`/`text` for their
single-mode paths.

### 4.4 Config (`configs/app.yaml`)

```yaml
fusion:
  weights: {semantic: 0.6, text: 0.4}
  rrf: {k: 60, weight: 0.1}
  min_max_epsilon: 1e-6
```

---

## 5. Search modes summary

| Mode | Backend(s) | Query translation |
|---|---|---|
| `semantic` | Milvus | translated |
| `temporal` | memmap DP | translated |
| `ocr` | ES `ocr_db` | raw |
| `asr` | ES `asr_db` | raw |
| `text` | ES `multimodal_text` | raw |
| `fusion` | Milvus + ES | translated (semantic arm) / raw (text arm) |
| `auto` | semantic/temporal | translated |

All modes return through the same `dict_to_result_FAST` serializer, so the
frontend renders them with the same result card.

---

## 6. Running it

```bash
# services (elasticsearch, kibana, milvus + etcd + minio)
docker compose up -d elastic milvus

# 1. build the Milvus vector index (+ metadata.csv + vectors_fp32.npy)
python -m scripts.retrieval.run --task build-milvus-index

# 2. ingest the unified text index (requires OCR/ASR JSONs + metadata.csv)
python -m scripts.text.index.run

# 3. serve the API
uvicorn main:app --host 127.0.0.1 --port 8000

# 4. smoke-test fusion
curl -X POST http://127.0.0.1:8000/api/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"city marathon","top_k":10,"search_mode":"fusion"}'
```

### Sample fusion response

```json
{"mode":"fusion","count":6,
 "results":[{
   "id":"L21_V007_000004","similarity":0.597,
   "matched_texts":["...","Welcome to the city <em>marathon</em> coverage"],
   "raw":{"fusion_components":{
     "semantic":{"normalized":0.990,"rank":1},
     "text":{"normalized":0.000,"rank":6}}}}
 ]}
```

---

## 7. Environment gotchas (seen in practice)

1. **ES disk watermark** — if the host disk is >90% full, Elasticsearch goes
   `red` and refuses shard allocation (`no_shard_available_action_exception`).
   Raise the watermarks for local dev only:
   `PUT /_cluster/settings {"persistent":{"cluster.routing.allocation.disk.watermark.low":"97%","...high":"98%","...flood_stage":"99%"}}`.
2. **ES client/server version** — see §3.5.
3. **pymilvus 3.0 deprecation** — see §2.8.
4. **`metadata.csv` row-order contract** — Milvus `row_id`, the memmap `.npy`
   row, and `metadata.csv` row must stay aligned; `BuildMilvusIndexPipeline`
   guarantees this, so do not hand-edit any one of them independently.
