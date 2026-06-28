# CONTEXT — event-retrieval-system domain glossary

This file names the domain concepts the codebase manipulates. Architecture
reviews and ADRs should use these terms verbatim rather than ad-hoc synonyms.

## Core concepts

- **Retrieval Backend** — a module that searches a **Vector Store** for
  keyframe embeddings matching a **Query Plan**. Two adapters exist:
  `FaissRetrievalSystem` (in-process HNSW index) and `MilvusRetrievalSystem`
  (Milvus Standalone over HTTP). Both implement the `RetrievalBackend`
  Protocol.

- **Vector Store** — the indexed embedding collection a **Retrieval Backend**
  searches. FAISS index file on disk, or a Milvus collection. A backend owns
  exactly one vector store.

- **Keyframe** — a representative still frame extracted from a video shot.
  Identified by `(dataset, video_id, keyframe_id_int)` and carrying a path to
  its image, its embedding, and optional map metadata (timestamp, fps,
  frame index).

- **Query Plan** — the parsed, mode-resolved form of a user query. Produces
  `events: list[list[str]]` (temporal events, each with semantic sub-queries)
  plus the effective search mode. Built by `build_query_plan` in
  `src/index/query_planning.py`.

- **Temporal Search** — ordered alignment of a multi-event query against a
  sequence of keyframes within a single video. Solves a strict-order dynamic
  program over the query×frame similarity matrix, then non-maximum-suppresses
  overlapping results. Lives in `src/index/temporal.py` and exposes
  `search(candidates, query_embeddings, ...) -> list[TemporalMatch]`.

- **Candidates** — the value object a **Retrieval Backend** assembles and
  hands to **Temporal Search**: per-video arrays of `video_id`,
  `timestamp_sec`, `row_index`, the metadata `records`, and the
  `embedding_matrix` rows for those candidates. Replaces the prior implicit
  DataFrame contract (`_faiss_id` column injection).

- **Result Serialization** — shaping a retrieval hit dict into the HTTP
  response shape (keyframe URL, video URL, frame id text, matched sequence).
  Lives in `src/api/serialization.py` and is shared by the FastAPI app and
  the mock API.

- **DresClient** — adapter around the external DRES (Diverse Interactive
  Video Retrieval Evaluation Server) HTTP API. Handles login, submission,
  and verdict normalization. Lives in `src/api/dres.py`. One external
  dependency = one real seam.

## Pipelines

- **Keyframe Extraction** — TransNetV2 shot detection + embedding-based
  clustering to produce **Keyframes** from raw video. Entry point:
  `backend/cli.py` → `KeyframeExtractionPipeline`.

- **Index Build** — walks a directory of per-keyframe `.npy` embeddings and
  map CSVs, builds a matrix + metadata table, and writes them into a
  **Vector Store**. Shared metadata assembly lives in
  `src/index/embedding_index.py`; FAISS-specific index creation in
  `src/index/faiss_index.py`; Milvus collection creation in
  `src/pipelines/build_milvus.py`.

## Config domains

- **Keyframe Extraction Config** — `configs/kf_extraction.yaml`, modeled by
  `AppConfig` in `src/utils/config.py`. Drives the extraction pipeline.

- **Retrieval Config** — `configs/app.yaml`, modeled by `RetrievalConfig`
  in `src/config/retrieval_config.py`. Drives the retrieval backends and the
  FastAPI app. Sections: `backend`, `model`, `milvus`, `faiss`, `search`,
  `ui`, `translate`, `speech`, `debug`, `paths`.

- **Indexing Config** — `configs/indexing.yaml`, consumed raw by the
  indexing CLIs (`src_faiss.py`, `src_milvus.py`, `src_embeddings.py`,
  `src_system.py`). Not yet typed.
