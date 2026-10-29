# Mapped retrieval and sequential search plans

## Prepare data

From `backend/`:

```bash
python -m scripts.build_segment_data --dataset v3c --maps-only
python -m scripts.build_segment_data --dataset v3c --embeddings-root ../data/embeddings/embeddings_qwen3_vl_embedding_2b_vllm
docker compose up -d elastic
python -m scripts.ocr.index.run --dataset v3c --recreate
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

The API rebuilds ASR and caption frame links during startup from the current `data/map_keyframes/<DATASET>/` CSVs and segment JSON. `--maps-only` is available to rebuild links offline. Maps are written under `data/map_data/map_asr` and `map_caption`, one CSV per video. Empty segments retain a row with blank keyframe ID. Time intervals are half-open `[start,end)`.

The second command reads per-video vectors from the shared model tree: `data/embeddings/embeddings_qwen3_vl_embedding_2b_vllm/asr/<DATASET>/<video_id>.npy` and the matching `caption/<DATASET>/<video_id>.npy`. Each is a 2D float32 matrix in the retained segment order of the same-numbered JSON; caption shots lacking mapped keyframes are omitted. The host reads that JSON to create metadata, then writes indexes under `data/database/faiss_qwen3_vl_embedding_2b_vllm/{keyframe,caption,asr}/<DATASET>/`. The source FAISS files are `keyframe.faiss`, `caption.faiss`, and `asr.faiss`; metadata accompanies each index. `vectors_fp32.npy` is a reconstructed vector cache generated from each FAISS index, not a second copy of the downloaded per-video vectors. Search queries for ASR and caption use the same Qwen vLLM encoder as visual retrieval. For one-step local development, omit `--embeddings-root` to embed during index build. Re-run the embedding notebook whenever source text or ordering changes. Do not edit generated CSVs independently of their source data.

No ASR Elasticsearch connection is created at startup. Legacy ES ASR scripts remain available for compatibility. OCR still uses Elasticsearch. Existing semantic, temporal, uploaded-image, image+text, color and frame-similarity endpoints are preserved. Legacy Auto retains semantic/temporal routing; the agent is the broader planner. Temporal plans can assign attached images to events using image_event_indices. Legacy fusion endpoints remain for older clients, but Fusion UI controls are removed.

## Search

Caption segments first receive every keyframe inside their shot. A keyframe outside all shots is assigned to the caption with the nearest start/end boundary. ASR segments receive one representative keyframe from each caption shot they overlap; if an ASR segment overlaps no shot, it receives the nearest keyframe before and after the segment when available. A frame can belong to multiple segments. CSV stores relation and offset. Playback starts at ASR start even when the thumbnail is later. The API regenerates these maps at startup; existing text embeddings do not need rebuilding when only mapping policy changes.

Caption extraction defaults to `extraction.shot_source: keyframes`: `--scenes-dir` must contain the final scene files emitted by keyframe extraction. Missing files cause an explicit error; even `--overwrite` never reruns or overwrites the shared scene detector output. Legacy detection is available only with `shot_source: detect`. Each source scene is one complete caption window, with no 10-second splitting, minimum-duration removal, or final duplicate merging. Profiles `24x448`, `16x448`, `12x448` set maximum sampled frames per scene input; sampling spans the entire scene. Default is `16x448`; the 24-frame profile reserves at least 8192 context tokens. Short scenes can yield fewer frames. Uniform-frame filtering can still omit a scene, so sharing boundaries does not guarantee every scene has a caption. Existing V3C captions must be regenerated from matching scene artifacts; no missing boundaries are inferred from keyframe spacing. `notebooks/extract_caption.ipynb` downloads matching scene files using `SCENES_ROOT_FOLDER_ID` or `SCENES_LOCAL_SOURCE` and selects `CAPTION_PROFILE` before running extraction.

`POST /api/search/plan` is the shared executor for manual requests and agent tools. It returns UI-ready results and accepts `dataset`, `query`, `mode`, `video_ids` plus the plan fields below. `/api/search/caption` and `/api/search/hierarchical` are aliases forcing the corresponding mode. Legacy `/api/search` also accepts caption/hierarchical; use the plan endpoint for custom hierarchical settings.

```json
{
  "dataset": "v3c",
  "query": "A cyclist moving through a street",
  "mode": "hierarchical",
  "direction": "top_down",
  "source_queries": {"caption": ["Cyclists riding through a city street"]},
  "source_weights": {"caption": 1},
  "visual_queries": ["A cyclist riding a bicycle on a city street"],
  "sequence_top_k": 30,
  "pool_k": 300,
  "top_k": 20
}
```

Top-down selects weighted segment ranks, expands mapped frames, deduplicates, applies OCR/speed filters and ranks only by visual similarity. Bottom-up starts with visual candidates and scores caption/ASR linked by reverse maps. `visual_weight` combines visual versus available context; source weights split context importance. Cosines are transformed to `[0,1]`; this is scale alignment, not probability calibration. Context missing for a frame does not become negative evidence. Neither direction inspects pixels with an agent.

`visual_queries`, `query_weights`, `operation=weighted|and` support weighted arithmetic or geometric scoring across visual text queries and attached images (in that order). `source_query_weights` optionally weights queries within each ASR/caption branch. AND is a soft conjunction. `speed=normal|timelapse|slow_motion|unknown` requires caption metadata. `ocr_query` filters candidate frame identities with phrase or terms matching. Hierarchical results are individual ranked frames; ASR/caption direct results retain mapped sequences. Empty mappings produce no frame candidate.

## Agent and pools

The real API exposes `/api/agent/status` and streamed `/api/agent/chat`. Codex CLI must be installed and logged in on the backend host. Existing `mock_server.py` remains an explicitly synthetic demo; it is not used by the real router. The runtime loads `.agents/skills/hierarchical-video-search/SKILL.md` as its actual instruction source.

The tools are `list_capabilities`, `register_plans`, `run_next_plan`. Plans run sequentially. Each pool is stored in `data/search_state/sessions.sqlite3` and emitted immediately as a `pool` event. The producer is independent of the client stream, so switching pools or losing the connection does not interrupt an active search. A stopped turn prevents subsequent plans; current model inference may finish before stopping. Keep one Uvicorn worker for this local agent queue.

`GET /api/agent/sessions/{id}` restores plans/results. `POST /api/agent/sessions/{id}/stop` stops queued work cooperatively. Feedback supplies `parent_pool_id`; a bottom-up plan with `restrict_to_parent=true` ranks that pool's frames, while false searches anew and retains lineage. Pools are snapshots and are not overwritten by child branches. A server restart preserves completed pools, but does not automatically resume interrupted inference. The UI persists session IDs locally and lets users inspect/select plans and give feedback while later plans run.

## Kaggle

Use `notebooks/embed_asr_caption_kaggle.ipynb` to download ASR/caption JSON and maps, precompute Qwen vLLM vectors, and save/upload `asr/<DATASET>/<video_id>.npy` or `caption/<DATASET>/<video_id>.npy` beside the image vectors in the shared model root. Then use `notebooks/hierarchical_search_kaggle.ipynb` to download those vectors plus original JSON, visual data/index, and keyframe maps; build the shared `faiss_qwen3_vl_embedding_2b_vllm/{keyframe,caption,asr}/<DATASET>/` indexes and their vector caches; install Elasticsearch locally without Docker; and await the foreground supervisor. The supervisor starts vLLM for query encoding, seeds OCR, hosts Uvicorn in-process, emits heartbeat logs, and stops owned processes on exit.

Supply a visual index matching the configured visual engine. vLLM requires a compatible separate install/endpoint. Notebook sources and Codex login cannot be invented: configure real sources and credentials through secrets. Foreground logging does not override Kaggle quotas, OOM or platform shutdown. Save outputs to durable storage.

## Cleanup

The obsolete Fusion settings component has been removed. Existing raw captions, ASR, OCR, frame maps and user-modified files are preserved. `data/mock-*`, screenshots and log files may contain useful local diagnostics; no automatic deletion is performed. Generated search state and mapping/index artifacts should be excluded from commits or packaged as dataset artifacts, not checked into source control.
