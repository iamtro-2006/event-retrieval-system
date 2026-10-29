# Mapped retrieval and sequential search plans

## Prepare data

From `backend/`:

```bash
python -m scripts.build_segment_data --dataset v3c --maps-only
python -m scripts.build_segment_data --dataset v3c
docker compose up -d elastic
python -m scripts.ocr.index.run --dataset v3c --recreate
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

The first command copies legacy keyframe CSVs into `data/map_data/map_keyframes/<DATASET>/` only when the destination does not exist. Existing source data is preserved. ASR and caption maps are rebuilt under `map_asr` and `map_caption`, one CSV per video, one segment/frame link per row. Empty segments retain a row with blank keyframe ID. Time intervals are half-open `[start,end)`. Once migrated, edit the maps in `map_data`, then rebuild links/indexes.

The second command uses one multilingual SentenceTransformer to build separate ASR/caption FAISS indexes and versioned atomic pointers. Long documents use token-bounded chunk mean pooling. This is a baseline text encoder, configurable with `--model`; changing it requires a rebuild. Segment IDs retain dataset/video/source. JSON accepts `start/end` or `start_time/end_time`, and transcript/text/caption fields. Source hashes reject stale text/index/map combinations. Do not edit generated CSVs independently of their source data.

No ASR Elasticsearch connection is created at startup. Legacy ES ASR scripts remain available for compatibility. OCR still uses Elasticsearch. Existing semantic, temporal, uploaded-image, image+text, color and frame-similarity endpoints are preserved. Legacy Auto retains semantic/temporal routing; the agent is the broader planner. Temporal plans can assign attached images to events using image_event_indices. Legacy fusion endpoints remain for older clients, but Fusion UI controls are removed.

## Search

ASR first selects keyframes within `[start,end)`. Otherwise it compares the nearest left/right candidates using continuous costs `log1p(d_left/tau)` and `right_weight*log1p(d_right/tau)`. Missing sides cost infinity; ties prefer left. Defaults are tau=5 seconds, right_weight=2, configurable by `--fallback-tau` and `--fallback-right-weight`. Right distance is measured from ASR end, left distance from ASR start. A frame can belong to multiple segments. CSV stores relation, offset_seconds and selection_cost. Playback starts at ASR start even when the thumbnail is a later frame. Rebuild maps after this policy update; unchanged text embeddings do not need rebuilding.

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

Use `notebooks/hierarchical_search_kaggle.ipynb`. It accepts local/Kaggle/Hugging Face/HTTP/Google Drive sources; downloads extracted ASR/OCR/caption data and visual indexes; builds maps and text indexes; installs official Elasticsearch locally without Docker; then awaits a foreground supervisor in the last cell. The supervisor forwards child output, runs OCR seed, hosts Uvicorn in-process, emits heartbeat logs and stops owned children on exit.

Supply a visual index matching the configured visual engine. vLLM requires a compatible separate install/endpoint. Notebook sources and Codex login cannot be invented: configure real sources and credentials through secrets. Foreground logging does not override Kaggle quotas, OOM or platform shutdown. Save outputs to durable storage.

## Cleanup

The obsolete Fusion settings component has been removed. Existing raw captions, ASR, OCR, frame maps and user-modified files are preserved. `data/mock-*`, screenshots and log files may contain useful local diagnostics; no automatic deletion is performed. Generated search state and mapping/index artifacts should be excluded from commits or packaged as dataset artifacts, not checked into source control.
