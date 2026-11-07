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

The second command reads one vector per segment from the shared model tree: `data/embeddings/embeddings_qwen3_vl_embedding_2b_vllm/asr/<DATASET>/<video_id>/<segment_number>.npy` and the matching caption tree. Caption shots are mapped to in-shot frames, nearest shot boundaries, or nearest keyframes on both sides for an empty shot; a missing vector skips only that segment. The host reads that JSON to create metadata, then writes indexes under `data/database/faiss_qwen3_vl_embedding_2b_vllm/{keyframe,caption,asr}/<DATASET>/`. The source FAISS files are `keyframe.faiss`, `caption.faiss`, and `asr.faiss`; metadata accompanies each index. `vectors_fp32.npy` is a reconstructed vector cache generated from each FAISS index, not a second copy of the downloaded per-video vectors. Search queries for ASR and caption use the same Qwen vLLM encoder as visual retrieval. For one-step local development, omit `--embeddings-root` to embed during index build. Re-run the embedding notebook whenever source text or ordering changes. Do not edit generated CSVs independently of their source data.

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

Top-down selects weighted segment ranks, expands mapped frames, deduplicates, applies OCR/speed filters and ranks only by visual similarity. It uses at least `2 × top_k` segment hits and a mapped-frame pool of at least `2 × top_k` before reranking. Bottom-up starts with visual candidates and scores caption/ASR linked by reverse maps. `visual_weight` combines visual versus available context; source weights split context importance. Cosines are transformed to `[0,1]`; this is scale alignment, not probability calibration. Context missing for a frame does not become negative evidence. Neither direction inspects pixels with an agent.

`visual_queries`, `query_weights`, `operation=weighted|and` support weighted arithmetic or geometric scoring across visual text queries and attached images (in that order). `source_query_weights` optionally weights queries within each ASR/caption branch. AND is a soft conjunction. `speed=normal|timelapse|slow_motion|unknown` requires caption metadata. `ocr_query` filters candidate frame identities with phrase or terms matching. Hierarchical results are individual ranked frames; ASR/caption direct results retain mapped sequences. Empty mappings produce no frame candidate.

## Agent and pools

The real API exposes `/api/agent/status` and streamed `/api/agent/chat`. Codex CLI must be installed and logged in on the backend host. Existing `mock_server.py` remains an explicitly synthetic demo; it is not used by the real router. The runtime loads `.agents/skills/hierarchical-video-search/SKILL.md` as its actual instruction source.

The tools are `list_capabilities`, `register_plans`, `run_next_plan`. Plans run sequentially. Each pool is stored in `data/search_state/sessions.sqlite3` and emitted immediately as a `pool` event. The producer is independent of the client stream, so switching pools or losing the connection does not interrupt an active search. A stopped turn prevents subsequent plans; current model inference may finish before stopping. Keep one Uvicorn worker for this local agent queue.

The runtime skill is self-contained because the bridge loads only its `SKILL.md` text. Its decision order is: establish evidence/requirements, choose scope, choose evidence path, write modality-specific queries and weights, register/execute, refine/report. Keep schema examples aligned with `SearchPlan`; adding a separate reference file does not automatically make it visible to the runtime.

Multi-query in one plan yields one pool; different plans yield independent pools. `query_weights` combines visual text queries followed by attached images. `source_query_weights` combines queries within an ASR/caption source, `source_weights` combines active sources, and `visual_weight` mixes visual/context only in bottom-up. Default optional weights are equal. Unequal weights should reflect user emphasis or distinguishing clues, not invented confidence. Visual `operation=and` is a soft geometric conjunction, never an exact constraint. Direct ASR/caption takes one query; source-specific multi-query is hierarchical. Runtime capabilities and actual source readiness always take precedence over examples. Skill structure/schema checks do not measure retrieval accuracy; real query evaluation remains necessary before claiming improved accuracy.

`GET /api/agent/sessions/{id}` restores plans/results. `POST /api/agent/sessions/{id}/stop` stops queued work cooperatively. `parent_pool_id` records branch lineage only. Use `scope` and `scope_pool_id` explicitly to restrict retrieval (see below). Pools are snapshots and are not overwritten by child branches. A server restart preserves completed pools, but does not automatically resume interrupted inference. The UI persists session IDs locally and lets users inspect/select plans and give feedback while later plans run.

## Scope contract and module ownership

This section describes the implemented real runtime, not a proposed tool design. Start here when continuing scope work without chat history.

| Scope | Required plan fields | Meaning | Supported modes |
| --- | --- | --- | --- |
| `dataset` (default) | No `video_ids` or `scope_pool_id` | Current dataset within outer UI/session limits | All available modes |
| `videos` | Nonempty `video_ids` | Specified videos intersected with outer limits | All available modes |
| `pool_frames` | `scope_pool_id` | Only mapped frame identities already in that completed pool | `semantic`, `hierarchical` |
| `pool_videos` | `scope_pool_id` | All frames in videos represented by that completed pool; can discover new moments | All available modes |

Available modes are semantic, temporal, ASR, caption, OCR, hierarchical, similarity and color, subject to source readiness. Pool-frame semantics for direct segment/OCR/temporal/similarity/color modes are not implemented and are rejected, never substituted. A pool scope is available only in an agent session: the manual plan endpoint rejects pool references. `video_ids` is a per-plan field for `videos`; the chat request's `video_ids` is the outer session limit. Pool references must resolve to completed entries in the same session/dataset. A selected pool is context, never an automatic restriction.

Examples of agent plan fields (pool identifiers come from capabilities):

```json
{"query":"skateboarder near graffiti", "mode":"semantic", "scope":"videos", "video_ids":["00001"]}
```

```json
{"query":"skateboarder near graffiti", "mode":"semantic", "scope":"pool_frames", "scope_pool_id":"<completed-pool-id>"}
```

Use `pool_videos` instead of `pool_frames` in the second example to find additional moments in those videos. Add `parent_pool_id` separately if recording branch lineage is desired.

Execution ownership:

1. `api/routers/plans.py`: request schema, validation, tool registration, session persistence, response serialization. `list_capabilities` exposes `scope_modes`, `current_scope`, and `referenceable_pools`.
2. `retrieval/search_scope.py`: scope/mode support matrix, completed-pool validation, expansion of sequence cards using their existing `matched_sequence`, outer-limit intersection, explicit empty scope. Does not invent neighboring frames.
3. `retrieval/system.py::search_plan`: API facade.
4. `retrieval/plan_search.py`: shared dispatch for manual/agent retrieval. Empty scopes return no results without calling backends.
5. `retrieval/retriever/common/candidate_filter.py`: request-local video/frame constraints through ContextVars, with restoration on exit/failure. This module was renamed from `video_filter.py`; use the new import path. It accepts IDs/identities, not pool IDs.
6. Semantic FAISS selects all allowed vector rows before top-k; segment retrieval scores eligible segments before truncation. Hierarchical top-down excludes segments without an eligible mapped frame and clips mapped frame expansion to scope. Bottom-up visual retrieval is scoped before context scoring. OCR video restrictions enter the Elasticsearch query; color restricts candidate rows before ranking. Temporal event candidate retrieval uses the scoped semantic search.
7. `frontend/src/components/AgentChat.jsx`: passes outer video and selected pool context, displays each plan's scope and persisted `resolved_scope` (videos/frame count), and permits revisiting completed pools.

Do not search global top-k and then filter: an in-scope candidate may be absent from the global list. Do not change the hierarchical score formulas as part of scope maintenance. `visual_weight` applies to bottom-up; top-down final ranking is visual only. Context weights are not confidence probabilities.

Compatibility: legacy `restrict_to_parent=true` with `parent_pool_id` normalizes to `scope=pool_frames`, `scope_pool_id=parent_pool_id`. Conflicting explicit fields are rejected. New agents must use explicit scope fields. Legacy manual requests with video IDs and no scope normalize to `videos`; an empty legacy video list means the default dataset scope. Empty resolved intersections and empty pools remain empty; they must never become unrestricted searches. ContextVar video allowlists retain legacy empty-means-unrestricted semantics internally, so always resolve and short-circuit empty scopes before backend dispatch.

## Mapping policy and regression checks

Nearby boundary frame fallback when a caption shot has no representative keyframe is intentional. Multiple captions may map to one frame. Preserve existing mapping relations and offsets; do not remove valid fallback links merely to enforce in-shot-only mapping.

`tests/test_csv_roundtrip_boundaries_empty_reverse_and_stale` is shorthand for the method of that name in `tests/test_mapped_search.py`. It currently expects an older in-shot-only mapping for one fixture and fails against the existing boundary fallback policy, including with the committed hierarchical implementation. This is a stale test expectation, not evidence that fallback mappings should be removed. Scope/scoring checks are independent of that fixture. Updating the fixture expectations to the accepted mapping policy remains follow-up work.

From `backend/`, with dependencies installed:

```bash
python -m unittest discover -s tests -p test_search_scope.py
python -m unittest discover -s tests -p test_search_plans.py
python -m unittest discover -s tests -p test_mapped_search.py
```

The mapping suite includes the stale expectation above; do not claim the full suite passes until it is updated. Scope tests cover pre-top-k retrieval, pool/video expansion, empty intersections, invalid pool references, legacy conversion, capability registration, context cleanup, and hierarchical scoring. No live model/GPU/index integration is implied by these unit tests.

From `frontend/`, run `npm run test:agent` and `npm run build`. On the current development workspace, isolated Python test dependencies may exist in ignored `.scope-test-deps/`; they are local tooling, not production dependencies or a required repository layout.

## Kaggle

Use `notebooks/embed_asr_caption_kaggle.ipynb` to download ASR/caption JSON and maps, precompute Qwen vLLM vectors, and save/upload each segment vector at `asr/<DATASET>/<video_id>/<segment_number>.npy` and `caption/<DATASET>/<video_id>/<segment_number>.npy` beside the image vectors in the shared model root. Then use `notebooks/hierarchical_search_kaggle.ipynb` to download those vectors plus original JSON, visual data/index, and keyframe maps; build the shared `faiss_qwen3_vl_embedding_2b_vllm/{keyframe,caption,asr}/<DATASET>/` indexes and their vector caches; install Elasticsearch locally without Docker; and await the foreground supervisor. The supervisor starts vLLM for query encoding, seeds OCR, hosts Uvicorn in-process, emits heartbeat logs, and stops owned processes on exit.

Supply a visual index matching the configured visual engine. vLLM requires a compatible separate install/endpoint. Notebook sources and Codex login cannot be invented: configure real sources and credentials through secrets. Foreground logging does not override Kaggle quotas, OOM or platform shutdown. Save outputs to durable storage.

## Cleanup

The obsolete Fusion settings component has been removed. Existing raw captions, ASR, OCR, frame maps and user-modified files are preserved. `data/mock-*`, screenshots and log files may contain useful local diagnostics; no automatic deletion is performed. Generated search state and mapping/index artifacts should be excluded from commits or packaged as dataset artifacts, not checked into source control.
