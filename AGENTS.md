# Agent entry point

Read these before changing retrieval behavior:
1. `README.md`, `backend/README.md`, `frontend/README.md` for entry points and setup.
2. `backend/docs/HIERARCHICAL_SEARCH.md` for the current search contract, scope matrix, mappings, and checks.
3. `backend/docs/ARCHITECTURE.md` for module boundaries; `backend/docs/PIPELINE_GUIDE.md` for offline data/index preparation.

## Current retrieval implementation

- Real agent: `backend/src/api/routers/plans.py`; runtime instructions: `.agents/skills/hierarchical-video-search/SKILL.md`. `mock_server.py` is a separate synthetic demo.
- Pipeline: API validates plans/resolves scope -> `RetrievalSystem.search_plan` -> `retrieval/plan_search.py` -> retrieval backends.
- `retrieval/search_scope.py` resolves scope and session-local completed pool references. `retriever/common/candidate_filter.py` enforces video/frame constraints; it does not resolve pool IDs.
- Scope is independent of mode. `parent_pool_id` is lineage only; `scope_pool_id` selects the pool used as scope. Never silently expand an empty scope or an outer UI/session video filter.
- Filter eligible candidates before top-k. `pool_frames` supports semantic/hierarchical only; reject unsupported combinations. `pool_videos` searches all frames of the referenced pool's videos within outer limits.
- Preserve hierarchical scoring: top-down uses visual rank after segment candidate selection; bottom-up mixes visual/context (default 0.7/0.3). Missing context is neutral.
- Mapping to nearby boundary frames when a shot has no representative keyframe is intentional. Do not remove this fallback to satisfy an old test. The old `test_csv_roundtrip_boundaries_empty_reverse_and_stale` expects a superseded policy; see the search document.
- Read actual registered tools/capabilities before implementing new retrieval tools. Other skill specifications do not prove runtime support.

## Working rules

Keep API serialization in API modules and retrieval algorithms in retrieval modules. Use `system.py` as the API facade. Keep generated data/indexes separate from code; preserve existing user modifications. Do not inspect media as part of the runtime hierarchical planner.

Run scoped retrieval and plan tests plus frontend agent tests/build after relevant changes. Commands and known limitations are in `backend/docs/HIERARCHICAL_SEARCH.md`. Verify against code if changing contracts, and update that document and the runtime skill together.
