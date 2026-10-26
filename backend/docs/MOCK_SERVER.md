# Local mock server + Codex chat

Standalone entrypoint: `backend/mock_server.py`. No real retrieval imports,
FAISS, Elasticsearch, model loading, or API key required. Use port **8001**;
the real backend remains on 8000. The Codex integration has not been connected to
`RetrievalSystem`: retrieval/embedding/index pipelines are separate. The real
Surrounding Frames route also accepts `all_frames=true` for the video browser;
that small API extension does not change the retrieval pipelines.

## Connection files

| File | Responsibility |
| --- | --- |
| `mock_server.py` | FastAPI routes, sessions and NDJSON streaming |
| `src/mock_codex.py` | CLI discovery, managed login, app-server transport, prompts and tool dispatch |
| `src/mock_tools.py` | Tool schemas, argument validation and `MockCatalog` |
| `src/mock_workers.py` | Bounded worker processes and cancellation |
| `../frontend/src/api/agentAPI.js` | HTTP requests and NDJSON decoding |
| `../frontend/src/components/AgentChat.jsx` | Chat state, cancellation and sidebar activity log |
| `../frontend/src/components/AgentReasoningPicker.jsx` | Model catalog and effort slider |

```text
SearchBar -> AgentChat -> agentAPI -> mock_server
                                    -> Codex app-server <-> MockTools / workers
                                    -> NDJSON events -> sidebar log + result grid
```

Prompts and tool definitions are written in Python. There is no application skill
loader or retrieval-specific `SKILL.md` workflow yet. The model may repeat searches
within its tool budget, but refinement is not a mandatory programmed loop.

## Start

Install Python 3.11+, Node.js 22.12+ and the official
[Codex CLI](https://learn.chatgpt.com/docs/cli) on the new machine. On Windows,
use a native `codex.exe` installation. Open a new terminal after installing it,
then sign in under the same OS user that will run the backend:

```powershell
codex --version
codex login
codex login status
```

Each machine uses its own local CLI installation and login. Credentials are not
part of the repository and are not configured in Vite. Make sure the source files,
`frontend/.env.mock` and lockfile have been committed/pushed before cloning on
another machine. Virtual environments, `node_modules`, data and runtime state are
not transferred by Git.

From the repository root, create an isolated mock environment and start backend:

```powershell
python -m venv backend/.venv-mock
.\backend\.venv-mock\Scripts\python.exe -m pip install -r backend/requirements-mock.txt
cd backend
.\.venv-mock\Scripts\python.exe -m uvicorn mock_server:app --host 127.0.0.1 --port 8001
```

For macOS/Linux, use `python3` to create the environment and
`backend/.venv-mock/bin/python` instead of the Windows `Scripts/python.exe` path.
No activation is required when calling the environment's Python directly.

In another terminal, starting at the repository root:

```powershell
cd frontend
npm ci
npm run dev:mock
```

The backend starts app-server subprocesses automatically; do not start a separate
`codex app-server` terminal. Only the lightweight `requirements-mock.txt` is needed,
not the GPU dependencies in the main `requirements.txt`.

Open the printed Vite URL, normally http://127.0.0.1:5173. If busy, use
`npm run dev:mock -- --port 5174`. Localhost frontend ports are allowed by CORS.
API docs: http://127.0.0.1:8001/docs.

`.env.mock` overrides API/media/DRES settings only in mock mode. Existing `.env`
is preserved; `npm run dev` still uses its settings for the real backend.

## Dataset behavior

The server scans existing image files once during startup. `data/` is ignored by
Git: copy the images separately on a new machine. Chat works without images;
retrieval and inspection require local dataset files:

```text
data/keyframes/V3C/<video_id>/*.{jpg,jpeg,png,webp}
data/keyframes/LHE/<video_id>/*.{jpg,jpeg,png,webp}
data/videos/{V3C,LHE}/...  # optional local media
```

Set `MOCK_DATA_ROOT` to use another data root. Restart after adding files.
Startup may take several seconds on large datasets. Images always belong to the
selected physical dataset; video filters apply. Empty datasets return no images,
never images borrowed from another dataset or remote placeholders.

Semantic/temporal/OCR/ASR/auto/fusion/color/multimodal/similarity searches return
random samples. Scores, transcripts and event labels are simulated. Playback timing comes from
`data/map_keyframes/{V3C,LHE}/<video_id>.csv` under the same `MOCK_DATA_ROOT`.
CSV columns are `keyframe_id,video_id,frame_idx,timestamp_sec,fps`; invalid or
missing rows fall back to numeric filename seconds and set `synthetic_timing=true`.
Neighbors are actual adjacent files in the same video, sorted by numeric filename.
The Surrounding Images modal requests `all_frames=true` to browse the entire video,
highlights and centers the chosen frame, and allows scrolling to either end.
The frontend caches the six most recently opened video lists; clicking Surround
on another frame in the same video re-centers the existing list without refetching.
The modal uses TanStack Virtual to mount only visible rows plus two buffer rows on
each side. Images retain native lazy loading and async decoding. Metadata for the
whole video remains available, while offscreen cards are unmounted; fixed row
geometry preserves scroll position and centering as images load.
Playback requires a matching local video. Seeking uses mapped timestamps where
available; the player explicitly marks synthetic timing when no mapping exists.
Mock reranking preserves supplied order. DRES, speech and translation endpoints
return explicit mock responses without external calls. The independent God Mode
relay is not started by this server.

## Chat engines

Use the **main search composer** for every search/chat submission. It always
uses the expanded layout (156 px minimum height) on home and result screens.
Input attachments are at the left; the controls are aligned to the right.

- **Reasoning off:** choose a retrieval model and search mode, then run direct
  retrieval. This remains simulated in mock mode.
- **Reasoning on:** Codex is selected automatically. The model/effort popup shows
  the available model name, reasoning slider and reset button. Clicking the model
  name opens the model list. There is no Account default, engine or worker picker.
  The initial model and effort come from the catalog; incompatible effort is reset
  when changing models. The selected effort is sent as `turn/start.effort` and is
  inherited by workers. The request uses preferred search mode `auto`.
- **While running:** a purple glow indicates activity and Send becomes Stop.
  Stop aborts the stream and cancels/joins workers. Direct retrieval Stop cancels
  the client request; synchronous real retrieval may still finish on the backend.

On result screens the composer floats over the image grid, 18 px from the bottom
plus any safe-area inset. Idle opacity is 0.68; hover, focus, an open popup or an
active request makes it fully visible. Touch devices keep it visible. The blue
Queue launcher remains a separate fixed button at the bottom right.

**Assistant log** shares the expanded left sidebar. Opening it replaces navigation
with activity; Back to menu or Close restores navigation. Hiding it keeps chat
history and does not stop the current turn. The log has no second input/send/stop
field. It renders Markdown in assistant and worker reports. Suggestion and reuse
actions populate the main composer without submitting; reused groups attach exact
frame IDs. Color canvas also applies a draft and waits for the main Send action.

The API retains two providers for development and tests:

- **Demo** runs capabilities → search → inspect through the shared registry and
  streams a scripted answer. Large groups use concurrent image preparation with
  explicit Demo reports; no AI interprets images in this mode.
- **Codex · real AI** uses the installed Codex app-server. The model chooses tools,
  sees real image pixels for inspect, and streams chat/tool activity. Auth/quota
  failures are surfaced; there is no silent fallback to Demo.

For Codex mode, install the official CLI and sign in from your terminal:

```powershell
codex login
codex login status
```

Run backend under the same OS user/environment. Set `MOCK_CODEX_BIN` if needed.
The bridge reuses the CLI's managed login without reading/exporting tokens or
editing user config. It does not implement the separate Sign in with ChatGPT flow.
This is a local development integration, not a hosted account-sharing service.
Codex turns consume account usage. A model catalog entry does not prove inference
access; a successfully completed turn does.

App-server protocol: initialize with experimentalApi → account/read → ephemeral
thread/start with dynamicTools → turn/start → item/tool/call/tool responses →
agentMessage deltas → turn/completed. State DB lives under `data/mock-state`.
Official reference: https://learn.chatgpt.com/docs/app-server.

Each HTTP turn owns one Codex process/thread, terminated on completion, Stop,
disconnect or timeout. Delegated workers each own a separate process, ephemeral
thread and isolated runtime directory beneath `data/mock-state/workers/`.
`codex.stderr.log` in each runtime helps diagnose startup failures. The backend keeps bounded recent chat text and known frame
IDs in memory for follow-ups; it does not resume a persistent Codex thread.
Restarting backend expires sessions. Dataset/engine/model changes start new chat.
Closing the panel hides it; **Stop** cancels work. **New chat** clears messages
and the displayed results. No LangGraph or multi-agent dependency is needed.

Limits: 12 coordinator tool calls, 240 seconds after coordinator turn startup,
2 concurrent chat coordinators, up to 3 concurrent workers globally. The API
accepts a per-request limit of 1–3; the current frontend uses 3 internally. Each worker has at most 4 retrieval
tool calls and 90 seconds after startup, with no recursive delegation. Sessions:
100 with one-hour idle expiry, 16 stored messages/session. Image inputs: up to 6
JPEG/PNG/WebP/GIF data URLs per turn (2 MB each in FE). Bulk inspection accepts up
to 48 known frame IDs split into batches of 6; independent delegation accepts up
to 6 tasks. Extra tasks queue behind the worker limit. Multiple agents consume
additional account usage and are used only when the coordinator delegates.
Unsupported requests for shell/approval tools are rejected;
the child uses read-only sandbox and disables shell, apps, and web search.

## Shared tools

`src/mock_tools.py` owns schema validation and the common catalog. Dataset is fixed
by the session, not supplied by the model. Inspect/present accept known frame IDs.

| Tool | Behavior |
| --- | --- |
| list_capabilities | Dataset counts, modes, limitations |
| search | Random samples using requested search mode |
| inspect_frames | Actual resized image pixels and frame metadata |
| get_surrounding_frames | Real neighboring files from the same video |
| similarity_search | Simulated similarity from an existing source frame |
| present_results | Choose/reorder known frames on the frontend grid |
| parallel_inspect | Independent vision workers inspect groups of up to 6 images and return evidence reports |
| parallel_tasks | Independent workers run bounded retrieval/analysis tasks and return reports/candidates |

HTTP routes and Codex share `MockCatalog`; `src/mock_codex.py` handles transport.
A real integration can replace the catalog adapter with `RetrievalSystem` calls
while retaining the tool contracts.

`POST /api/agent/chat`: `message`, `dataset`, `provider`, optional `model`,
`reasoning_effort`,
`session_id`, `top_k`, `video_ids`, `max_workers`, `context_frame_ids`, `images`,
`search_mode`, `color_cells`. Streams NDJSON events: session, status,
tool_start, tool_end, worker_start, worker_delta, worker_end, results, delta, done/error. FE supports cancellation,
truncated-stream detection, tool argument details and live updates to the existing
result grid. Connection and model catalog: `GET /api/agent/status`; tool schemas:
`GET /api/agent/tools`. Status includes `default_reasoning_effort` and
`supported_reasoning_efforts` for each model, obtained from app-server `model/list`.
Only a completed chat verifies inference access; a model catalog alone does not.

## Connection troubleshooting

- Check `/api/health` and `/api/agent/status`. `codex.available=true` confirms
  login/catalog access; send a short chat to verify inference and quota.
- If CLI discovery fails, set `MOCK_CODEX_BIN` to the absolute native executable
  before starting backend. On Windows, this should point to `codex.exe`, not an
  npm `.cmd` shim. Do not hard-code another user's installation path in source.
- Run `codex login status` from the same user/environment as the backend. Login
  and quota failures are returned to the UI without a Demo fallback.
- Inspect `data/mock-state/codex.stderr.log` for startup errors. `MOCK_DATA_ROOT`
  changes both the dataset root and its `mock-state` subdirectory.
- Restart Vite after changing `VITE_*` variables. Use `dev:mock`, not `dev`, for
  port 8001. Restart backend after adding images or changing backend Python code.


## Video playback and keyframe strip

`GET /api/video-keyframes?dataset=...&video_id=...` returns chronological rows with
`keyframe_id`, decoded `frame_idx`, `timestamp` (seconds), `fps`, `image_url`,
`image_rel_path` and `synthetic_timing`. The real endpoint builds this contract
from that dataset's FAISS metadata; mock builds it from the local catalog and CSV
mapping. Both use `src/api/legacy/video_keyframes.py` for the response contract.
The core embedding/retrieval/search algorithms are unchanged.

The frontend resolves `image_rel_path` through the same media host configuration
as search results. It never opens a local CSV path or guesses another dataset.
CSV files in mock are read on the backend from `MOCK_DATA_ROOT`; `map_rel_path`
and `/mapkeyframes/...` URLs remain dataset-relative for media hosting.

The old custom ruler/zoom/range timeline is replaced by a virtualized thumbnail
strip. Labels show sparse keyframe IDs, not decoded video frame numbers. Clicking
an image seeks to its timestamp. Playback highlights the last keyframe at or before
the current time, moves the gold star and centers that image. Scrolling clamps at
the first/last image without adding blank frames. Exact milliseconds, seek-step,
Copy ID, Queue and Submit remain available; the native video controls also remain.

## Checks

From the repository root (Windows):

```powershell
.\backend\.venv-mock\Scripts\python.exe -m pip install httpx
cd backend
.\.venv-mock\Scripts\python.exe -m unittest discover -s tests -p test_mock_server.py -v
.\.venv-mock\Scripts\python.exe -m unittest discover -s tests -p test_video_keyframes.py -v
cd ../frontend
npm run lint
npm run build:mock
npm run test:agent
npm run test:video
npm run test:ui-mocks
```

Tests use tiny temporary datasets and fake Codex protocol messages; no credentials
or real data needed. The agent command checks both NDJSON streaming and catalog/model-effort selection.
Live Codex smoke tests consume quota and can be run from the main composer.
