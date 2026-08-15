"""FastAPI app entrypoint.

Đây là chỗ DUY NHẤT gọi `src.retrieval.system.build_system(...)` — đúng theo
docstring của `build_system()`: "dựng 1 lần trong `api/main.py` lifespan, lưu
vào `app.state`". MỌI router (cả nhánh search MỚI lẫn nhánh legacy) chỉ đọc
lại `app.state.retrieval_system` (xem `src/api/legacy/deps.py` +
`routers/search.py::get_retrieval_system`), không tự build.

Chạy dev server (2 cách tương đương, xem `main.py` ở backend root):
    uvicorn src.api.main:app --reload --port 8000

Config load từ `CONFIG_PATH` env var (mặc định `configs/app.yaml`) — xem
`configs/app.yaml` cho shape `semantic.models` (nhiều model). `ocr`/`asr` là
optional (xem `system.py::_build_ocr_pipeline_or_none/_build_asr_pipeline_or_none`).

Router được chia thành 2 nhánh, CÙNG tồn tại song song (không nhánh nào thay
thế nhánh nào — xem `src/api/legacy/__init__.py`):
  - `routers/search.py`      : nhánh search MỚI, `/api/search/{semantic,
    temporal,ocr,asr,auto,advanced}` + `/api/search/models` — mỗi mode 1
    endpoint riêng, response strict-typed qua `schemas/search.py`.
  - `routers/legacy_search.py`, `routers/health.py`, `routers/dres.py`,
    `routers/speech.py` : API GỐC (`main.py` file lẻ cũ) — `POST /api/search`
    (1 endpoint hợp nhất, chọn mode qua `search_mode`), `/api/health`,
    `/api/config`, `/api/frame-info`, `/api/surrounding-frames`,
    `/api/similarity-search`, `/api/dres/login`, `/api/dres/submit`,
    `/api/speech/transcribe` — giữ NGUYÊN contract cũ cho frontend cũ.
"""

from __future__ import annotations

import asyncio
import cProfile
import io
import os
import pstats
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api.legacy.paths import LegacyPaths, load_yaml
from src.api.routers.dres import router as dres_router
from src.api.routers.health import router as legacy_health_router
from src.api.routers.legacy_search import router as legacy_search_router
from src.api.routers.search import router as search_router
from src.api.routers.speech import router as speech_router
from src.retrieval.system import RetrievalSystem, build_system

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return load_yaml(path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    config_path_str = os.environ.get("CONFIG_PATH", "configs/app.yaml")
    config_path = Path(config_path_str)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path

    config = load_config(config_path)
    app.state.cfg = config
    app.state.legacy_paths = LegacyPaths(REPO_ROOT, config_path, config)
    app.state.should_profile = bool(config.get("debug", {}).get("profile", False))

    # OCR/ASR/model load lỗi hạ tầng (thiếu index/checkpoint, ES down...) đã
    # được `build_system()`/`IndexManager` tự nuốt + skip từng phần optional
    # (in cảnh báo, không raise) — chỉ khi KHÔNG model semantic nào load được
    # thì `build_system()` mới thật sự raise. Giữ pattern degraded-start của
    # nhánh mới: app vẫn lên được, mọi endpoint cần retrieval_system trả 503
    # (xem `get_retrieval_system`/`get_legacy_system`) thay vì app crash.
    try:
        app.state.retrieval_system = build_system(config)
        app.state.retrieval_system_error = None
    except Exception as exc:  # pragma: no cover - lỗi khởi động hạ tầng
        print(f"[api.main] build_system() failed: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        app.state.retrieval_system = None
        app.state.retrieval_system_error = str(exc)

    paths: LegacyPaths = app.state.legacy_paths
    if paths.keyframes_root.exists():
        app.mount("/static/keyframes", StaticFiles(directory=str(paths.keyframes_root)), name="keyframes")
    if paths.videos_root.exists():
        app.mount("/static/videos", StaticFiles(directory=str(paths.videos_root)), name="videos")
    if paths.map_keyframe_root.exists():
        app.mount("/static/map-keyframes", StaticFiles(directory=str(paths.map_keyframe_root)), name="map_keyframes")

    yield

    app.state.retrieval_system = None


app = FastAPI(
    title="event-retrieval-system API",
    description="Video retrieval backend: semantic / temporal / ocr / asr search + advanced_search (RRF fusion), "
    "cùng API gốc (`/api/search`, `/api/health`, `/api/config`, `/api/dres/*`, `/api/speech/transcribe`, ...) "
    "giữ nguyên cho frontend cũ.",
    lifespan=lifespan,
)


# Middleware profile + log bottleneck report cho mọi request /api/* — port
# nguyên từ `main.py` cũ, chỉ khác chỗ đọc cờ bật/tắt từ `request.app.state`
# (set 1 lần trong lifespan) thay vì biến module-level `SHOULD_PROFILE`.
@app.middleware("http")
async def profile_and_bottleneck_tracker(request: Request, call_next):
    """Middleware to profile API endpoints and log bottleneck reports."""
    if not getattr(request.app.state, "should_profile", False) or not request.url.path.startswith("/api/"):
        return await call_next(request)

    pr = cProfile.Profile()
    pr.enable()
    start_time = time.perf_counter()

    response = await call_next(request)
    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(30)

    report_text = (
        f"\n{'='*40} BOTTLENECK PROFILE REPORT {'='*40}\n"
        f"[REQ] {request.method} {request.url.path} | Total Latency: {latency_ms}ms\n"
        f"{'-'*107}\n{s.getvalue()}{'='*107}\n"
    )
    print(report_text)

    # Offload disk I/O to a background thread to prevent event loop blocking
    def _write_log(report: str) -> None:
        log_file = request.app.state.legacy_paths.backend_dir / "search_profile_log.txt"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(report)

    asyncio.create_task(asyncio.to_thread(_write_log, report_text))
    return response


# Permissive CORS cho dev (frontend chạy port khác) — SIẾT LẠI origin cụ thể
# trước khi deploy production, đừng giữ "*" nếu API có auth/cookie.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Nhánh search MỚI (`/api/search/semantic`, `/temporal`, `/ocr`, `/asr`,
# `/auto`, `/advanced`, `/models`).
app.include_router(search_router)

# API GỐC — giữ nguyên path cũ (`/api/search`, `/api/health`, `/api/config`,
# `/api/frame-info`, `/api/surrounding-frames`, `/api/similarity-search`,
# `/api/dres/login`, `/api/dres/submit`, `/api/speech/transcribe`).
app.include_router(legacy_search_router)
app.include_router(legacy_health_router)
app.include_router(dres_router)
app.include_router(speech_router)


@app.get("/")
def root_health() -> dict[str, Any]:
    """Health check tối giản cho load balancer / uptime check (nhánh mới).
    Xem `GET /api/health` cho health check ĐẦY ĐỦ (API gốc, dùng bởi FE cũ)."""
    system: RetrievalSystem | None = getattr(app.state, "retrieval_system", None)
    return {
        "status": "ok" if system is not None else "degraded",
        "retrieval_system_ready": system is not None,
        "error": getattr(app.state, "retrieval_system_error", None),
    }
