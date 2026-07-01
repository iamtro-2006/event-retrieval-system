"""FastAPI application entrypoint.

NOTE: No `main.py` / FastAPI app was found in the uploaded project, so this file
is provided as a ready-to-run reference. If you already have a `main.py`
elsewhere (not included in the upload), port the two blocks marked
`# --- integration point ---` into it instead of running this file directly.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers.search import router as search_router
from src.ocr.pipelines.factory import build_ocr_search_pipeline
from src.retrieval.models.retrieval_system import FaissRetrievalSystem

logger = logging.getLogger("api")


def load_app_config(path: str = "configs/app.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build heavyweight, process-shared resources once at startup."""
    cfg = load_app_config()
    faiss_cfg = cfg["faiss"]
    model_cfg = cfg["model"]

    # --- integration point: OCR pipeline -----------------------------------
    # Built independently of FAISS; if Elasticsearch is unreachable at startup,
    # the app still boots and OCR search simply raises a 503 until fixed rather
    # than crashing semantic/temporal search.
    try:
        ocr_pipeline = build_ocr_search_pipeline(config_path="configs/ocr.yaml")
    except Exception:
        logger.exception("Failed to initialize OCR search pipeline; OCR mode will be unavailable.")
        ocr_pipeline = None

    # --- integration point: retrieval system --------------------------------
    app.state.retrieval_system = FaissRetrievalSystem(
        index_path=faiss_cfg["index_path"],
        metadata_path=faiss_cfg["metadata_path"],
        model_name=model_cfg["name"],
        pretrained=model_cfg["pretrained"],
        device=model_cfg.get("device", "auto"),
        precision=model_cfg.get("precision", "fp32"),
        normalize=model_cfg.get("normalize", True),
        ef_search=faiss_cfg.get("ef_search", 64),
        faiss_threads=faiss_cfg.get("threads"),
        vector_cache_mode=faiss_cfg.get("vector_cache_mode"),
        vector_cache_dtype=faiss_cfg.get("vector_cache_dtype", "float32"),
        vector_cache_path=faiss_cfg.get("vector_cache_path"),
        allow_npy_fallback=faiss_cfg.get("allow_npy_fallback", False),
        compile_model=model_cfg.get("compile", False),
        ocr_search_pipeline=ocr_pipeline,
    )

    yield

    app.state.retrieval_system = None


app = FastAPI(title="Video Retrieval API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
