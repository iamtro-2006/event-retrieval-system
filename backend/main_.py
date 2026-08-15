"""Compatibility entrypoint for ASGI servers.

The runtime FastAPI app lives in `src.api.main`. Keep this file tiny so there
is only one system bootstrap path and one config contract.

Run either command from `backend_fixed_v4`:
    uvicorn main:app --reload --port 8000
    uvicorn src.api.main:app --reload --port 8000
"""

from src.api.main import app

__all__ = ["app"]
