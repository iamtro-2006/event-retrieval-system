"""Pydantic request models cho các endpoint API GỐC (`main.py` cũ) — port
nguyên, KHÔNG đổi field/kiểu dữ liệu vì frontend cũ gửi đúng shape này.

Khác với `src/api/schemas/search.py` (nhánh search MỚI, `extra="forbid"`
nghiêm ngặt), các model ở đây giữ đúng độ "lỏng" của bản gốc (không set
`extra="forbid"`) để không vô tình reject request cũ có thêm field lạ mà
FE cũ có thể đã gửi kèm.
"""

from __future__ import annotations

from pydantic import BaseModel

from src.retrieval.retriever.common.orchestrator import SearchMode


class SearchRequest(BaseModel):
    """Payload cho `POST /api/search` (endpoint search HỢP NHẤT gốc, chọn
    mode qua `search_mode` — khác với nhánh mới `/api/search/{semantic,
    temporal,ocr,asr,auto,advanced}` là các endpoint tách riêng)."""

    query: str
    top_k: int | None = None
    candidate_multiplier: int | None = None
    use_split: bool | None = None
    use_translate: bool | None = None
    search_mode: SearchMode | None = "semantic"
    duration_limit: float | None = -1


class FusionSearchRequest(BaseModel):
    """Payload cho `POST /api/search/fusion` — nhánh search MỚI
    (`Orchestrator.advanced_search` / RRF fusion), nhưng trả về CÙNG shape
    với `POST /api/search` (qua `dict_to_result_FAST`) để tái dùng nguyên
    UI/serializer hiện có thay vì thêm 1 shape response mới.

    "Method" trên UI (semantic model / temporal / ocr / asr) map trực tiếp
    vào `weights`: key là model_key (semantic) hoặc "temporal"/"ocr"/"asr".
    """

    query: str
    semantic_models: list[str] = []
    temporal: bool = False
    use_ocr: bool = False
    use_asr: bool = False
    top_k: int | None = None
    candidate_multiplier: int | None = None
    use_split: bool | None = None
    use_translate: bool | None = None
    duration_limit: float | None = -1
    weights: dict[str, float] | None = None


class DresLoginRequest(BaseModel):
    dres_url: str
    username: str
    password: str


class DresSubmitRequest(BaseModel):
    dres_url: str
    session_id: str
    evaluation_id: str | None = None
    video_id: str
    frame_id: int
    timestamp: float | None = None


class SimilaritySearchRequest(BaseModel):
    video_id: str
    frame_id: int
    top_k: int | None = None
