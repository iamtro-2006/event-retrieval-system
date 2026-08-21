"""Pydantic request/response schema cho `src/api/routers/search.py`.

Đây là điểm nối duy nhất giữa HTTP layer và `RetrievalSystem`
(`src/retrieval/system.py`) — router KHÔNG tự định nghĩa shape kết quả,
chỉ serialize DataFrame trả về từ `RetrievalSystem.search_*` qua
`SearchResultItem` (xem `_records_from_df()` trong `routers/search.py`).

Lưu ý quan trọng (KHÔNG phá contract cũ, xem handoff_prompt.md mục 2):
- `AdvancedSearchRequest` dùng `semantic_models: list[str]` + `temporal: bool`
  — KHÔNG có field `temporal_models` (thiết kế cũ đã bị bỏ, xem
  `Orchestrator.advanced_search`). Nếu 1 client cũ gửi `temporal_models`,
  Pydantic sẽ reject field lạ (model được set `extra="forbid"`) thay vì âm
  thầm bỏ qua, để lỗi lộ ra sớm thay vì search chạy sai use-case.
- `translate` (bật/tắt dịch vi->en trước khi search) chỉ có trên
  `SemanticSearchRequest`/`TemporalSearchRequest` (kế thừa)/`AutoSearchRequest`
  — CỐ TÌNH không có trên `OcrSearchRequest`/`AsrSearchRequest`/
  `AdvancedSearchRequest`, vì `Orchestrator` dùng chung 1 raw query cho
  ocr/asr (cần giữ nguyên tiếng Việt) — xem docstring
  `RetrievalSystem.search_advanced`/`search_ocr`/`search_asr` trong
  `src/retrieval/system.py`.
- `AdvancedSearchRequest.weights` được validate strict (xem
  `_validate_weight_keys`): key phải là 1 trong `semantic_models` đã tick,
  hoặc `"temporal"/"ocr"/"asr"` — key lạ/sai chính tả trả 422 thay vì bị
  `Orchestrator.advanced_search` âm thầm bỏ qua (đã làm trong lần sửa sau
  `handoff_prompt_v2.md` mục 5).
- `SearchResultItem` cố tình rất "mở" (`extra="allow"`, hầu hết field
  Optional) vì 4 search mode (semantic/temporal/ocr/asr) trả về các cột
  KHÁC NHAU trên cùng 1 DataFrame shape chung tối thiểu (xem
  `METADATA_DISPLAY_COLUMNS` trong `retrieval/index/faiss_index.py` +
  `retrieval_score`/`rank`/`search_mode`/`model_key` là các cột luôn có,
  còn `matched_texts`, `matched_sequence`, `temporal_start_time`,
  `rrf_score`, `source_models`, ... chỉ xuất hiện tuỳ mode/nguồn). Model API
  KHÔNG được liệt kê cứng toàn bộ field rồi lỡ làm rớt field mới khi
  pipeline nội bộ thêm cột.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


class BaseSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, description="Câu query gốc (tiếng Việt hoặc tiếng Anh).")
    top_k: int = Field(10, ge=1, le=500)


class SemanticSearchRequest(BaseSearchRequest):
    use_split: bool = True
    reasoning: bool = False
    candidate_multiplier: int = Field(5, ge=1, le=50)
    model_key: str | None = Field(
        None, description="Model dùng để search khi backend load nhiều model. Bỏ trống = model mặc định."
    )
    translate: bool | None = Field(
        None,
        description="Dịch query vi->en trước khi search (xem `translate` trong config cho ngôn ngữ nguồn/đích "
        "thực tế). None (mặc định) = dùng `translate.enabled_default` trong config. true/false = ép buộc, "
        "ghi đè config cho riêng request này. true khi subsystem dịch không khả dụng -> 503.",
    )


class TemporalSearchRequest(SemanticSearchRequest):
    duration_limit: float = Field(-1, description="Giới hạn thời lượng chuỗi event (giây). -1 = không giới hạn.")
    reasoning: bool = False


class OcrSearchRequest(BaseSearchRequest):
    pass


class AsrSearchRequest(BaseSearchRequest):
    pass


class AutoSearchRequest(BaseSearchRequest):
    use_split: bool = True
    reasoning: bool = False
    candidate_multiplier: int = Field(5, ge=1, le=50)
    translate: bool | None = Field(
        None,
        description="Xem `SemanticSearchRequest.translate` — an toàn cho auto mode vì auto luôn resolve về "
        "semantic hoặc temporal, không bao giờ ocr/asr.",
    )


class AdvancedSearchRequest(BaseSearchRequest):
    """Xem `Orchestrator.advanced_search` cho full contract — đây chỉ là lớp
    validate HTTP input, KHÔNG lặp lại logic nghiệp vụ."""

    semantic_models: list[str] = Field(
        default_factory=list,
        description="Danh sách model_key đã tick (checklist semantic). Rỗng = không chạy semantic, "
        "và tắt luôn temporal dù `temporal=True` (không có model nào để chạy).",
    )
    temporal: bool = Field(
        False,
        description="Bật/tắt temporal search kết hợp, dùng CHUNG `semantic_models` — KHÔNG có checklist "
        "model riêng cho temporal (thiết kế cũ `temporal_models` đã bỏ).",
    )
    use_ocr: bool = False
    use_asr: bool = False
    use_split: bool = True
    reasoning: bool = False
    candidate_multiplier: int = Field(5, ge=1, le=50)
    duration_limit: float = -1
    weights: dict[str, float] | None = Field(
        None,
        description="Weight theo nhóm nguồn: semantic, ocr, asr. Backend chuẩn hoá tổng weight về 1.0; "
        "nguồn tắt sẽ không được tính.",
    )
    include_per_source: bool = Field(
        False,
        description="True = trả kèm breakdown từng nguồn riêng lẻ trước khi fuse (`per_source`, dùng để "
        "debug/hiển thị UI). Mặc định False để tránh payload lớn không cần thiết.",
    )

    @model_validator(mode="after")
    def _validate_weight_keys(self) -> "AdvancedSearchRequest":
        """Strict validate `weights` (handoff_prompt_v2.md mục 5): trước đây
        `Orchestrator.advanced_search` chỉ đơn giản KHÔNG dùng tới key lạ/sai
        chính tả trong `weights` (không raise) — client gõ nhầm `"temportal"`
        thay vì `"temporal"`, hay 1 model_key đã bị bỏ tick, sẽ bị bỏ qua âm
        thầm, khiến weight tưởng đã áp dụng nhưng thực ra không. Validate ở
        đây (lớp HTTP) để lỗi lộ ra sớm dưới dạng 422, KHÔNG đổi hành vi của
        `Orchestrator.advanced_search` (vẫn giữ nguyên, không raise) — chỉ
        chặn request sai NGAY TỪ router trước khi chạy search."""

        if not self.weights:
            return self
        allowed = set(self.semantic_models)
        # Optional group-level semantic weight used after semantic-model RRF.
        allowed.add("semantic")
        allowed.add("temporal")
        allowed.add("ocr")
        allowed.add("asr")
        unknown = sorted(set(self.weights) - allowed)
        if unknown:
            raise ValueError(
                f"weights có key không hợp lệ: {unknown}. Key hợp lệ: model_key đã tick trong "
                f"semantic_models ({sorted(self.semantic_models)}), hoặc 'temporal'/'ocr'/'asr' "
                "(chỉ khi tính năng tương ứng được bật)."
            )
        return self


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


class SearchResultItem(BaseModel):
    """1 hàng kết quả — xem docstring module này về lý do `extra="allow"`."""

    model_config = ConfigDict(extra="allow")

    dataset: str | None = None
    video_id: str | None = None
    keyframe_id: str | None = None
    keyframe_path: str | None = None
    timestamp_sec: float | None = None
    frame_idx: int | None = None
    retrieval_score: float | None = None
    rank: int | None = None
    display_rank: int | None = None
    search_mode: str | None = None
    model_key: str | None = None


class QueryPlanOut(BaseModel):
    """Debug info — cách query gốc được tách thành event/sub-query, xem
    `Orchestrator.build_query_plan` / `QueryPlan`."""

    mode: str
    use_split: bool
    events: list[list[str]]


class SearchResponse(BaseModel):
    query: str
    mode: str
    count: int
    results: list[SearchResultItem]
    query_plan: QueryPlanOut | None = None


class AdvancedSearchResponse(BaseModel):
    query: str
    count: int
    results: list[SearchResultItem]
    per_source: dict[str, list[SearchResultItem]] | None = Field(
        None, description="Chỉ có khi request `include_per_source=True`."
    )


class AvailableModelsResponse(BaseModel):
    models: list[str]


class ErrorResponse(BaseModel):
    detail: str
    extra: dict[str, Any] | None = None
