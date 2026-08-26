# Tach tu retrieval_system.py cu (REFACTOR_PLAN.md Prompt 4): day la phan
# "auto mode" - chon mode (semantic/temporal/ocr/asr/auto) va gop ket qua tu
# nhieu nguon. Orchestrator KHONG tu quan ly viec ghi index (do la viec cua
# indexer/*), no chi goi vao SemanticIndex (embedding+FAISS) va cac search
# pipeline cua OCR/ASR (Elasticsearch-backed) da duoc build san boi factory.
#
# Luu y (chua hoan tat theo dung nguyen tac muc 5 cua REFACTOR_PLAN.md):
# constructor van nhan truc tiep OCRSearchPipeline/ASRSearchPipeline (concrete
# types) thay vi chi phu thuoc vao BaseRetriever interface, vi 2 pipeline nay
# co method `.search(query, top_k)` tra ve shape hits rieng (khong phai list[Hit])
# ma _enrich_ocr_hits/_enrich_asr_hits can xu ly truoc khi chuan hoa. Nang cap
# len dung BaseRetriever se can sua ca ocr_search/asr_search pipeline de tra ve
# Hit chuan hoa - de lai cho 1 pass rieng.
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING

import numpy as np
import pandas as pd

from src.retrieval.index.faiss_index import METADATA_DISPLAY_COLUMNS, FaissIndex
from src.retrieval.index.index_manager import IndexManager
from src.retrieval.retriever.common.scoring import reciprocal_rank_fusion
from src.retrieval.retriever.temporal_search.pipeline.search import temporal_search_from_score_candidates
from src.retrieval.retriever.semantic_search.pipeline.search import clean_queries

if TYPE_CHECKING:
    from src.retrieval.retriever.semantic_search.pipeline.search import SearchPipeline as SemanticSearchPipeline
    from src.retrieval.retriever.temporal_search.pipeline.search import SearchPipeline as TemporalSearchPipeline
    from src.retrieval.retriever.ocr_search.pipeline.search import SearchPipeline as OCRSearchPipeline
    from src.retrieval.retriever.asr_search.pipeline.search import SearchPipeline as ASRSearchPipeline
    from src.query_enrichment.llm_query_engine import LLMQueryEngine

SearchMode = Literal["semantic", "temporal", "ocr", "asr", "auto"]


def split_temporal_events(query: str) -> list[str]:
    """Split a complex query into distinct temporal events.

    Semicolons and full-stops act as temporal separators,
    while commas remain as semantic subqueries within an event.

    Args:
        query: The raw input query string.

    Returns:
        A list of cleaned temporal event strings.
    """
    return clean_queries(re.split(r"[.;]+", str(query or "").replace("\n", " ")))


def split_semantic_queries(event: str) -> list[str]:
    """Split a temporal event into semantic subqueries.

    Args:
        event: A single temporal event string.

    Returns:
        A list containing the full event and its comma-separated semantic parts.
    """
    event = str(event or "").strip()
    if not event:
        return []
    return clean_queries([event, *(part.strip() for part in event.split(","))])


@dataclass(frozen=True)
class QueryPlan:
    """Dataclass representing a structured execution plan for search queries."""

    query: str
    mode: SearchMode
    use_split: bool
    events: list[list[str]]

    @property
    def event_queries(self) -> list[str]:
        """Get the primary query string for each temporal event."""
        return [event[0] for event in self.events if event]

    @property
    def flat_queries(self) -> list[str]:
        """Get a flattened list of all semantic subqueries across all events."""
        return clean_queries([query for event in self.events for query in event])


class Orchestrator:
    """Chooses a search mode and dispatches to the right retriever(s).

    This is the only component that knows about every retriever — the
    retrievers themselves do not know about each other or about this class.
    All 4 backends (semantic/temporal/ocr/asr) are plain `SearchPipeline`
    instances with the SAME shape (built via each module's
    `factory.build_*_search_pipeline(...)`, exposing a single `.search(...)`
    entrypoint) — semantic/temporal wrap `ClipFaissIndex` directly, ocr/asr
    wrap their Elasticsearch repository. No separate "model" class on top.
    """

    def __init__(
        self,
        index: "FaissIndex | IndexManager",
        semantic_search: "SemanticSearchPipeline",
        temporal_search: "TemporalSearchPipeline",
        ocr_search_pipeline: "OCRSearchPipeline | None" = None,
        asr_search_pipeline: "ASRSearchPipeline | None" = None,
        llm_query_engine: "LLMQueryEngine | None" = None,
        min_len_for_paraphrase: int = 12,
        max_subqueries: int = 4,
        default_source_weights: dict[str, float] | None = None,
    ) -> None:
        """Compose an orchestrator from an already-built `FaissIndex`/`IndexManager` +
        4 `SearchPipeline` instances (build all via their respective
        `retrieval.index.factory` / `retrieval.retriever.{semantic,temporal,ocr,asr}_search.factory`
        modules — all 4 factories have the same `build_*_search_pipeline(...)` shape).

        Args:
            index: Loaded FAISS index — either a single `FaissIndex` (1
                model) or an `IndexManager` (many models); used to resolve
                FAISS metadata rows when enriching OCR/ASR hits (any one
                model's metadata works, since metadata is model-agnostic —
                the `IndexManager.default_model_key` one is used).
            semantic_search: Semantic `SearchPipeline` built on `index`, used
                for `mode="semantic"`. Called uniformly via `.search(...)`.
            temporal_search: Temporal `SearchPipeline` built on `index`, used
                for `mode="temporal"`. Called uniformly via `.search(...)`.
            ocr_search_pipeline: Optional OCR `SearchPipeline` (Elasticsearch-backed).
                When provided, enables `mode="ocr"`.
            asr_search_pipeline: Optional ASR `SearchPipeline` (Elasticsearch-backed).
                When provided, enables `mode="asr"`.
            llm_query_engine: Optional `LLMQueryEngine` (see
                `src/query_enrichment/llm_query_engine.py`). When `None`
                (default) every mode behaves EXACTLY as before (pure regex
                `split_temporal_events`/`split_semantic_queries`). When set,
                `use_split=True` additionally triggers, per `build_query_plan`:
                  - semantic: LLM paraphrase of short queries into
                    complementary sub-queries (falls back to the regex
                    comma-split on failure or when the query is already
                    long — see `min_len_for_paraphrase`).
                  - temporal/auto: LLM-based scene/event segmentation
                    instead of naive `.`/`;` splitting (falls back to regex
                    on failure), each event then paraphrased as above.
                  - ocr/asr: NEVER used — those two modes always search on
                    the raw query text (see `ocr_search`/`asr_search`).
            min_len_for_paraphrase: Only queries with at most this many
                whitespace-separated words get LLM-paraphrased (a query
                already this long is assumed to be detailed/specific enough
                that generating more sub-queries would dilute it rather than
                help).
            max_subqueries: Upper bound on how many paraphrased sub-queries
                the LLM generates per semantic paraphrase call.
            default_source_weights: Optional default fusion weights used by
                `advanced_search` when `weights` is not supplied for a given
                source, expressed as a rough percentage split, e.g.
                `{"visual": 0.5, "ocr": 0.3, "asr": 0.2}` — "visual" covers
                semantic model(s) AND temporal (both are CLIP-embedding
                based). See `advanced_search`/`_resolve_default_weight`.
        """
        self.index_manager = index if isinstance(index, IndexManager) else None
        # `self.index` always resolves to a concrete FaissIndex, used for
        # metadata lookups (OCR/ASR enrichment) which are model-agnostic.
        self.index: FaissIndex = index.get() if isinstance(index, IndexManager) else index
        self.semantic_search = semantic_search
        self.temporal_search = temporal_search
        self.ocr_search_pipeline = ocr_search_pipeline
        self.asr_search_pipeline = asr_search_pipeline
        self.llm_query_engine = llm_query_engine
        self.min_len_for_paraphrase = max(1, int(min_len_for_paraphrase))
        self.max_subqueries = max(1, int(max_subqueries))
        self.default_source_weights = default_source_weights or {"semantic": 0.8, "ocr": 0.1, "asr": 0.1}

    def _split_events(self, query: str, mode: SearchMode, use_split: bool) -> list[str]:
        """Split `query` into temporal events/scenes.

        Regex `.`/`;` splitting (`split_temporal_events`) is always the
        fallback. When an `llm_query_engine` is wired in, `use_split=True`,
        and `mode` is "temporal" or "auto", the LLM is asked to segment the
        query by MEANING (handles run-on sentences with no punctuation,
        merges clauses that describe one simultaneous scene, drops bare
        event labels/numbering) instead. OCR/ASR never reach this path with
        an LLM engine active (`run_search` calls `ocr_search`/`asr_search`
        directly on `plan.query`, not through event splitting).
        """
        if self.llm_query_engine is not None and use_split and mode in ("temporal", "auto"):
            llm_events = self.llm_query_engine.split_temporal_events(query)
            if llm_events:
                return llm_events
        return split_temporal_events(query)

    def _split_semantic(self, text: str, use_split: bool) -> list[str]:
        """Split semantic sub-queries with the deterministic legacy splitter.

        `use_split` no longer triggers semantic paraphrasing. LLM enrichment
        is reserved for temporal scene/order reasoning and source-focused
        fusion rewrites, both of which are explicitly bounded by the UI
        request and do not expand every ordinary semantic query.
        """
        return split_semantic_queries(text) if use_split else clean_queries([text])

    def build_query_plan(self, query: str, mode: SearchMode = "semantic", use_split: bool = True, reasoning: bool = False) -> QueryPlan:
        """Parse and structure a raw query into a QueryPlan object."""
        query = str(query or "").strip()
        events = []
        for text in (self._split_events(query, mode, use_split) if reasoning else split_temporal_events(query)):
            parts = self._split_semantic(text, use_split)
            if parts:
                events.append(parts)
        return QueryPlan(query=query, mode=mode, use_split=use_split, events=events)

    def _enrich_ocr_hits(self, hits: list[dict], top_k: int) -> pd.DataFrame:
        """Join raw OCR hits with FAISS metadata and normalize scores to `[0, 1]`.

        Hits whose (video_id, keyframe_id) cannot be resolved against the FAISS
        metadata (e.g. stale/partial OCR index) are dropped rather than surfaced
        as broken results.
        """
        if not hits:
            return pd.DataFrame()

        max_score = max(float(hit["score"]) for hit in hits) or 1.0

        rows: list[dict] = []
        for hit in hits:
            meta = self.index.metadata_row_for_ocr_hit(hit["video_id"], hit["keyframe_id"])
            if meta is None:
                continue

            item = {col: meta.get(col) for col in METADATA_DISPLAY_COLUMNS}
            ocr_score = float(hit["score"])
            normalized_score = ocr_score / max_score  # bound to (0, 1] for the frontend

            item.update(
                ocr_score=ocr_score,
                score=normalized_score,
                retrieval_score=normalized_score,
                matched_texts=hit["texts"],
                search_mode="ocr",
            )
            rows.append(item)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame.from_records(rows).sort_values("retrieval_score", ascending=False)
        df = df.head(int(top_k)).reset_index(drop=True)
        df["display_rank"] = np.arange(1, len(df) + 1)
        df["rank"] = df["display_rank"]
        return df

    def ocr_search(self, query: str, top_k: int = 10, oversample_factor: int = 3) -> pd.DataFrame:
        """Execute an OCR (on-screen text) search and return results shaped like semantic search.

        Args:
            query: Free-text query matched against OCR'd on-screen text.
            top_k: Number of results to return after metadata enrichment/dedup.
            oversample_factor: Fetch `top_k * oversample_factor` raw hits from
                Elasticsearch before enrichment, to compensate for hits dropped
                during metadata resolution (e.g. stale OCR entries).

        Returns:
            A DataFrame with the same display columns as semantic search, plus
            `matched_texts` (OCR strings that matched) and `ocr_score` (raw BM25 score).

        Raises:
            RuntimeError: If no OCR backend was injected at construction time.
        """
        if self.ocr_search_pipeline is None:
            raise RuntimeError(
                "OCR search is not available: Orchestrator was built without an "
                "`ocr_search_pipeline`. Build one via "
                "src.retrieval.retriever.ocr_search.factory.build_ocr_search_pipeline(...) and pass it in."
            )

        query = str(query or "").strip()
        if not query:
            return pd.DataFrame()

        raw_top_k = max(int(top_k) * max(1, int(oversample_factor)), int(top_k))
        hits = self.ocr_search_pipeline.search(query=query, top_k=raw_top_k)
        return self._enrich_ocr_hits(hits, top_k)

    def _enrich_asr_hits(
        self, hits: list[dict], top_k: int, max_frames_per_hit: int = 50
    ) -> pd.DataFrame:
        """Join raw ASR hits (speech segments) with FAISS metadata and normalize scores.

        Unlike OCR (one keyframe per hit -> displayed exactly like semantic
        search), each ASR hit is a *spoken segment* that spans many keyframes.
        To make that visually distinct in the frontend (the same way
        `temporal_search` results are), every accepted segment is collapsed
        into a SINGLE result row shaped like a temporal-search hit:
          - a representative frame at the top level (used as the card cover),
          - a `matched_sequence` list containing every keyframe inside the
            segment's `[start_time, end_time]` window.

        The frontend has no ASR-specific code path: `ResultCard`/`GroupedResults`
        already render any row carrying a non-empty `matched_sequence` as a
        horizontal "temporal sequence" strip (see `TemporalSequence.jsx`), so
        populating `matched_sequence` here is what makes ASR hits *look like*
        temporal search, exactly as requested. Each frame's `sub_query` field
        is set to the segment's transcript text so it is shown under the frame.

        `top_k` limits the number of distinct segments (hits) returned, not
        the resulting frame count.
        """
        if not hits:
            return pd.DataFrame()

        max_score = max(float(hit["score"]) for hit in hits) or 1.0

        rows: list[dict] = []
        accepted = 0
        for hit in hits:
            if accepted >= int(top_k):
                break

            frames = self.index.metadata_rows_for_asr_hit(
                hit["video_id"], hit["start_time"], hit["end_time"]
            )
            if not frames:
                continue

            frames = frames[: int(max_frames_per_hit)]
            accepted += 1
            asr_score = float(hit["score"])
            normalized_score = asr_score / max_score  # bound to (0, 1] for the frontend
            transcript_text = str(hit["text"])

            matched_sequence: list[dict] = []
            for seq_idx, meta in enumerate(frames):
                seq_item = {col: meta.get(col) for col in METADATA_DISPLAY_COLUMNS}
                seq_item.update(
                    score=normalized_score,
                    candidate_score=normalized_score,
                    candidate_rank=seq_idx + 1,
                    sub_query_idx=seq_idx,
                    sub_query=transcript_text,
                )
                matched_sequence.append(seq_item)

            # Representative "cover" frame for the row: the middle keyframe of
            # the segment reads better as a thumbnail than the first/last one.
            anchor_meta = frames[len(frames) // 2]
            item = {col: anchor_meta.get(col) for col in METADATA_DISPLAY_COLUMNS}
            item.update(
                asr_score=asr_score,
                score=normalized_score,
                retrieval_score=normalized_score,
                avg_score=normalized_score,
                video_score=normalized_score,
                matched_texts=[transcript_text],
                temporal_start_time=hit["start_time"],
                temporal_end_time=hit["end_time"],
                temporal_duration_sec=max(0.0, float(hit["end_time"]) - float(hit["start_time"])),
                search_mode="asr",
                display_rank=accepted,
                matched_sequence=matched_sequence,
            )
            rows.append(item)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame.from_records(rows).sort_values("display_rank", kind="stable")
        df = df.reset_index(drop=True)
        df["rank"] = df["display_rank"]
        return df

    def asr_search(
        self,
        query: str,
        top_k: int = 10,
        oversample_factor: int = 3,
        max_frames_per_hit: int = 50,
    ) -> pd.DataFrame:
        """Execute an ASR (speech transcript) search and return results shaped like semantic search.

        Args:
            query: Free-text query matched against ASR'd speech transcripts.
            top_k: Number of distinct speech segments (hits) to return after
                metadata enrichment/dedup. Each hit may expand to several rows
                (one per matching keyframe).
            oversample_factor: Fetch `top_k * oversample_factor` raw hits from
                Elasticsearch before enrichment, to compensate for hits dropped
                during metadata resolution (e.g. a segment with no keyframes in range).
            max_frames_per_hit: Safety cap on how many frames a single segment
                can expand to.

        Returns:
            A DataFrame with one row per matched speech segment, shaped like a
            `temporal_search` hit: the same display columns as semantic search,
            plus `matched_texts` (the segment's transcript), `asr_score` (raw
            BM25 score), and `matched_sequence` (every keyframe inside the
            segment's time window, so the frontend renders it as a
            temporal-sequence strip). Row ordering/sequence number comes from
            `display_rank`/`rank`, same as temporal search — no separate
            `segment_id` is needed or used.

        Raises:
            RuntimeError: If no ASR backend was injected at construction time.
        """
        if self.asr_search_pipeline is None:
            raise RuntimeError(
                "ASR search is not available: Orchestrator was built without an "
                "`asr_search_pipeline`. Build one via "
                "src.retrieval.retriever.asr_search.factory.build_asr_search_pipeline(...) and pass it in."
            )

        query = str(query or "").strip()
        if not query:
            return pd.DataFrame()

        raw_top_k = max(int(top_k) * max(1, int(oversample_factor)), int(top_k))
        hits = self.asr_search_pipeline.search(query=query, top_k=raw_top_k)
        return self._enrich_asr_hits(hits, top_k, max_frames_per_hit)

    def run_search(
        self,
        query: str,
        mode: SearchMode = "semantic",
        use_split: bool = True,
        top_k: int = 10,
        candidate_multiplier: int = 5,
        duration_limit: float = -1,
        model_key: str | None = None,
        reasoning: bool = False,
    ) -> tuple[pd.DataFrame, QueryPlan]:
        """Execute a search based on the specified mode and query plan.

        Args:
            model_key: Chỉ áp dụng cho `mode` "semantic"/"temporal"/"auto" —
                chọn model nào để search (khi `self.index_manager` có nhiều
                model). Bỏ trống = model mặc định. Không ảnh hưởng OCR/ASR
                (không dùng embedding model).
        """
        plan = self.build_query_plan(query, mode, use_split, reasoning=reasoning)
        if not plan.events:
            return pd.DataFrame(), plan

        candidate_k = max(int(top_k) * int(candidate_multiplier), int(top_k))

        if mode == "auto":
            effective_mode = "temporal" if len(plan.events) > 1 else "semantic"
        else:
            effective_mode = mode

        if effective_mode == "semantic":
            return self.semantic_search.search(plan.events, top_k, candidate_k, model_key=model_key), plan
        if effective_mode == "temporal":
            return (
                self.temporal_search.search(plan.events, top_k, candidate_k, duration_limit, model_key=model_key),
                plan,
            )
        if effective_mode == "ocr":
            # OCR search operates on the raw query text (Elasticsearch does its own
            # tokenization/fuzzy matching), not the CLIP-oriented temporal/semantic split.
            return self.ocr_search(plan.query, top_k), plan
        if effective_mode == "asr":
            # ASR search operates on the raw query text (Elasticsearch does its own
            # tokenization/fuzzy matching), not the CLIP-oriented temporal/semantic split.
            return self.asr_search(plan.query, top_k), plan

        raise ValueError(f"Unsupported search mode: {effective_mode}")

    def _effective_weight(
        self, key: str, semantic_models: list[str], weights: dict[str, float]
    ) -> float:
        """Resolve the fusion weight for one source `key`.

        An explicit entry in `weights` always wins (unchanged behavior —
        `weights` defaults to `{}`, giving 1.0 for everyone, exactly like
        before this feature existed). When `weights` does NOT specify `key`,
        instead of silently defaulting to 1.0 for every source (which is
        what made every source count equally regardless of how many
        semantic models were ticked), split `self.default_source_weights`
        ("visual"/"ocr"/"asr" percentages) across the active sources: each
        ticked semantic model (and "temporal", which is the same visual
        embedding space) shares the "visual" percentage evenly, while "ocr"
        and "asr" each get their own percentage outright.
        """
        if key in weights:
            return float(weights[key])
        if key not in ("ocr", "asr") and "semantic" in weights:
            return float(weights["semantic"]) / max(1, len(semantic_models))
        defaults = self.default_source_weights
        if key not in ("ocr", "asr"):
            return float(defaults.get("semantic", 1.0))
        if key == "ocr":
            return float(defaults.get("ocr", 1.0))
        if key == "asr":
            return float(defaults.get("asr", 1.0))
        return 1.0

    def _focused_queries_or_empty(self, query: str, targets: list[str]) -> dict[str, str]:
        """LLM-generated per-source query rewrites (see
        `LLMQueryEngine.generate_focused_queries`), or `{}` when no engine
        is wired in / the call fails — callers must treat `{}` as "use the
        original query for every source" (never raises, never blocks
        `advanced_search` from completing).
        """
        if self.llm_query_engine is None or not targets:
            return {}
        return self.llm_query_engine.generate_focused_queries(query, targets)

    def available_semantic_models(self) -> list[str]:
        """Model keys usable for the `advanced_search` semantic checklist —
        only ones actually loaded (and, for text queries, that support
        `encode_text`) show up here."""
        if self.index_manager is None:
            return [self.index.model_key]
        return self.index_manager.text_search_keys()

    def advanced_search(
        self,
        query: str,
        semantic_models: list[str] | None = None,
        temporal: bool = False,
        use_ocr: bool = False,
        use_asr: bool = False,
        top_k: int = 10,
        use_split: bool = True,
        candidate_multiplier: int = 5,
        duration_limit: float = -1,
        weights: dict[str, float] | None = None,
        rrf_k: int = 60,
        raw_query: str | None = None,
        reasoning: bool = False,
    ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        """"Advanced search": chạy đồng thời nhiều method (semantic/temporal/
        ocr/asr), rồi hợp nhất bằng Reciprocal Rank Fusion — tương ứng UI
        checklist kiểu:

            # semantic (mỗi model chạy riêng, KHÔNG kết hợp embedding)
            [x] siglip2-so400m   [ ] vitH-378-quickgelu
            [x] long-clipL       [ ] BEiT-3   [ ] BLIP2
            # temporal
            [x] on/off  (dùng CHUNG danh sách model đã tick ở semantic)
            # ocr
            [x]
            # asr
            [ ]

        Semantic: mỗi model_key trong `semantic_models` ĐƯỢC TÍCH chạy ĐỘC
        LẬP (không kết hợp embedding — mỗi model vẫn search 1 mình trong
        không gian embedding của chính nó), kết quả riêng từng model được
        trả kèm qua `per_source` để debug/hiển thị.

        Temporal: KHÔNG còn checklist model riêng — chỉ 1 công tắc on/off
        (`temporal`) dùng chung đúng tập `semantic_models` đã tick. Mỗi model
        đó build candidate per-event bằng FAISS riêng (thuật toán retrieval
        không đổi), các candidate được fuse bằng RRF theo (video, keyframe,
        sub_query) TRƯỚC, rồi DP alignment chạy ĐÚNG 1 LẦN trên candidate
        pool đã fuse (xem `temporal_search.pipeline.search.SearchPipeline.
        search_combined`) — thay cho thiết kế cũ (mỗi model tick tự chạy
        nguyên 1 lượt DP riêng rồi mới RRF kết quả video ở cuối).

        Toàn bộ nguồn (semantic x N model, temporal đã fuse, ocr, asr) sau đó
        được fuse thêm 1 lần nữa (kết hợp RANK, không kết hợp embedding/score
        thô) thành 1 danh sách cuối cùng.

        Args:
            semantic_models: Danh sách model_key dùng cho CẢ semantic search
                VÀ (nếu `temporal=True`) làm input cho temporal search kết
                hợp (rỗng/None = không chạy semantic, và tắt luôn temporal
                dù `temporal=True`, vì không có model nào để chạy).
            temporal: Bật/tắt temporal search kết hợp trên `semantic_models`.
            use_ocr/use_asr: Bật/tắt OCR/ASR search.
            weights: Optional per-source weight dict, key có thể là
                model_key (áp cho semantic search của model đó), "temporal"
                (áp cho kết quả temporal đã fuse), "ocr", hoặc "asr". Mặc
                định 1.0 cho mọi nguồn.
            rrf_k: Xem `scoring.reciprocal_rank_fusion` — dùng cho cả fuse
                cấp method cuối cùng lẫn fuse candidate cấp model bên trong
                `search_combined` (temporal).

        Returns:
            `(fused_df, per_source)` — `fused_df` là kết quả cuối đã fuse +
            sort; `per_source` là dict `{"semantic:<model_key>": df, ...,
            "temporal": df, "temporal_candidates:<model_key>": df, ...}` cho
            từng nguồn riêng lẻ (để UI có thể hiển thị breakdown nếu muốn) —
            các key `"temporal_candidates:<model_key>"` là breakdown TRƯỚC
            khi fuse, chỉ để debug, không tham gia fuse cuối cùng.
        """
        semantic_models = list(semantic_models or [])
        weights = weights or {}
        # Normalize group weights against sources enabled for this request.
        # Disabled OCR/ASR are forced to zero; this also protects direct
        # callers from sending weights whose sum is not exactly 1.0.
        active_groups = ["semantic"] if semantic_models else []
        if use_ocr and self.ocr_search_pipeline is not None:
            active_groups.append("ocr")
        if use_asr and self.asr_search_pipeline is not None:
            active_groups.append("asr")
        if active_groups:
            raw_group_weights = {
                group: max(0.0, float(weights.get(group, self.default_source_weights.get(group, 0.0))))
                for group in active_groups
            }
            total_group_weight = sum(raw_group_weights.values())
            if total_group_weight > 0:
                weights = {**weights, **{group: value / total_group_weight for group, value in raw_group_weights.items()}}

        candidate_k = max(int(top_k) * int(candidate_multiplier), int(top_k))
        # Semantic/temporal use the translated query when requested by the
        # facade; OCR/ASR use the original-language query supplied as
        # ``raw_query``. This keeps the two language paths independent.
        raw_query = query if raw_query is None else str(raw_query)
        # `mode="temporal"` here (instead of "semantic") so that, when an
        # `llm_query_engine` is wired in and `temporal=True`, event
        # boundaries come from LLM scene segmentation rather than naive
        # `.`/`;` splitting — see `_split_events`. This only changes how
        # `plan.events`/`raw_plan.events` are computed; it does not affect
        # anything else (mode is not otherwise read off `plan`/`raw_plan`
        # below, `effective_mode`/dispatch is decided independently per
        # source further down).
        plan_mode: SearchMode = "temporal" if temporal else "semantic"
        plan = self.build_query_plan(query, mode=plan_mode, use_split=use_split, reasoning=reasoning)
        if temporal and reasoning:
            # The translated and raw-language plans represent the same
            # temporal sequence. Reuse the LLM-derived event boundaries so a
            # Fusion temporal request makes one temporal reasoning call, not
            # one call per language path.
            raw_plan = QueryPlan(
                query=raw_query,
                mode=plan_mode,
                use_split=use_split,
                events=plan.events,
            )
        else:
            raw_plan = self.build_query_plan(raw_query, mode=plan_mode, use_split=use_split, reasoning=reasoning)

        per_source: dict[str, pd.DataFrame] = {}
        ranked_lists: list[pd.DataFrame] = []
        list_weights: list[float] = []

        def _flatten_sequences(df: pd.DataFrame) -> pd.DataFrame:
            """Convert sequence hits to frame hits for non-temporal fusion.

            RRF identity is a keyframe identity.  Keeping ``matched_sequence``
            on the parent row would therefore fuse one whole sequence as one
            item and would hide all frames except its representative frame.
            When temporal mode is off, every source must contribute the same
            frame-shaped rows to RRF.
            """
            if df is None or df.empty or "matched_sequence" not in df.columns:
                return df

            rows: list[dict] = []
            for _, parent in df.iterrows():
                sequence = parent.get("matched_sequence")
                if not isinstance(sequence, (list, tuple)) or not sequence:
                    rows.append(parent.to_dict())
                    continue

                parent_values = parent.to_dict()
                for frame in sequence:
                    if not isinstance(frame, dict):
                        continue
                    item = {**parent_values, **frame}
                    item.pop("matched_sequence", None)
                    rows.append(item)

            if not rows:
                return pd.DataFrame()
            expanded = pd.DataFrame.from_records(rows)
            expanded["rank"] = np.arange(1, len(expanded) + 1)
            expanded["display_rank"] = expanded["rank"]
            return expanded

        def _add(label: str, weight_key: str, df: pd.DataFrame) -> None:
            if df is None or df.empty:
                return
            df = df.copy()
            if not temporal:
                df = _flatten_sequences(df)
                if df.empty:
                    return
            df["search_mode"] = df.get("search_mode", weight_key.split(":")[0])
            per_source[label] = df
            ranked_lists.append(df)
            list_weights.append(self._effective_weight(weight_key, semantic_models, weights))

        def _sequence_to_frames(df: pd.DataFrame) -> pd.DataFrame:
            """Expand sequence hits into frame candidates for temporal fusion."""
            if df is None or df.empty or "matched_sequence" not in df.columns:
                return df.copy() if df is not None else pd.DataFrame()
            rows = []
            for _, parent in df.iterrows():
                sequence = parent.get("matched_sequence")
                if not isinstance(sequence, (list, tuple)) or not sequence:
                    rows.append(parent.to_dict())
                    continue
                base = parent.to_dict()
                for frame in sequence:
                    if isinstance(frame, dict):
                        item = {**base, **frame}
                        item.pop("matched_sequence", None)
                        # Sequence position is not the query-event index.
                        # The parent hit is already being assigned to the
                        # event currently searched.
                        item.pop("sub_query_idx", None)
                        rows.append(item)
            return pd.DataFrame.from_records(rows) if rows else pd.DataFrame()

        def _event_candidates(df: pd.DataFrame, event_idx: int) -> pd.DataFrame:
            """Normalize one source to frame candidates belonging to one event."""
            frames = _sequence_to_frames(df)
            if frames.empty:
                return frames
            if "sub_query_idx" in frames.columns:
                frames = frames[frames["sub_query_idx"].fillna(event_idx).astype(int) == event_idx]
            frames = frames.copy()
            frames["sub_query_idx"] = event_idx
            frames["search_mode"] = frames.get("search_mode", "candidate")
            # RRF must rank individual frame candidates, not the parent ASR
            # segment rank inherited by every flattened frame.
            frames["rank"] = np.arange(1, len(frames) + 1)
            frames["display_rank"] = frames["rank"]
            return frames

        def _weighted_event_fusion(
            semantic_sources: list[tuple[str, pd.DataFrame]],
            auxiliary_sources: list[tuple[str, pd.DataFrame]],
        ) -> pd.DataFrame:
            """RRF semantic models first, then weighted-fuse OCR/ASR."""
            semantic_rrf = reciprocal_rank_fusion(
                [df for _, df in semantic_sources],
                weights=[1.0] * len(semantic_sources),
                rrf_k=rrf_k,
                top_k=candidate_k,
            )
            source_lists: list[tuple[str, pd.DataFrame, float]] = []
            if not semantic_rrf.empty:
                source_lists.append(("semantic", semantic_rrf, self._effective_weight("semantic", semantic_models, weights)))
            for label, df in auxiliary_sources:
                if not df.empty:
                    source_lists.append((label, df, self._effective_weight(label, semantic_models, weights)))
            if not source_lists:
                return pd.DataFrame()

            frames = []
            for label, df, weight in source_lists:
                frame = df.copy()
                score_col = "rrf_score" if "rrf_score" in frame.columns else "score"
                values = pd.to_numeric(frame[score_col], errors="coerce").fillna(0.0)
                maximum = float(values.max()) if len(values) else 0.0
                frame["_fusion_score"] = (values / maximum if maximum > 0 else 0.0) * float(weight)
                frame["_fusion_source"] = label
                frames.append(frame)

            combined = pd.concat(frames, ignore_index=True)
            identity = [c for c in ("dataset", "video_id", "keyframe_id") if c in combined.columns]
            if not identity:
                return pd.DataFrame()
            result = (
                combined.groupby(identity, as_index=False)
                .agg(rrf_score=("_fusion_score", "sum"), matched_sources=("_fusion_source", "nunique"))
                .sort_values("rrf_score", ascending=False)
                .head(candidate_k)
                .reset_index(drop=True)
            )
            best_rows = combined.sort_values("_fusion_score", ascending=False).drop_duplicates(identity)
            metadata = best_rows.drop(columns=[c for c in ("rrf_score", "matched_sources") if c in best_rows.columns])
            result = result.merge(metadata, on=identity, how="left")
            result["rank"] = np.arange(1, len(result) + 1)
            result["display_rank"] = result["rank"]
            result["search_mode"] = "temporal"
            return result

        source_frames_by_event: dict[int, list[tuple[str, pd.DataFrame, float]]] = {}

        # Per-source query rewrites (LLM-generated, best-effort — see
        # `_focused_queries_or_empty`): each semantic model gets a variant
        # emphasizing visual composition, "ocr" gets a variant emphasizing
        # literal on-screen text, "asr" a variant emphasizing spoken
        # content. `{}` (no engine, or LLM call failed) means every source
        # just uses `plan`/`raw_plan` as before this feature existed.
        semantic_targets = [f"semantic:{m}" for m in semantic_models]
        focused_semantic = self._focused_queries_or_empty(plan.query, semantic_targets) if (not temporal and use_split and reasoning) else {}
        focused_flat = (
            self._focused_queries_or_empty(
                raw_plan.query, [t for t, on in (("ocr", use_ocr), ("asr", use_asr)) if on]
            )
            if (not temporal and use_split and reasoning)
            else {}
        )

        # Keep the query plan and the actual LLM rewrites available to the API
        # layer for diagnostics.  These are response metadata, not retrieval
        # inputs, so they must reflect the values actually used by this call.
        debug_weights = {
            key: self._effective_weight(key, semantic_models, weights)
            for key in [*semantic_models, "temporal", "ocr", "asr"]
            if key in semantic_models or (key == "temporal" and temporal)
            or (key == "ocr" and use_ocr) or (key == "asr" and use_asr)
        }
        debug_metadata = {
            "events": plan.events,
            "event_queries": plan.event_queries,
            "focused_queries": {**focused_semantic, **focused_flat},
            "weights": debug_weights,
            "reasoning": bool(reasoning),
        }

        def _plan_for(base_plan: QueryPlan, focused_query: str | None) -> QueryPlan:
            if not focused_query:
                return base_plan
            return self.build_query_plan(focused_query, mode="semantic", use_split=use_split)

        for model_key in semantic_models:
            if temporal:
                for event_idx, event in enumerate(plan.events):
                    # Temporal fusion needs the full candidate pool for DP;
                    # applying final top_k at each event would discard valid
                    # sequence frames before cross-event alignment.
                    df = self.semantic_search.search([event], candidate_k, candidate_k, model_key=model_key)
                    part = _event_candidates(df, event_idx)
                    if not part.empty:
                        per_source[f"semantic:{model_key}:event:{event_idx}"] = part
                        source_frames_by_event.setdefault(event_idx, []).append(
                            (
                                f"semantic:{model_key}",
                                part,
                                self._effective_weight(model_key, semantic_models, weights),
                            )
                        )
            else:
                model_plan = _plan_for(plan, focused_semantic.get(f"semantic:{model_key}"))
                df = self.semantic_search.search(model_plan.events, top_k, candidate_k, model_key=model_key)
                _add(f"semantic:{model_key}", model_key, df)

        if temporal:
            for event_idx, event in enumerate(plan.events):
                raw_event = raw_plan.events[event_idx] if event_idx < len(raw_plan.events) else event
                event_query = raw_event[0] if raw_event else ""
                # Focused rewrite per event (kept small/best-effort: one call
                # per event, cached by the engine on identical text).
                event_focus = self._focused_queries_or_empty(
                    event_query, [t for t, on in (("ocr", use_ocr), ("asr", use_asr)) if on]
                ) if (use_split and reasoning) else {}
                debug_metadata["focused_queries"].update({f"event:{event_idx}:{key}": value for key, value in event_focus.items()})
                if use_ocr and self.ocr_search_pipeline is not None:
                    ocr_query = event_focus.get("ocr", event_query)
                    ocr_df = _event_candidates(self.ocr_search(ocr_query, candidate_k), event_idx)
                    if not ocr_df.empty:
                        per_source[f"ocr:event:{event_idx}"] = ocr_df
                        source_frames_by_event.setdefault(event_idx, []).append(
                            ("ocr", ocr_df, self._effective_weight("ocr", semantic_models, weights))
                        )
                if use_asr and self.asr_search_pipeline is not None:
                    asr_query = event_focus.get("asr", event_query)
                    asr_df = _event_candidates(self.asr_search(asr_query, candidate_k), event_idx)
                    if not asr_df.empty:
                        per_source[f"asr:event:{event_idx}"] = asr_df
                        source_frames_by_event.setdefault(event_idx, []).append(
                            ("asr", asr_df, self._effective_weight("asr", semantic_models, weights))
                        )

            fused_event_frames = []
            for event_idx, sources in source_frames_by_event.items():
                semantic_sources = [(label, df) for label, df, _ in sources if label.startswith("semantic:")]
                auxiliary_sources = [(label, df) for label, df, _ in sources if label in ("ocr", "asr")]
                fused = _weighted_event_fusion(semantic_sources, auxiliary_sources)
                if not fused.empty:
                    fused["sub_query_idx"] = event_idx
                    fused_event_frames.append(fused)

            candidate_pool = pd.concat(fused_event_frames, ignore_index=True) if fused_event_frames else pd.DataFrame()
            if not candidate_pool.empty:
                fused_temporal = temporal_search_from_score_candidates(
                    candidate_pool,
                    plan.event_queries,
                    duration_limit=duration_limit,
                    top_k_videos=top_k,
                    score_col="rrf_score",
                )
                if not fused_temporal.empty:
                    fused_temporal["search_mode"] = "temporal"
                    per_source["temporal"] = fused_temporal
                fused_temporal.attrs.update(debug_metadata)
                return fused_temporal, per_source
            empty = pd.DataFrame()
            empty.attrs.update(debug_metadata)
            return empty, per_source

        if use_ocr and self.ocr_search_pipeline is not None:
            # OCR search trên text tiếng Việt thô -> phải dùng raw_plan.query
            # (chưa dịch), giống hệt nhánh temporal=True ở trên. KHÔNG dùng
            # plan.query (đã dịch cho semantic/temporal). `focused_flat`
            # overrides with an LLM rewrite emphasizing literal on-screen
            # text when available.
            _add("ocr", "ocr", self.ocr_search(focused_flat.get("ocr", raw_plan.query), top_k))

        if use_asr and self.asr_search_pipeline is not None:
            # Tương tự use_ocr ở trên; `focused_flat["asr"]` emphasizes
            # spoken/transcript content when an LLM rewrite is available.
            _add("asr", "asr", self.asr_search(focused_flat.get("asr", raw_plan.query), top_k))

        # `_add` already recorded each source's weight via `_effective_weight`
        # at append time (see below) — `list_weights` stays as collected.
        fused = reciprocal_rank_fusion(ranked_lists, weights=list_weights, rrf_k=rrf_k, top_k=top_k)
        if not temporal and not fused.empty:
            # Non-temporal fusion has one presentation contract: every
            # contribution is a semantic-style frame. Do not let a row that
            # happened to originate from OCR/ASR change the UI card shape.
            frame_only_columns = (
                "matched_sequence",
                "matched_texts",
                "ocr_score",
                "asr_score",
                "temporal_start_time",
                "temporal_end_time",
                "temporal_duration_sec",
                "video_score",
                "avg_score",
                "alignment_score",
                "selected_indices",
            )
            fused = fused.drop(columns=[c for c in frame_only_columns if c in fused.columns])
            fused["search_mode"] = "semantic"
        fused.attrs.update(debug_metadata)
        return fused, per_source
