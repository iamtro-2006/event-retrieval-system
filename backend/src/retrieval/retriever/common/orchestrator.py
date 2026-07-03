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

from src.retrieval.index.clip_faiss_index import METADATA_DISPLAY_COLUMNS, ClipFaissIndex
from src.retrieval.retriever.semantic_search.pipeline.search import clean_queries

if TYPE_CHECKING:
    from src.retrieval.retriever.semantic_search.pipeline.search import SearchPipeline as SemanticSearchPipeline
    from src.retrieval.retriever.temporal_search.pipeline.search import SearchPipeline as TemporalSearchPipeline
    from src.retrieval.retriever.ocr_search.pipeline.search import SearchPipeline as OCRSearchPipeline
    from src.retrieval.retriever.asr_search.pipeline.search import SearchPipeline as ASRSearchPipeline

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
        index: ClipFaissIndex,
        semantic_search: "SemanticSearchPipeline",
        temporal_search: "TemporalSearchPipeline",
        ocr_search_pipeline: "OCRSearchPipeline | None" = None,
        asr_search_pipeline: "ASRSearchPipeline | None" = None,
    ) -> None:
        """Compose an orchestrator from an already-built `ClipFaissIndex` +
        4 `SearchPipeline` instances (build all via their respective
        `retrieval.index.factory` / `retrieval.retriever.{semantic,temporal,ocr,asr}_search.factory`
        modules — all 4 factories have the same `build_*_search_pipeline(...)` shape).

        Args:
            index: Loaded FAISS + OpenCLIP index, used to resolve FAISS
                metadata rows when enriching OCR/ASR hits.
            semantic_search: Semantic `SearchPipeline` built on `index`, used
                for `mode="semantic"`. Called uniformly via `.search(...)`.
            temporal_search: Temporal `SearchPipeline` built on `index`, used
                for `mode="temporal"`. Called uniformly via `.search(...)`.
            ocr_search_pipeline: Optional OCR `SearchPipeline` (Elasticsearch-backed).
                When provided, enables `mode="ocr"`.
            asr_search_pipeline: Optional ASR `SearchPipeline` (Elasticsearch-backed).
                When provided, enables `mode="asr"`.
        """
        self.index = index
        self.semantic_search = semantic_search
        self.temporal_search = temporal_search
        self.ocr_search_pipeline = ocr_search_pipeline
        self.asr_search_pipeline = asr_search_pipeline

    def build_query_plan(self, query: str, mode: SearchMode = "semantic", use_split: bool = True) -> QueryPlan:
        """Parse and structure a raw query into a QueryPlan object."""
        query = str(query or "").strip()
        events = []
        for text in split_temporal_events(query):
            parts = split_semantic_queries(text) if use_split else clean_queries([text])
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
                segment_id=hit["segment_id"],
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
            BM25 score), `segment_id`, and `matched_sequence` (every keyframe
            inside the segment's time window, so the frontend renders it as a
            temporal-sequence strip).

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
    ) -> tuple[pd.DataFrame, QueryPlan]:
        """Execute a search based on the specified mode and query plan."""
        plan = self.build_query_plan(query, mode, use_split)
        if not plan.events:
            return pd.DataFrame(), plan

        candidate_k = max(int(top_k) * int(candidate_multiplier), int(top_k))

        if mode == "auto":
            effective_mode = "temporal" if len(plan.events) > 1 else "semantic"
        else:
            effective_mode = mode

        if effective_mode == "semantic":
            return self.semantic_search.search(plan.events, top_k, candidate_k), plan
        if effective_mode == "temporal":
            return self.temporal_search.search(plan.events, top_k, candidate_k, duration_limit), plan
        if effective_mode == "ocr":
            # OCR search operates on the raw query text (Elasticsearch does its own
            # tokenization/fuzzy matching), not the CLIP-oriented temporal/semantic split.
            return self.ocr_search(plan.query, top_k), plan
        if effective_mode == "asr":
            # ASR search operates on the raw query text (Elasticsearch does its own
            # tokenization/fuzzy matching), not the CLIP-oriented temporal/semantic split.
            return self.asr_search(plan.query, top_k), plan

        raise ValueError(f"Unsupported search mode: {effective_mode}")
