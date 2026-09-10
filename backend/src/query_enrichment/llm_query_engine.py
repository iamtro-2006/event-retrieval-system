"""LLM-backed query enrichment engine.

This module adds an OPTIONAL, purely-additive layer on top of the existing
regex-based `split_query` / `split_temporal_events` / `split_semantic_queries`
functions (see `retrieval/retriever/common/orchestrator.py` and
`retrieval/retriever/common/query_parser.py`). Nothing here is imported or
executed unless a caller explicitly builds a `LLMQueryEngine` and wires it
into the `Orchestrator` (`llm_query_engine=...`) — every other module keeps
working exactly as before when it is `None` (the default).

Scope (theo yeu cau nghiep vu):
- semantic: paraphrase 1 cau query thanh N cau con, moi cau mieu ta 1 khia
  canh (bo cuc chung, bo cuc + local, local, ...), CHI chay khi query du
  "ngan" (duoi `min_len_for_paraphrase` tu) — query dai qua thi coi nhu da
  du chi tiet, khong can paraphrase them.
- ocr/asr: KHONG dung engine nay (khong co method nao duoc goi tu 2 pipeline
  do — xem orchestrator.ocr_search/asr_search, van dung raw query nhu cu).
- temporal/auto: LLM tu tach cau thanh cac "event"/scene (khong dua vao dau
  cau `.`/`;` don thuan), moi event lai duoc paraphrase kieu semantic o tren.
- fusion (advanced_search): sinh ra N cau query, moi cau nham vao 1 nguon
  (semantic/ocr/asr) de tan dung the manh rieng cua tung nguon.

This uses an OpenAI-compatible client dedicated to query enrichment. It is
separate from the Google-only translation subsystem because query enrichment
and language translation are different concerns.
"""

from __future__ import annotations

import json
import os
import re
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

_DEFAULT_BASE_URL = "https://api.xah.io/v1"
_DEFAULT_MODEL = "claude-haiku-4.5"
_TEMPORAL_PROMPT_VERSION = 3

_STRONG_TEMPORAL_BOUNDARY_RE = re.compile(
    r"(?:\b(?:and\s+then|then|after\s+that|before\s+that|followed\s+by|"
    r"subsequently|next|later|finally)\b|\b(?:rồi|sau\s+đó|trước\s+đó|"
    r"tiếp\s+theo|kế\s+tiếp|cuối\s+cùng)\b)",
    flags=re.IGNORECASE,
)

_POSSIBLE_MULTI_EVENT_RE = re.compile(
    r"\b(?:and|then|after|before|followed\s+by|subsequently|next|later|finally|"
    r"và|rồi|sau|trước|tiếp\s+theo|kế\s+tiếp|cuối\s+cùng)\b",
    flags=re.IGNORECASE,
)

# --- Prompts ---------------------------------------------------------------

_SEMANTIC_SYSTEM_PROMPT = (
    "You are an expert visual retrieval query-expansion engine for CLIP-style "
    "video keyframe search. Given ONE user query, produce exactly {n} useful "
    "and materially different English sub-queries. These are separate search "
    "queries, not ordinary linguistic paraphrases.\n"
    "The set MUST cover complementary visual evidence from the SAME scene:\n"
    "1) GLOBAL SPATIAL CONTEXT: describe the whole scene, setting, layout, "
    "subject placement, and the relationship between the main subjects;\n"
    "2) GLOBAL + SALIENT DETAIL: keep the overall scene and add the most "
    "visually distinctive object, action, color, gesture, or interaction;\n"
    "3) LOCAL EVIDENCE: focus on a different concrete object or action while "
    "retaining enough context to locate it in the scene;\n"
    "4+) use other distinct objects, actions, spatial relations, or visual "
    "attributes explicitly implied by the query.\n"
    "Hard requirements:\n"
    "- Do NOT merely change tense, articles, word order, or synonyms. Every "
    "sub-query must add a different retrieval angle or visual emphasis.\n"
    "- Do NOT repeat the original query verbatim or produce near-duplicates.\n"
    "- Do NOT split the query into unrelated fragments; preserve the shared "
    "scene and temporal meaning where relevant.\n"
    "- Do NOT invent people, objects, colors, locations, actions, or details "
    "not supported by the query.\n"
    "- You may make modest, commonsense visual inferences that naturally "
    "enrich the scene and help visual retrieval, as long as they are strongly "
    "supported by the query and remain plausible in its context.\n"
    "- Keep the original evidence as the anchor: never change the main "
    "subjects, objects, actions, relations, event order, or overall scene. "
    "Do not introduce specific details that are merely speculative, "
    "unmotivated, or likely to mislead retrieval; inferred details must stay "
    "secondary to the explicitly stated evidence.\n"
    "- Prefer useful contextual enrichment (scene composition, interaction, "
    "spatial relationship, likely visual focus) over decorative invention. If "
    "an inference is uncertain, keep it broad rather than making it a precise "
    "claim about appearance, material, color, lighting, clothing, location, "
    "or camera view.\n"
    "- If the query contains too few distinct visual details, vary the angle "
    "between setting/composition, subject interaction, and the strongest "
    "available object/action rather than repeating the same sentence.\n"
    "Respond ONLY with a JSON array of {n} strings, no markdown or explanation."
)

_TEMPORAL_SYSTEM_PROMPT = (
    "You split a video-search query into the smallest useful CHRONOLOGICAL "
    "sequence of visually searchable events. Accuracy of event order is the "
    "highest priority.\n"
    "ORDER ALGORITHM:\n"
    "1. Read the whole query and identify only events explicitly stated.\n"
    "2. Build the timeline from explicit relations: X before Y => X,Y; "
    "X after Y => Y,X; X then/followed by Y => X,Y.\n"
    "3. For clauses without an explicit relation, retain their narrated "
    "left-to-right order. Never reorder events using commonsense assumptions.\n"
    "4. Number steps only after the timeline is fixed. Each step number is its "
    "position in the video, not its position in your answer draft.\n"
    "SEGMENTATION RULES:\n"
    "- Split sequential actions, states, positions, locations, or scene "
    "configurations. 'walks in and sits down' is two events.\n"
    "- Keep simultaneous actions joined by while/as/at the same time in one "
    "event. Keep one inseparable action in one event.\n"
    "- Do not invent transitions, causes, objects, people, or intermediate "
    "events. Do not omit or duplicate an explicit event.\n"
    "- Resolve pronouns by repeating the stated subject. Preserve the query's "
    "entities, negation, direction and locations; use its wording where possible.\n"
    "- Each event must be a short, self-contained English visual-search query. "
    "Do not include then/before/after/later, numbering, or multiple sequential "
    "states inside an event. A single scene produces one event.\n"
    "- Return ONE event only when the query contains exactly one visual state "
    "or one inseparable action at one point on the timeline. Two successive "
    "actions by the same subject are still two events, even in the same place.\n"
    "- Set timeline='single' only after checking that no subject changes action, "
    "state, position, or location over time. Otherwise set timeline='multiple'.\n"
    "Example: 'Before sitting down, the man closes the door; after he sits, "
    "a woman enters' => close door (1), man sits (2), woman enters (3).\n"
    "Example: 'A woman reads while a child draws, then both leave' => woman "
    "reads while child draws (1), both leave (2).\n"
    "Return ONLY one compact JSON object with exactly these keys: "
    "{\"timeline\":\"single|multiple\",\"event_count\":2,\"events\":["
    "{\"step\":1,\"event\":\"...\"},{\"step\":2,\"event\":\"...\"}]}. "
    "event_count must equal the events array length; timeline must be 'single' "
    "iff event_count is 1 and 'multiple' iff event_count is greater than 1."
)

_TEMPORAL_REPAIR_SYSTEM_PROMPT = (
    _TEMPORAL_SYSTEM_PROMPT
    + "\nThe previous answer was rejected or suspicious. Re-read the ORIGINAL_QUERY, "
    "rebuild its timeline from scratch, and return only the required JSON object. "
    "Do not copy the previous event count without verifying every stated action/state."
)

_FUSION_SYSTEM_PROMPT = (
    "You are a query-rewriting engine embedded in a multi-modal video "
    "search system that fuses several retrieval sources. Given ONE user "
    "query and a list of target sources, rewrite the query once PER SOURCE "
    "so each rewrite plays to that source's strength:\n"
    "- 'semantic:<model>' targets a CLIP-style visual embedding model — "
    "emphasize concrete visual composition, objects, colors, layout.\n"
    "- 'ocr' targets on-screen text search — emphasize any literal text, "
    "signage, captions, numbers mentioned or implied in the query; if the "
    "query has no on-screen text, keep it short and keyword-like.\n"
    "- 'asr' targets spoken transcript search — emphasize what would "
    "plausibly be SAID out loud in this scene (dialogue/narration topic), "
    "not visual details.\n"
    "- 'temporal' targets the same visual embedding model but for one "
    "specific event within a longer sequence — keep it focused on that "
    "single event only.\n"
    "Respond ONLY with a JSON object mapping each given source key to its "
    "rewritten query string (English), no markdown, no explanation. Include "
    "every requested key exactly once."
)


class _LRUCache:
    """Tiny thread-safe LRU cache for query-enrichment responses."""

    def __init__(self, maxsize: int = 2048):
        self._maxsize = maxsize
        self._data: "OrderedDict[tuple, Any]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def set(self, key, value):
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            if len(self._data) > self._maxsize:
                self._data.popitem(last=False)


def _parse_json_array(raw: str) -> list[str] | None:
    raw = (raw or "").strip()
    # Some models wrap output in ```json fences despite instructions.
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, list):
        return None
    return [str(item).strip() for item in data if str(item or "").strip()]


def _parse_json_object(raw: str) -> dict[str, str] | None:
    raw = (raw or "").strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    return {str(k): str(v).strip() for k, v in data.items() if str(v or "").strip()}


def _query_requires_multiple_events(query: str) -> bool:
    """Return whether the source contains an unambiguous timeline boundary."""
    query = str(query or "")
    delimited_parts = [part.strip() for part in re.split(r"[.;\n]+", query) if part.strip()]
    return len(delimited_parts) > 1 or bool(_STRONG_TEMPORAL_BOUNDARY_RE.search(query))


def _query_may_contain_multiple_events(query: str) -> bool:
    """Broader signal used to request a second opinion on a one-event answer."""
    query = str(query or "")
    return _query_requires_multiple_events(query) or bool(_POSSIBLE_MULTI_EVENT_RE.search(query))


def _parse_temporal_events(raw: str, query: str = "") -> list[str] | None:
    """Strictly validate the versioned temporal JSON contract."""
    raw = (raw or "").strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict) or set(data) != {"timeline", "event_count", "events"}:
        return None

    timeline = data.get("timeline")
    event_count = data.get("event_count")
    items = data.get("events")
    if timeline not in ("single", "multiple"):
        return None
    if isinstance(event_count, bool) or not isinstance(event_count, int) or event_count < 1:
        return None
    if not isinstance(items, list) or not items or event_count != len(items):
        return None
    if (timeline == "single") != (event_count == 1):
        return None

    parsed: list[tuple[int, str]] = []
    for item in items:
        if not isinstance(item, dict) or set(item) != {"step", "event"}:
            return None
        step = item.get("step")
        event_value = item.get("event")
        if isinstance(step, bool) or not isinstance(step, int):
            return None
        if not isinstance(event_value, str):
            return None
        event = event_value.strip()
        if step < 1 or not event:
            return None
        parsed.append((step, event))

    steps = [step for step, _ in parsed]
    if len(set(steps)) != len(steps) or sorted(steps) != list(range(1, len(steps) + 1)):
        return None
    parsed.sort(key=lambda item: item[0])
    events = [event for _, event in parsed]
    normalized = [re.sub(r"\s+", " ", event).strip().casefold() for event in events]
    if len(set(normalized)) != len(normalized):
        return None
    if event_count == 1 and _query_requires_multiple_events(query):
        return None
    if any(_STRONG_TEMPORAL_BOUNDARY_RE.search(event) for event in events):
        return None
    return events


class LLMQueryEngine:
    """Singleton wrapper around an OpenAI-compatible chat-completions endpoint,
    used purely for query enrichment (paraphrase / temporal split / fusion
    rewrite) — never for translation and never for OCR/ASR (those stay on raw-text Elasticsearch
    matching, see module docstring).
    """

    _instance: Optional["LLMQueryEngine"] = None
    _lock = threading.Lock()

    def __init__(
        self,
        api_key: str,
        base_url: str = _DEFAULT_BASE_URL,
        model: str = _DEFAULT_MODEL,
        timeout: float = 15.0,
        max_retries: int = 2,
        max_workers: int = 8,
        temperature: float = 0.3,
        max_output_tokens: int = 512,
        cache_size: int = 2048,
    ):
        if not api_key:
            raise ValueError(
                "LLMQueryEngine requires an API key. Set it via "
                "configs/app.yaml -> query_enrichment.llm.api_key or the "
                "QUERY_ENRICHMENT_API_KEY env var."
            )
        from openai import OpenAI  # lazy import: only needed when enrichment is enabled

        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries)
        self._model = model
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._cache = _LRUCache(maxsize=cache_size)
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="llm-query-enrich")

    @classmethod
    def get_instance(cls, **kwargs) -> "LLMQueryEngine":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(**kwargs)
        return cls._instance

    def _chat(self, system_prompt: str, user_text: str, cache_key: tuple) -> str | None:
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            print(f"[query_enrichment] LLM request start: {cache_key[0]}", flush=True)
            response = self._client.chat.completions.create(
                model=self._model,
                temperature=self._temperature,
                max_tokens=self._max_output_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ],
            )
            out = (response.choices[0].message.content or "").strip()
            print(f"[query_enrichment] LLM request success: {cache_key[0]}", flush=True)
        except Exception as exc:
            print(f"[query_enrichment] LLM request failed: {cache_key[0]}: {type(exc).__name__}: {exc}", flush=True)
            return None
        self._cache.set(cache_key, out)
        return out

    # -- semantic paraphrase -------------------------------------------------

    def paraphrase_semantic_query(self, query: str, max_subqueries: int = 4) -> list[str]:
        """Paraphrase `query` into up to `max_subqueries` complementary
        sub-queries (global composition, global+local, local, ...). Returns
        `[]` on any failure/timeout so the caller can fall back to the
        regex-based splitter — never raises.
        """
        query = str(query or "").strip()
        if not query or max_subqueries < 1:
            return []
        n = max(1, int(max_subqueries))
        cache_key = ("semantic", query.casefold(), n, self._model)
        raw = self._chat(_SEMANTIC_SYSTEM_PROMPT.format(n=n), query, cache_key)
        parsed = _parse_json_array(raw) if raw is not None else None
        if not parsed:
            return []
        return parsed[:n]

    # -- temporal event splitting --------------------------------------------

    def split_temporal_events(self, query: str) -> list[str]:
        """LLM-based replacement for the punctuation-only
        `orchestrator.split_temporal_events`. Returns `[]` on failure so the
        caller can fall back to the regex splitter — never raises.
        """
        query = str(query or "").strip()
        if not query:
            return []
        cache_key = ("temporal", _TEMPORAL_PROMPT_VERSION, query.casefold(), self._model)
        raw = self._chat(_TEMPORAL_SYSTEM_PROMPT, query, cache_key)
        parsed = _parse_temporal_events(raw, query) if raw is not None else None

        # A one-event answer containing conjunctions or temporal language is
        # syntactically valid but semantically risky. Ask once more rather than
        # silently collapsing a multi-stage query into one temporal event.
        needs_confirmation = bool(
            parsed
            and len(parsed) == 1
            and _query_may_contain_multiple_events(query)
        )
        if parsed and not needs_confirmation:
            return parsed

        repair_input = json.dumps(
            {
                "ORIGINAL_QUERY": query,
                "REJECTED_OR_SUSPICIOUS_RESPONSE": raw or "",
            },
            ensure_ascii=False,
        )
        repair_key = (
            "temporal-repair",
            _TEMPORAL_PROMPT_VERSION,
            query.casefold(),
            self._model,
        )
        repaired_raw = self._chat(
            _TEMPORAL_REPAIR_SYSTEM_PROMPT, repair_input, repair_key
        )
        repaired = (
            _parse_temporal_events(repaired_raw, query)
            if repaired_raw is not None
            else None
        )
        return repaired or []

    # -- fusion-focused rewrite -----------------------------------------------

    def generate_focused_queries(self, query: str, targets: list[str]) -> dict[str, str]:
        """Rewrite `query` once per entry in `targets` (e.g.
        `["semantic:vitH-378-quickgelu", "ocr", "asr", "temporal"]`), each
        rewrite emphasizing that source's strength. Returns `{}` on failure
        so the caller falls back to using the same raw query for every
        source — never raises.
        """
        query = str(query or "").strip()
        targets = [str(t).strip() for t in targets if str(t or "").strip()]
        if not query or not targets:
            return {}
        cache_key = ("fusion", query.casefold(), tuple(sorted(targets)), self._model)
        user_text = json.dumps({"query": query, "sources": targets}, ensure_ascii=False)
        raw = self._chat(_FUSION_SYSTEM_PROMPT, user_text, cache_key)
        parsed = _parse_json_object(raw) if raw is not None else None
        if not parsed:
            return {}
        return {t: parsed[t] for t in targets if t in parsed}


def build_query_engine_or_none(config: dict[str, Any] | None) -> "LLMQueryEngine | None":
    """Build a singleton `LLMQueryEngine` from the `query_enrichment.llm`
    section of the app config, or `None` if disabled/misconfigured.

    Mirrors `system.py::_build_translator_or_none` — an infra failure here
    (missing key, no network) must NOT crash the whole app, it just disables
    the enrichment feature and every mode falls back to its previous
    (regex-based) behavior.
    """
    config = config or {}
    if not config.get("enabled", False):
        return None
    llm_cfg = config.get("llm") or {}
    try:
        return LLMQueryEngine.get_instance(
            api_key=llm_cfg.get("api_key") or os.getenv("QUERY_ENRICHMENT_API_KEY", ""),
            base_url=llm_cfg.get("base_url", _DEFAULT_BASE_URL),
            model=llm_cfg.get("model", _DEFAULT_MODEL),
            timeout=float(llm_cfg.get("timeout", 15.0)),
            max_retries=int(llm_cfg.get("max_retries", 2)),
            max_workers=int(llm_cfg.get("max_workers", 8)),
            temperature=float(llm_cfg.get("temperature", 0.3)),
            max_output_tokens=int(llm_cfg.get("max_output_tokens", 512)),
            cache_size=int(llm_cfg.get("cache_size", 2048)),
        )
    except Exception as exc:  # pragma: no cover - infra failure (no key/network)
        print(f"[query_enrichment] LLMQueryEngine init failed, disabling query enrichment: {type(exc).__name__}: {exc}")
        return None
