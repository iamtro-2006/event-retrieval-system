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

Design giong het `translation/llm_translator.py` (OpenAI-compatible client,
singleton, LRU cache, threadpool cho batch) de nhat quan style + tai dung
kinh nghiem van hanh (retry/timeout) da co trong repo, nhung TACH THANH
MODULE RIENG (khong sua translation/*) vi day la 1 moi quan tam khac (query
enrichment, khong phai dich ngon ngu).
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
    "You are a temporal-event segmentation engine embedded in a video "
    "search system. The user query describes a sequence of one or more "
    "events/scenes that happen over time in a video (e.g. 'a man opens the "
    "door then a woman walks in and sits down'). Split it into an ORDERED "
    "list of distinct events/scenes.\n"
    "Rules:\n"
    "- Segment by temporal meaning, not punctuation. Every sequentially "
    "different action, state, position, location, or visual configuration "
    "that occurs at a different time is a SEPARATE event, even when the same "
    "subject remains present. A subject changing from one place/state to "
    "another place/state must produce separate ordered scenes, not one event "
    "described with words such as 'then', 'later', 'alternately', or 'and'.\n"
    "- Keep clauses in ONE event only when they describe the same simultaneous "
    "scene or one inseparable action. Do not merge two sequential states just "
    "because there is no explicit verb for the transition.\n"
    "- Drop bare event markers/labels/numbering (e.g. 'Event 1:', 'canh 1', "
    "list bullets) — keep only the descriptive content.\n"
    "- Each output event must be a self-contained English sentence that can be "
    "searched independently: repeat the subject and include the relevant "
    "state/action plus its scene context. Preserve the original temporal "
    "order. Do not use an unresolved reference such as 'then there' or merge "
    "multiple scenes into one sentence.\n"
    "- If the query only describes a single scene, return a single-element "
    "list.\n"
    "Respond ONLY with a JSON array of strings (one per event, in order), "
    "no markdown, no explanation."
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
    """Tiny thread-safe LRU cache, same pattern as `translation/llm_translator.py`."""

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


class LLMQueryEngine:
    """Singleton wrapper around an OpenAI-compatible chat-completions endpoint,
    used purely for query enrichment (paraphrase / temporal split / fusion
    rewrite) — never for translation (see `translation/llm_translator.py`
    for that) and never for OCR/ASR (those stay on raw-text Elasticsearch
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
        from openai import OpenAI  # lazy import, same rationale as llm_translator.py

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
        cache_key = ("temporal", query.casefold(), self._model)
        raw = self._chat(_TEMPORAL_SYSTEM_PROMPT, query, cache_key)
        parsed = _parse_json_array(raw) if raw is not None else None
        return parsed or []

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
