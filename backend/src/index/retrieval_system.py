from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import faiss
import numpy as np
import pandas as pd
import torch
import open_clip

from PIL import Image
from src.ui import rerank_multi_query
from src.ui.temporal_search import temporal_search_from_candidates


SearchMode = Literal["semantic", "temporal", "ocr", "asr", "auto"]


def resolve_device(device_name: str):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _clean_queries(queries: list[str]) -> list[str]:
    cleaned = []
    seen = set()

    for query in queries:
        query = str(query or "").strip()
        query = re.sub(r"\s+", " ", query)

        if not query:
            continue

        key = query.lower()
        if key in seen:
            continue

        seen.add(key)
        cleaned.append(query)

    return cleaned


def split_temporal_events(query: str) -> list[str]:
    query = str(query or "").replace("\n", " ")
    events = re.split(r"[.;]+", query)
    return _clean_queries(events)


def split_semantic_queries(event: str) -> list[str]:
    event = str(event or "").strip()

    if not event:
        return []

    parts = [event]
    parts.extend(part.strip() for part in event.split(","))

    return _clean_queries(parts)


@dataclass(frozen=True)
class QueryPlan:
    query: str
    mode: SearchMode
    use_split: bool
    events: list[list[str]]

    @property
    def event_queries(self) -> list[str]:
        return [event[0] for event in self.events if event]

    @property
    def flat_queries(self) -> list[str]:
        queries: list[str] = []
        for event in self.events:
            queries.extend(event)
        return _clean_queries(queries)


class FaissRetrievalSystem:
    def __init__(
        self,
        index_path: str | Path,
        metadata_path: str | Path,
        model_name: str,
        pretrained: str,
        device: str = "auto",
        precision: str = "fp32",
        normalize: bool = True,
    ):
        self.index = faiss.read_index(str(index_path))
        self.metadata = pd.read_csv(metadata_path)

        self.device = resolve_device(device)
        self.normalize = normalize

        if self.device.type == "cpu" and precision in {"fp16", "amp"}:
            precision = "fp32"

        self.precision = precision
        self.use_amp = self.device.type == "cuda" and precision == "amp"

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            precision=precision,
            device=self.device,
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    def build_query_plan(
        self,
        query: str,
        mode: SearchMode = "semantic",
        use_split: bool = True,
    ) -> QueryPlan:
        query = str(query or "").strip()
        events_text = split_temporal_events(query)

        events: list[list[str]] = []

        for event_text in events_text:
            if use_split:
                event_queries = split_semantic_queries(event_text)
            else:
                event_queries = _clean_queries([event_text])

            if event_queries:
                events.append(event_queries)

        return QueryPlan(
            query=query,
            mode=mode,
            use_split=use_split,
            events=events,
        )

    @torch.inference_mode()
    def encode_text(self, query: str) -> np.ndarray:
        return self.encode_texts([query])

    @torch.inference_mode()
    def encode_texts(self, queries: list[str]) -> np.ndarray:
        queries = _clean_queries(queries)

        if not queries:
            return np.empty((0, self.index.d), dtype=np.float32)

        tokens = self.tokenizer(queries).to(self.device)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=self.use_amp,
        ):
            emb = self.model.encode_text(tokens)

            if self.normalize:
                emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        return emb.float().cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def encode_image(self, image_path: str | Path) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=self.use_amp,
        ):
            emb = self.model.encode_image(image_tensor)

            if self.normalize:
                emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        return emb.float().cpu().numpy().astype(np.float32)
    
    def _search_embeddings(
        self,
        query_embeddings: np.ndarray,
        queries: list[str],
        top_k: int,
    ) -> list[pd.DataFrame]:
        if query_embeddings.size == 0 or not queries:
            return []

        scores, indices = self.index.search(query_embeddings, int(top_k))

        meta_cols = self.metadata.columns.tolist()
        meta_values = self.metadata.values
        result_dfs: list[pd.DataFrame] = []

        for query_idx, query in enumerate(queries):
            idx_row = indices[query_idx]
            valid_mask = idx_row >= 0
            if not valid_mask.any():
                continue

            valid_positions = np.where(valid_mask)[0]
            valid_indices = idx_row[valid_mask].astype(int)
            valid_scores = scores[query_idx][valid_mask]

            batch = meta_values[valid_indices]
            rows = []
            for i in range(len(valid_indices)):
                item = {meta_cols[c]: batch[i, c] for c in range(len(meta_cols))}
                item["rank"] = int(valid_positions[i]) + 1
                item["score"] = float(valid_scores[i])
                item["query"] = query
                item["sub_query"] = query
                item["sub_query_idx"] = query_idx
                rows.append(item)

            result_dfs.append(pd.DataFrame(rows))

        return result_dfs

    def search(self, query: str, top_k: int = 10) -> pd.DataFrame:
        return self.multi_query_search(
            queries=[query],
            top_k=top_k,
            candidate_k=top_k,
        )

    def similarity_search_by_image(
        self,
        image_path: str | Path,
        top_k: int = 20,
    ) -> pd.DataFrame:
        image_emb = self.encode_image(image_path)

        scores, indices = self.index.search(image_emb, int(top_k))

        rows = []

        for rank, idx in enumerate(indices[0]):
            if idx < 0:
                continue

            item = self.metadata.iloc[int(idx)].to_dict()
            item["rank"] = rank + 1
            item["score"] = float(scores[0][rank])
            item["retrieval_score"] = float(scores[0][rank])
            item["query"] = str(image_path)
            rows.append(item)

        return pd.DataFrame(rows)
    
    def multi_query_search(
        self,
        queries: list[str],
        top_k: int = 10,
        candidate_k: int | None = None,
    ) -> pd.DataFrame:
        queries = _clean_queries(queries)

        if not queries:
            return pd.DataFrame()

        top_k = int(top_k)
        candidate_k = max(int(candidate_k or top_k), top_k)

        query_embeddings = self.encode_texts(queries)
        result_dfs = self._search_embeddings(
            query_embeddings=query_embeddings,
            queries=queries,
            top_k=candidate_k,
        )

        results = rerank_multi_query(result_dfs)

        if results.empty:
            return results

        if "alignment_score" in results.columns and "retrieval_score" not in results.columns:
            results["retrieval_score"] = results["alignment_score"]

        results = results.head(top_k).copy()
        results["display_rank"] = range(1, len(results) + 1)

        return results

    def semantic_search(
        self,
        events: list[list[str]],
        top_k: int = 10,
        candidate_k: int = 500,
    ) -> pd.DataFrame:
        queries: list[str] = []

        for event in events:
            queries.extend(event)

        return self.multi_query_search(
            queries=queries,
            top_k=top_k,
            candidate_k=candidate_k,
        )

    def _build_temporal_candidates(
        self,
        events: list[list[str]],
        candidate_k: int,
    ) -> pd.DataFrame:
        all_candidates: list[pd.DataFrame] = []

        for event_idx, event_queries in enumerate(events):
            event_results = self.multi_query_search(
                queries=event_queries,
                top_k=candidate_k,
                candidate_k=candidate_k,
            )

            if event_results.empty:
                continue

            event_results = event_results.copy()

            score_col = (
                "retrieval_score"
                if "retrieval_score" in event_results.columns
                else "score"
            )

            event_results["candidate_score"] = event_results[score_col].astype(float)
            event_results["candidate_rank"] = range(1, len(event_results) + 1)
            event_results["sub_query_idx"] = event_idx
            event_results["sub_query"] = event_queries[0]

            all_candidates.append(event_results)

        if not all_candidates:
            return pd.DataFrame()

        return pd.concat(all_candidates, ignore_index=True)

    def temporal_search(
        self,
        events: list[list[str]],
        top_k: int = 10,
        candidate_k: int = 500,
        duration_limit: float = -1,
    ) -> pd.DataFrame:
        if not events:
            return pd.DataFrame()

        candidate_k = max(int(candidate_k), int(top_k))

        event_queries = [event[0] for event in events if event]
        query_embeddings = self.encode_texts(event_queries)

        candidate_df = self._build_temporal_candidates(
            events=events,
            candidate_k=candidate_k,
        )

        if candidate_df.empty:
            return pd.DataFrame()

        return temporal_search_from_candidates(
            query_embeddings=query_embeddings,
            sub_queries=event_queries,
            candidate_df=candidate_df,
            duration_limit=duration_limit,
            top_k_videos=top_k,
            max_sequences_per_video=3,
            overlap_threshold=0.6,
        )

    def run_search(
        self,
        query: str,
        mode: SearchMode = "semantic",
        use_split: bool = True,
        top_k: int = 10,
        candidate_multiplier: int = 5,
        duration_limit: float = -1,
    ) -> tuple[pd.DataFrame, QueryPlan]:
        plan = self.build_query_plan(
            query=query,
            mode=mode,
            use_split=use_split,
        )

        if not plan.events:
            return pd.DataFrame(), plan

        candidate_k = max(int(top_k) * int(candidate_multiplier), int(top_k))

        if mode == "auto":
            mode = "temporal" if len(plan.events) > 1 else "semantic"

        if mode == "semantic":
            results = self.semantic_search(
                events=plan.events,
                top_k=top_k,
                candidate_k=candidate_k,
            )
            return results, plan

        if mode == "temporal":
            results = self.temporal_search(
                events=plan.events,
                top_k=top_k,
                candidate_k=candidate_k,
                duration_limit=duration_limit,
            )
            return results, plan

        if mode in {"ocr", "asr"}:
            raise NotImplementedError(f"Search mode '{mode}' is not implemented yet")

        raise ValueError(f"Unsupported search mode: {mode}")