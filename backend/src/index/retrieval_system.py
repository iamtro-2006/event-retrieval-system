from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
import open_clip

from src.ui.temporal_search import temporal_search_from_candidates


def resolve_device(device_name: str):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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

        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            precision=precision,
            device=self.device,
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    @torch.inference_mode()
    def encode_text(self, query: str) -> np.ndarray:
        tokens = self.tokenizer([query]).to(self.device)

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
    def encode_texts(self, queries: list[str]) -> np.ndarray:
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

    def search(self, query: str, top_k: int = 10) -> pd.DataFrame:
        query_emb = self.encode_text(query)

        scores, indices = self.index.search(query_emb, top_k)

        rows = []

        for rank, idx in enumerate(indices[0]):
            if idx < 0:
                continue

            item = self.metadata.iloc[int(idx)].to_dict()
            item["rank"] = rank + 1
            item["score"] = float(scores[0][rank])
            item["query"] = query
            rows.append(item)

        return pd.DataFrame(rows)

    def temporal_search(
        self,
        sub_queries: list[str],
        top_k: int = 10,
        candidate_k: int = 500,
        duration_limit: float = -1,
    ) -> pd.DataFrame:
        if not sub_queries:
            return pd.DataFrame()

        candidate_k = max(int(candidate_k), top_k)

        query_embeddings = self.encode_texts(sub_queries)

        all_candidates = []

        for sub_query_idx, sub_query in enumerate(sub_queries):
            query_emb = query_embeddings[sub_query_idx : sub_query_idx + 1]
            scores, indices = self.index.search(query_emb, candidate_k)

            for rank, idx in enumerate(indices[0]):
                if idx < 0:
                    continue

                item = self.metadata.iloc[int(idx)].to_dict()
                item["candidate_score"] = float(scores[0][rank])
                item["candidate_rank"] = rank + 1
                item["sub_query_idx"] = sub_query_idx
                item["sub_query"] = sub_query
                all_candidates.append(item)

        if not all_candidates:
            return pd.DataFrame()

        candidate_df = pd.DataFrame(all_candidates)

        return temporal_search_from_candidates(
            query_embeddings=query_embeddings,
            sub_queries=sub_queries,
            candidate_df=candidate_df,
            duration_limit=duration_limit,
            top_k_videos=top_k,
            max_sequences_per_video=3,
            overlap_threshold=0.6
        )