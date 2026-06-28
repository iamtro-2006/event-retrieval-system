from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image
from pymilvus import MilvusClient

from src.index.milvus_schema import PK_FIELD, SCALAR_FIELD_NAMES, VECTOR_FIELD
from src.index.query_planning import (
    QueryPlan,
    SearchMode,
    aggregate_multi_query,
    build_query_plan,
    clean_queries,
    resolve_effective_mode,
)
from src.index.temporal import (
    Candidates,
    matches_to_dataframe,
    search as temporal_search,
)
from src.utils.device import resolve_device


ALL_OUTPUT_FIELDS = [PK_FIELD, *SCALAR_FIELD_NAMES, VECTOR_FIELD]
SCALAR_OUTPUT_FIELDS = [PK_FIELD, *SCALAR_FIELD_NAMES]


def _json_safe_scalar(val: Any) -> Any:
    if val is None:
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    return val


def search_result_to_dict(result: dict) -> dict[str, Any]:
    """Convert a Milvus search hit dict into a flat metadata record.

    pymilvus 3.0 nests scalar fields under ``entity`` and puts the PK
    in ``id`` and the score in ``distance`` at the top level.
    """
    entity = result.get("entity", result)

    pk = entity.get(PK_FIELD, result.get("id", result.get(PK_FIELD, "")))
    record: dict[str, Any] = {}
    record[PK_FIELD] = pk
    record["_pk"] = pk

    for name in SCALAR_FIELD_NAMES:
        record[name] = _json_safe_scalar(entity.get(name, result.get(name)))

    if "distance" in result:
        record["_distance"] = float(result["distance"])
    elif "score" in result:
        record["_distance"] = float(result["score"])
    elif "distance" in entity:
        record["_distance"] = float(entity["distance"])

    vec = entity.get(VECTOR_FIELD, result.get(VECTOR_FIELD))
    if vec is not None:
        record["_vector"] = np.asarray(vec, dtype=np.float32)
    return record


class MilvusRetrievalSystem:
    """Retrieval backend backed by Milvus Standalone.

    The CLIP model stays in-process for query/image encoding.
    All vector storage, scalar filtering, and ANN search is delegated to Milvus.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        collection_name: str = "keyframes",
        model_name: str = "ViT-L-16-SigLIP-256",
        pretrained: str = "webli",
        device: str = "auto",
        precision: str = "fp32",
        normalize: bool = True,
        consistency_level: str = "Bounded",
        default_ef: int = 64,
        metric_type: str = "IP",
        compile_model: bool = False,
    ) -> None:
        self.collection_name = collection_name
        self.consistency_level = consistency_level
        self.default_ef = int(default_ef)
        self.metric_type = metric_type

        self.uri = f"http://{host}:{port}"
        self.client = MilvusClient(uri=self.uri)

        if not self.client.has_collection(collection_name):
            raise RuntimeError(
                f"Milvus collection '{collection_name}' does not exist. "
                "Run build_milvus.py first."
            )

        self.client.load_collection(collection_name)

        desc = self.client.describe_collection(collection_name)
        fields_info = {f["name"]: f for f in desc.get("fields", [])}
        vec_field_info = fields_info.get(VECTOR_FIELD, {})
        self._dim = int(vec_field_info.get("params", {}).get("dim", 0))
        if self._dim == 0:
            raise RuntimeError(
                f"Cannot detect vector dimension for collection '{collection_name}'"
            )

        stats = self.client.get_collection_stats(collection_name)
        self._num_entities = int(stats.get("row_count", 0))

        self.device = resolve_device(device)
        self.normalize = bool(normalize)
        if self.device.type == "cpu" and precision in {"fp16", "amp", "bf16"}:
            precision = "fp32"
        self.precision = precision
        self.autocast_dtype = (
            torch.float16 if precision in {"fp16", "amp"} else torch.bfloat16
        )
        self.use_autocast = self.device.type == "cuda" and precision in {
            "amp",
            "fp16",
            "bf16",
        }

        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            precision=precision,
            device=self.device,
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        if compile_model and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(
                    self.model, mode="reduce-overhead", fullgraph=False
                )
            except Exception as exc:
                print(f"[MODEL] torch.compile skipped: {type(exc).__name__}: {exc}")

        print(
            f"[MILVUS] connected to {self.uri} | collection={collection_name} | "
            f"dim={self._dim} | entities={self._num_entities} | "
            f"device={self.device} | precision={self.precision}"
        )

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def num_entities(self) -> int:
        return self._num_entities

    @property
    def cache_info(self) -> dict[str, Any]:
        return {
            "mode": "milvus",
            "collection_name": self.collection_name,
            "uri": self.uri,
            "num_entities": self._num_entities,
            "dim": self._dim,
            "consistency_level": self.consistency_level,
            "metric_type": self.metric_type,
            "default_ef": self.default_ef,
        }

    def _make_search_params(self, ef: int | None = None) -> dict[str, Any]:
        return {
            "metric_type": self.metric_type,
            "params": {"ef": int(ef or self.default_ef)},
        }

    def close(self) -> None:
        self.client.close()

    def build_query_plan(
        self,
        query: str,
        mode: SearchMode = "semantic",
        use_split: bool = True,
    ) -> QueryPlan:
        return build_query_plan(query, mode, use_split)

    @torch.inference_mode()
    def encode_text(self, query: str) -> np.ndarray:
        return self.encode_texts([query])

    @torch.inference_mode()
    def encode_texts(self, queries: list[str]) -> np.ndarray:
        queries = clean_queries(queries)
        if not queries:
            return np.empty((0, self._dim), dtype=np.float32)
        tokens = self.tokenizer(queries).to(self.device, non_blocking=True)
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.autocast_dtype,
            enabled=self.use_autocast,
        ):
            emb = self.model.encode_text(tokens)
            if self.normalize:
                emb = torch.nn.functional.normalize(emb, dim=-1)
        return np.ascontiguousarray(emb.float().cpu().numpy(), dtype=np.float32)

    @torch.inference_mode()
    def encode_image(self, image_path: str | Path) -> np.ndarray:
        with Image.open(image_path) as image:
            tensor = self.preprocess(image.convert("RGB")).unsqueeze(0)
        tensor = tensor.to(self.device, non_blocking=True)
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.autocast_dtype,
            enabled=self.use_autocast,
        ):
            emb = self.model.encode_image(tensor)
            if self.normalize:
                emb = torch.nn.functional.normalize(emb, dim=-1)
        return np.ascontiguousarray(emb.float().cpu().numpy(), dtype=np.float32)

    def _milvus_search(
        self,
        query_vectors: np.ndarray,
        limit: int,
        ef: int | None = None,
        output_fields: list[str] | None = None,
        filter_expr: str = "",
    ) -> list[list[dict]]:
        if query_vectors.size == 0:
            return []
        vectors_list = [v.tolist() for v in np.asarray(query_vectors, dtype=np.float32)]
        search_params = self._make_search_params(ef)
        return self.client.search(
            collection_name=self.collection_name,
            data=vectors_list,
            filter=filter_expr,
            limit=int(limit),
            output_fields=output_fields or SCALAR_OUTPUT_FIELDS,
            search_params=search_params,
        )

    def _milvus_query(
        self,
        filter_expr: str,
        output_fields: list[str] | None = None,
        limit: int = 10,
    ) -> list[dict]:
        return self.client.query(
            collection_name=self.collection_name,
            filter=filter_expr,
            output_fields=output_fields or SCALAR_OUTPUT_FIELDS,
            limit=int(limit),
        )

    def multi_query_search(
        self,
        queries: list[str],
        top_k: int = 10,
        candidate_k: int | None = None,
        query_embeddings: np.ndarray | None = None,
        ef: int | None = None,
    ) -> pd.DataFrame:
        queries = clean_queries(queries)
        if not queries:
            return pd.DataFrame()
        candidate_k = max(int(candidate_k or top_k), int(top_k))
        embeddings = (
            query_embeddings
            if query_embeddings is not None
            else self.encode_texts(queries)
        )

        results = self._milvus_search(
            query_vectors=embeddings,
            limit=candidate_k,
            ef=ef,
            output_fields=SCALAR_OUTPUT_FIELDS,
        )

        n_queries = len(queries)
        max_len = max((len(r) for r in results), default=0)
        if max_len == 0:
            return pd.DataFrame()

        scores = np.full((n_queries, max_len), -1.0, dtype=np.float32)
        indices = np.full((n_queries, max_len), -1, dtype=np.int64)
        metadata_records: list[dict[str, Any]] = []

        for qi, hits in enumerate(results):
            for hi, hit in enumerate(hits):
                rec = search_result_to_dict(hit)
                row_idx = len(metadata_records)
                metadata_records.append(rec)
                indices[qi, hi] = row_idx
                scores[qi, hi] = rec.get("_distance", 0.0)

        return aggregate_multi_query(
            scores=scores,
            indices=indices,
            queries=queries,
            top_k=int(top_k),
            metadata_records=metadata_records,
        )

    def search(self, query: str, top_k: int = 10) -> pd.DataFrame:
        return self.multi_query_search([query], top_k, top_k)

    def semantic_search(
        self,
        events: list[list[str]],
        top_k: int = 10,
        candidate_k: int = 500,
        ef: int | None = None,
    ) -> pd.DataFrame:
        queries = clean_queries([query for event in events for query in event])
        embeddings = self.encode_texts(queries)
        return self.multi_query_search(queries, top_k, candidate_k, embeddings, ef=ef)

    def _build_temporal_candidates(
        self,
        events: list[list[str]],
        all_queries: list[str],
        all_embeddings: np.ndarray,
        candidate_k: int,
        ef: int | None = None,
    ) -> pd.DataFrame:
        results = self._milvus_search(
            query_vectors=all_embeddings,
            limit=candidate_k,
            ef=ef,
            output_fields=ALL_OUTPUT_FIELDS,
        )

        frames: list[pd.DataFrame] = []
        offset = 0
        for event_idx, event_queries in enumerate(events):
            count = len(event_queries)
            offset += count
            event_hits = (
                results[offset - count : offset] if len(results) >= offset else []
            )

            if not event_hits:
                continue

            n_queries = len(event_queries)
            max_len = max((len(r) for r in event_hits), default=0)
            if max_len == 0:
                continue

            scores = np.full((n_queries, max_len), -1.0, dtype=np.float32)
            indices = np.full((n_queries, max_len), -1, dtype=np.int64)
            metadata_records: list[dict[str, Any]] = []

            for qi, hits in enumerate(event_hits):
                for hi, hit in enumerate(hits):
                    rec = search_result_to_dict(hit)
                    row_idx = len(metadata_records)
                    metadata_records.append(rec)
                    indices[qi, hi] = row_idx
                    scores[qi, hi] = rec.get("_distance", 0.0)

            event_results = aggregate_multi_query(
                scores=scores,
                indices=indices,
                queries=event_queries,
                top_k=candidate_k,
                metadata_records=metadata_records,
            )
            if event_results.empty:
                continue

            event_results = event_results.copy()
            event_results["candidate_score"] = event_results[
                "retrieval_score"
            ].to_numpy(np.float32)
            event_results["candidate_rank"] = np.arange(1, len(event_results) + 1)
            event_results["sub_query_idx"] = event_idx
            event_results["sub_query"] = event_queries[0]
            frames.append(event_results)

        return (
            pd.concat(frames, ignore_index=True, copy=False)
            if frames
            else pd.DataFrame()
        )

    def temporal_search(
        self,
        events: list[list[str]],
        top_k: int = 10,
        candidate_k: int = 500,
        duration_limit: float = -1,
        ef: int | None = None,
    ) -> pd.DataFrame:
        if not events:
            return pd.DataFrame()
        candidate_k = max(int(candidate_k), int(top_k))
        all_queries = [query for event in events for query in event]
        all_embeddings = self.encode_texts(all_queries)

        offsets = np.cumsum([0] + [len(event) for event in events[:-1]])
        event_embeddings = all_embeddings[offsets]
        event_queries = [event[0] for event in events]

        candidate_df = self._build_temporal_candidates(
            events, all_queries, all_embeddings, candidate_k, ef=ef
        )
        if candidate_df.empty:
            return pd.DataFrame()

        candidates = self._build_temporal_candidates_object(candidate_df)
        matches = temporal_search(
            candidates=candidates,
            query_embeddings=event_embeddings,
            sub_queries=event_queries,
            duration_limit=duration_limit,
            top_k_videos=top_k,
        )
        return matches_to_dataframe(matches)

    def _build_temporal_candidates_object(
        self, candidate_df: pd.DataFrame
    ) -> Candidates:
        """Assemble a Candidates value object from the candidate DataFrame.

        Uses positional row indices into the per-call embedding matrix built
        from the ``_vector`` column returned by Milvus.
        """
        vectors = (
            candidate_df["_vector"].tolist()
            if "_vector" in candidate_df.columns
            else []
        )
        if not vectors:
            raise RuntimeError(
                "Temporal search requires vectors in the candidate results. "
                "Ensure ALL_OUTPUT_FIELDS is used when searching."
            )
        matrix = np.stack([np.asarray(v, dtype=np.float32) for v in vectors])
        return Candidates(
            video_id=candidate_df["video_id"].to_numpy(),
            timestamp_sec=pd.to_numeric(candidate_df["timestamp_sec"], errors="coerce")
            .fillna(0)
            .to_numpy(np.float32),
            row_index=np.arange(len(candidate_df), dtype=np.int64),
            records=candidate_df.to_dict(orient="records"),
            embedding_matrix=matrix,
        )

    def similarity_search_by_image(
        self,
        image_path: str | Path,
        top_k: int = 20,
        ef: int | None = None,
    ) -> pd.DataFrame:
        query_vec = self.encode_image(image_path)
        results = self._milvus_search(
            query_vectors=query_vec,
            limit=top_k,
            ef=ef,
            output_fields=SCALAR_OUTPUT_FIELDS,
        )
        if not results or not results[0]:
            return pd.DataFrame()

        rows: list[dict[str, Any]] = []
        for rank, hit in enumerate(results[0], 1):
            rec = search_result_to_dict(hit)
            rec["rank"] = rank
            rec["display_rank"] = rank
            rec["score"] = float(hit.get("distance", 0.0))
            rec["retrieval_score"] = float(hit.get("distance", 0.0))
            rec["query"] = str(image_path)
            rows.append(rec)

        return pd.DataFrame.from_records(rows)

    def run_search(
        self,
        query: str,
        mode: SearchMode = "semantic",
        use_split: bool = True,
        top_k: int = 10,
        candidate_multiplier: int = 5,
        duration_limit: float = -1,
        search_ef: int | None = None,
    ) -> tuple[pd.DataFrame, QueryPlan]:
        plan = self.build_query_plan(query, mode, use_split)
        if not plan.events:
            return pd.DataFrame(), plan
        candidate_k = max(int(top_k) * int(candidate_multiplier), int(top_k))
        effective_mode = resolve_effective_mode(mode, plan)

        if effective_mode == "semantic":
            return self.semantic_search(
                plan.events, top_k, candidate_k, ef=search_ef
            ), plan
        if effective_mode == "temporal":
            return self.temporal_search(
                plan.events, top_k, candidate_k, duration_limit, ef=search_ef
            ), plan
        if effective_mode in {"ocr", "asr"}:
            raise NotImplementedError(
                f"Search mode '{effective_mode}' is not implemented yet"
            )
        raise ValueError(f"Unsupported search mode: {effective_mode}")

    def get_frame(self, video_id: str, keyframe_id: int) -> dict[str, Any] | None:
        results = self._milvus_query(
            filter_expr=f'video_id == "{video_id}" and keyframe_id_int == {int(keyframe_id)}',
            output_fields=SCALAR_OUTPUT_FIELDS,
            limit=1,
        )
        if not results:
            return None
        return search_result_to_dict(results[0])

    def get_video_frames(self, video_id: str) -> list[dict[str, Any]]:
        results = self._milvus_query(
            filter_expr=f'video_id == "{video_id}"',
            output_fields=SCALAR_OUTPUT_FIELDS,
            limit=10000,
        )
        frames = [search_result_to_dict(r) for r in results]
        frames.sort(key=lambda r: int(r.get("keyframe_id_int", 0) or 0))
        return frames
