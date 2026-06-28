from __future__ import annotations

import os

import numpy as np
import pytest


MILVUS_TEST_URI = os.environ.get("MILVUS_TEST_URI", "")


def _skip_if_no_milvus():
    if not MILVUS_TEST_URI:
        pytest.skip("MILVUS_TEST_URI not set; skipping Milvus integration tests")


def _make_test_collection(client, collection_name, dim=128):
    from pymilvus import DataType, MilvusClient
    from src.index.milvus_schema import PK_FIELD, SCALAR_FIELDS, VECTOR_FIELD

    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(
        field_name=PK_FIELD, datatype=DataType.VARCHAR, max_length=128, is_primary=True
    )
    schema.add_field(field_name=VECTOR_FIELD, datatype=DataType.FLOAT_VECTOR, dim=dim)
    for name, dtype, kwargs in SCALAR_FIELDS:
        schema.add_field(
            field_name=name,
            datatype=dtype,
            is_partition_key=(name == "video_id"),
            **kwargs,
        )

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name=VECTOR_FIELD,
        index_type="HNSW",
        metric_type="IP",
        params={"M": 16, "efConstruction": 100},
    )

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )
    client.load_collection(collection_name)


def _insert_test_data(client, collection_name, dim=128):
    from src.index.milvus_schema import PK_FIELD, VECTOR_FIELD

    rng = np.random.RandomState(42)
    records = []
    for i in range(50):
        dataset = "root"
        video_id = f"V{i // 10:03d}"
        kf_int = i % 10
        vec = rng.randn(dim).astype(np.float32)
        vec /= np.linalg.norm(vec)

        rec = {
            PK_FIELD: f"{dataset}/{video_id}/{kf_int:06d}",
            VECTOR_FIELD: vec.tolist(),
            "dataset": dataset,
            "video_id": video_id,
            "keyframe_id": f"{kf_int:06d}",
            "keyframe_id_int": kf_int,
            "source_name": f"{kf_int:06d}",
            "frame_idx": kf_int * 30,
            "timestamp_sec": float(kf_int * 1.0),
            "fps": 30.0,
            "keyframe_path": f"data/processed/keyframes/{dataset}/{video_id}/{kf_int:06d}.jpg",
            "embedding_path": f"data/processed/embeddings/{dataset}/{video_id}/{kf_int:06d}.npy",
            "map_path": f"data/processed/map_keyframes/{dataset}/{video_id}.csv",
        }
        records.append(rec)

    client.upsert(collection_name=collection_name, data=records)
    client.flush(collection_name)
    return records


@pytest.fixture(scope="module")
def milvus_client():
    _skip_if_no_milvus()
    from pymilvus import MilvusClient

    client = MilvusClient(uri=MILVUS_TEST_URI)
    yield client
    client.close()


@pytest.fixture(scope="module")
def test_collection(milvus_client):
    collection_name = "test_keyframes_unit"
    dim = 128
    _make_test_collection(milvus_client, collection_name, dim)
    _insert_test_data(milvus_client, collection_name, dim)
    yield collection_name
    milvus_client.drop_collection(collection_name)


class TestMilvusSchema:
    def test_schema_creation(self):
        from src.index.milvus_schema import (
            PK_FIELD,
            SCALAR_FIELDS,
            VECTOR_FIELD,
            build_schema,
        )

        schema = build_schema(dim=512)
        field_names = [f.name for f in schema.fields]
        assert PK_FIELD in field_names
        assert VECTOR_FIELD in field_names
        for name, _, _ in SCALAR_FIELDS:
            assert name in field_names

    def test_pk_is_primary(self):
        from src.index.milvus_schema import PK_FIELD, build_schema

        schema = build_schema(dim=128)
        pk_field = next(f for f in schema.fields if f.name == PK_FIELD)
        assert pk_field.is_primary is True

    def test_video_id_is_partition_key(self):
        from src.index.milvus_schema import build_schema

        schema = build_schema(dim=128)
        vid_field = next(f for f in schema.fields if f.name == "video_id")
        assert vid_field.is_partition_key is True


class TestMilvusSearch:
    def test_search_returns_results(self, milvus_client, test_collection):
        from src.index.milvus_retrieval import SCALAR_OUTPUT_FIELDS

        rng = np.random.RandomState(99)
        query_vec = rng.randn(128).astype(np.float32)
        query_vec /= np.linalg.norm(query_vec)

        results = milvus_client.search(
            collection_name=test_collection,
            data=[query_vec.tolist()],
            limit=5,
            output_fields=SCALAR_OUTPUT_FIELDS,
            search_params={"metric_type": "IP", "params": {"ef": 64}},
        )
        assert len(results) == 1
        assert len(results[0]) == 5
        for hit in results[0]:
            entity = hit.get("entity", hit)
            assert "video_id" in entity
            assert "keyframe_id_int" in entity

    def test_query_by_video_id(self, milvus_client, test_collection):
        from src.index.milvus_retrieval import SCALAR_OUTPUT_FIELDS

        results = milvus_client.query(
            collection_name=test_collection,
            filter='video_id == "V000"',
            output_fields=SCALAR_OUTPUT_FIELDS,
            limit=100,
        )
        assert len(results) == 10
        for r in results:
            assert r["video_id"] == "V000"

    def test_query_by_video_and_frame(self, milvus_client, test_collection):
        from src.index.milvus_retrieval import SCALAR_OUTPUT_FIELDS

        results = milvus_client.query(
            collection_name=test_collection,
            filter='video_id == "V001" and keyframe_id_int == 3',
            output_fields=SCALAR_OUTPUT_FIELDS,
            limit=1,
        )
        assert len(results) == 1
        assert results[0]["video_id"] == "V001"
        assert results[0]["keyframe_id_int"] == 3


class TestMilvusRetrievalSystem:
    def test_get_video_frames(self, milvus_client, test_collection):
        _skip_if_no_milvus()

        class FakeBackend:
            def __init__(self, client, collection_name):

                self._client = client
                self._collection = collection_name
                self._dim = 128

            def get_video_frames(self, video_id):
                from src.index.milvus_retrieval import (
                    search_result_to_dict,
                    SCALAR_OUTPUT_FIELDS,
                )

                results = self._client.query(
                    collection_name=self._collection,
                    filter=f'video_id == "{video_id}"',
                    output_fields=SCALAR_OUTPUT_FIELDS,
                    limit=10000,
                )
                frames = [search_result_to_dict(r) for r in results]
                frames.sort(key=lambda r: int(r.get("keyframe_id_int", 0) or 0))
                return frames

        backend = FakeBackend(milvus_client, test_collection)
        frames = backend.get_video_frames("V002")
        assert len(frames) == 10
        kf_ids = [f["keyframe_id_int"] for f in frames]
        assert kf_ids == sorted(kf_ids)


class TestQueryPlanning:
    def test_build_query_plan_semantic(self):
        from src.index.query_planning import build_query_plan

        plan = build_query_plan("a person walking", "semantic", True)
        assert plan.mode == "semantic"
        assert len(plan.events) == 1
        assert plan.events[0][0] == "a person walking"

    def test_build_query_plan_temporal(self):
        from src.index.query_planning import build_query_plan

        plan = build_query_plan("a person walking. then sitting down", "temporal", True)
        assert len(plan.events) == 2
        assert plan.events[0][0] == "a person walking"
        assert plan.events[1][0] == "then sitting down"

    def test_clean_queries_dedup(self):
        from src.index.query_planning import clean_queries

        result = clean_queries(["hello", "hello", "world"])
        assert result == ["hello", "world"]

    def test_resolve_effective_mode_auto(self):
        from src.index.query_planning import build_query_plan, resolve_effective_mode

        plan_single = build_query_plan("single query", "auto")
        assert resolve_effective_mode("auto", plan_single) == "semantic"

        plan_multi = build_query_plan("first event. second event", "auto")
        assert resolve_effective_mode("auto", plan_multi) == "temporal"


class TestAggregateMultiQuery:
    def test_empty_indices(self):
        from src.index.query_planning import aggregate_multi_query

        df = aggregate_multi_query(
            scores=np.empty((0, 0), dtype=np.float32),
            indices=np.empty((0, 0), dtype=np.int64),
            queries=["test"],
            top_k=10,
        )
        assert df.empty

    def test_single_query_single_result(self):
        from src.index.query_planning import aggregate_multi_query

        scores = np.array([[0.9]], dtype=np.float32)
        indices = np.array([[0]], dtype=np.int64)
        records = [{"video_id": "V001", "keyframe_id_int": 0}]

        df = aggregate_multi_query(
            scores, indices, ["test"], 10, metadata_records=records
        )
        assert len(df) == 1
        assert df.iloc[0]["video_id"] == "V001"
        assert df.iloc[0]["retrieval_score"] == pytest.approx(0.91, abs=0.01)

    def test_multi_query_aggregation(self):
        from src.index.query_planning import aggregate_multi_query

        scores = np.array([[0.9, 0.8], [0.7, 0.85]], dtype=np.float32)
        indices = np.array([[0, 1], [0, 1]], dtype=np.int64)
        records = [
            {"video_id": "V001", "keyframe_id_int": 0},
            {"video_id": "V002", "keyframe_id_int": 1},
        ]

        df = aggregate_multi_query(
            scores, indices, ["q1", "q2"], 10, metadata_records=records
        )
        assert len(df) == 2
        assert "alignment_score" in df.columns
        assert "coverage_score" in df.columns
