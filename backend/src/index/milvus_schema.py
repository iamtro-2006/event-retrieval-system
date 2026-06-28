from __future__ import annotations

from pymilvus import DataType, MilvusClient


VECTOR_FIELD = "vector"
PK_FIELD = "pk"

SCALAR_FIELDS = [
    ("dataset", DataType.VARCHAR, {"max_length": 64}),
    ("video_id", DataType.VARCHAR, {"max_length": 64}),
    ("keyframe_id", DataType.VARCHAR, {"max_length": 16}),
    ("keyframe_id_int", DataType.INT64, {}),
    ("source_name", DataType.VARCHAR, {"max_length": 128}),
    ("frame_idx", DataType.INT64, {}),
    ("timestamp_sec", DataType.DOUBLE, {}),
    ("fps", DataType.FLOAT, {}),
    ("keyframe_path", DataType.VARCHAR, {"max_length": 512}),
    ("embedding_path", DataType.VARCHAR, {"max_length": 512}),
    ("map_path", DataType.VARCHAR, {"max_length": 512}),
]

SCALAR_FIELD_NAMES = [name for name, _, _ in SCALAR_FIELDS]


def build_schema(dim: int):
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(
        field_name=PK_FIELD,
        datatype=DataType.VARCHAR,
        max_length=128,
        is_primary=True,
    )
    schema.add_field(
        field_name=VECTOR_FIELD,
        datatype=DataType.FLOAT_VECTOR,
        dim=dim,
    )
    for name, dtype, kwargs in SCALAR_FIELDS:
        schema.add_field(
            field_name=name,
            datatype=dtype,
            is_partition_key=(name == "video_id"),
            **kwargs,
        )
    return schema


def build_index_params(cfg: dict):
    index_cfg = cfg["index"]
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name=index_cfg["field_name"],
        index_type=index_cfg["index_type"],
        metric_type=index_cfg["metric_type"],
        params=index_cfg["params"],
    )
    return index_params


def create_collection(
    client: MilvusClient,
    collection_name: str,
    dim: int,
    cfg: dict,
):
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    schema = build_schema(dim)
    index_params = build_index_params(cfg)
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )
    client.load_collection(collection_name)


def detect_dimension(embeddings_root) -> int:
    import numpy as np
    from src.index.embedding_index import collect_embedding_files

    paths = collect_embedding_files(embeddings_root)
    if not paths:
        raise RuntimeError(f"No embedding files found in: {embeddings_root}")
    vec = np.load(paths[0]).astype(np.float32).reshape(-1)
    return int(vec.shape[0])
