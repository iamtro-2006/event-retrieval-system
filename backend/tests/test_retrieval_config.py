from __future__ import annotations

from pathlib import Path

import pytest

from src.config.retrieval_config import RetrievalConfig, load_retrieval_config, to_dict


CONFIG_YAML = """
backend: milvus

milvus:
  host: localhost
  port: 19530
  collection_name: test_keyframes
  consistency_level: Bounded
  search_params:
    metric_type: IP
    params:
      ef: 128

faiss:
  index_path: data/faiss.index
  metadata_path: data/metadata.csv
  ef_search: 64
  threads: 4
  vector_cache_mode: memmap
  vector_cache_dtype: float16
  vector_cache_path: data/vectors.npy

model:
  name: ViT-B-16-quickgelu
  pretrained: dfn2b
  device: cuda
  precision: fp16
  normalize: true
  compile: false

paths:
  keyframes_root: data/keyframes
  videos_root: data/videos
  map_keyframe_path: data/maps

search:
  candidate_multiplier: 3
  default_top_k: 20
  max_top_k: 200

ui:
  surrounding_radius: 5
  max_surrounding_radius: 10

translate:
  enabled_default: true
  source: vi
  target: en

speech:
  model_size: base
  device: cpu
  compute_type: int8
  lazy_load: true

debug:
  profile: true
"""


@pytest.fixture
def config_file(tmp_path):
    path = tmp_path / "app.yaml"
    path.write_text(CONFIG_YAML, encoding="utf-8")
    return path


class TestLoadRetrievalConfig:
    def test_backend(self, config_file):
        cfg = load_retrieval_config(config_file)
        assert cfg.backend == "milvus"

    def test_milvus(self, config_file):
        cfg = load_retrieval_config(config_file)
        assert cfg.milvus.host == "localhost"
        assert cfg.milvus.port == 19530
        assert cfg.milvus.collection_name == "test_keyframes"
        assert cfg.milvus.search_params.ef == 128
        assert cfg.milvus.search_params.metric_type == "IP"

    def test_faiss(self, config_file):
        cfg = load_retrieval_config(config_file)
        assert cfg.faiss.index_path == "data/faiss.index"
        assert cfg.faiss.vector_cache_mode == "memmap"
        assert cfg.faiss.vector_cache_dtype == "float16"

    def test_model(self, config_file):
        cfg = load_retrieval_config(config_file)
        assert cfg.model.name == "ViT-B-16-quickgelu"
        assert cfg.model.pretrained == "dfn2b"
        assert cfg.model.device == "cuda"
        assert cfg.model.precision == "fp16"

    def test_paths(self, config_file):
        cfg = load_retrieval_config(config_file)
        assert cfg.paths.keyframes_root == "data/keyframes"
        assert cfg.paths.videos_root == "data/videos"

    def test_search(self, config_file):
        cfg = load_retrieval_config(config_file)
        assert cfg.search.candidate_multiplier == 3
        assert cfg.search.default_top_k == 20
        assert cfg.search.max_top_k == 200

    def test_translate(self, config_file):
        cfg = load_retrieval_config(config_file)
        assert cfg.translate.enabled_default is True
        assert cfg.translate.source == "vi"

    def test_debug(self, config_file):
        cfg = load_retrieval_config(config_file)
        assert cfg.debug.profile is True

    def test_defaults_on_empty(self, tmp_path):
        path = tmp_path / "empty.yaml"
        path.write_text("backend: faiss\n", encoding="utf-8")
        cfg = load_retrieval_config(path)
        assert cfg.backend == "faiss"
        assert cfg.milvus.host == "localhost"
        assert cfg.search.default_top_k == 20
        assert cfg.faiss.vector_cache_dtype == "float32"


class TestToDict:
    def test_roundtrip_shape(self, config_file):
        cfg = load_retrieval_config(config_file)
        raw = to_dict(cfg)
        assert raw["backend"] == "milvus"
        assert raw["milvus"]["host"] == "localhost"
        assert raw["milvus"]["search_params"]["params"]["ef"] == 128
        assert raw["milvus"]["search_params"]["metric_type"] == "IP"
        assert raw["model"]["name"] == "ViT-B-16-quickgelu"
        assert raw["faiss"]["index_path"] == "data/faiss.index"
