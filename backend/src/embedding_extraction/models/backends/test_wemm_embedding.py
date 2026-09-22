from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch
import yaml

from src.embedding_extraction.models.backends.sentence_transformer_image import (
    validate_image_batch,
)
from src.embedding_extraction.models.backends.wemm_embedding import WeMMEncoder
from src.embedding_extraction.models.embedder import encode_keyframe_images
from src.embedding_extraction.models.registry import MODEL_PRESETS, load_model


class _FakeSentenceTransformer:
    calls = []

    def __init__(self, model_id, **kwargs):
        self.calls.append((model_id, kwargs))

    def eval(self):
        return self

    def get_sentence_embedding_dimension(self):
        return 2048

    @staticmethod
    def _vectors(count):
        return np.tile(np.eye(1, 2048, dtype=np.float32), (count, 1))

    def encode_document(self, documents, **kwargs):
        self.last_encode = ("encode_document", documents, kwargs)
        return self._vectors(len(documents))

    def encode_query(self, queries, **kwargs):
        self.last_encode = ("encode_query", queries, kwargs)
        return self._vectors(len(queries))


class WeMMEmbeddingTests(unittest.TestCase):
    def test_registry_official_apis_and_path_native_dispatch(self):
        fake_package = types.ModuleType("sentence_transformers")
        fake_package.SentenceTransformer = _FakeSentenceTransformer
        name = "tencent/WeMM-Embedding-2B"
        with patch.dict(sys.modules, {"sentence_transformers": fake_package}):
            loaded = load_model(name, precision="fp32", device_name="cpu")
            self.assertEqual(MODEL_PRESETS[name]["backend"], "wemm_embedding")
            self.assertIsNone(loaded.preprocess)
            self.assertEqual(loaded.embedding_dim, 2048)
            self.assertTrue(loaded.supports_text)

            paths = [Path("/frames/a.jpg"), Path("/frames/b.jpg")]
            vectors, valid = encode_keyframe_images(
                loaded.model,
                loaded.preprocess,
                loaded.device,
                paths,
                2,
                loaded.precision,
            )
            self.assertEqual(valid, paths)
            self.assertEqual(vectors.shape, (2, 2048))
            self.assertEqual(vectors.dtype, np.float32)
            self.assertEqual(loaded.model.model.last_encode[0], "encode_document")
            self.assertEqual(
                loaded.model.model.last_encode[1],
                [{"image": str(path)} for path in paths],
            )
            self.assertIs(
                loaded.model.model.last_encode[2]["normalize_embeddings"], False
            )

            queries = ["a person cooking", "a bicycle"]
            query_vectors = loaded.model.encode_text(queries)
            self.assertIsInstance(query_vectors, torch.Tensor)
            self.assertEqual(query_vectors.dtype, torch.float32)
            self.assertEqual(query_vectors.shape, (2, 2048))
            self.assertEqual(loaded.model.model.last_encode[0], "encode_query")
            self.assertEqual(loaded.model.model.last_encode[1], queries)
            self.assertIs(
                loaded.model.model.last_encode[2]["normalize_embeddings"], False
            )

        kwargs = _FakeSentenceTransformer.calls[-1][1]
        self.assertEqual(
            kwargs["revision"],
            "bbd6cd4bf52cfc6716f752a2df80b2706720bd95",
        )
        self.assertEqual(kwargs["model_kwargs"]["attn_implementation"], "sdpa")

    def test_revision_drift_is_rejected(self):
        with self.assertRaises(ValueError):
            load_model(
                "tencent/WeMM-Embedding-2B",
                precision="fp32",
                device_name="cpu",
                revision="wrong",
            )

    def test_validation_rejects_bad_shape_nan_and_norm(self):
        good = np.tile(np.eye(1, 2048, dtype=np.float32), (2, 1))
        self.assertEqual(validate_image_batch(good, 2, 2048).dtype, np.float32)
        with self.assertRaises(ValueError):
            validate_image_batch(good[:, :1024], 2, 2048)
        bad = good.copy()
        bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            validate_image_batch(bad, 2, 2048)
        with self.assertRaises(ValueError):
            validate_image_batch(good * 2, 2, 2048)

    def test_query_validation_rejects_invalid_vectors(self):
        class FakeModel:
            def __init__(self, value):
                self.value = value

            def encode_query(self, texts, **kwargs):
                return self.value

        good = np.zeros((1, 2048), dtype=np.float32)
        good[0, 0] = 1.0
        for bad in (good[:, :1024], good * 2, np.full_like(good, np.nan)):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                WeMMEncoder(
                    FakeModel(bad),
                    method="encode_document",
                    normalize_embeddings=False,
                    embedding_dim=2048,
                ).encode_text(["query"])

    def test_path_native_dispatch_uses_backend_dimension(self):
        class ThreeDimensionalPathModel:
            embedding_dim = 3

            def encode_image_paths(self, paths):
                return np.tile(
                    np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
                    (len(paths), 1),
                )

        paths = [Path("/frames/a.jpg"), Path("/frames/b.jpg")]
        vectors, valid = encode_keyframe_images(
            ThreeDimensionalPathModel(),
            None,
            torch.device("cpu"),
            paths,
            2,
            "fp32",
        )
        self.assertEqual(valid, paths)
        self.assertEqual(vectors.shape, (2, 3))
        self.assertEqual(vectors.dtype, np.float32)

    def test_existing_tensor_backend_path_is_unchanged(self):
        class TensorModel:
            def encode_image(self, batch):
                self.batch = batch
                return torch.tensor([[3.0, 4.0]], dtype=torch.float32)

        model = TensorModel()
        with tempfile.TemporaryDirectory() as folder:
            image_path = Path(folder) / "frame.jpg"
            self.assertTrue(
                cv2.imwrite(str(image_path), np.zeros((4, 4, 3), dtype=np.uint8))
            )
            vectors, valid = encode_keyframe_images(
                model,
                lambda _: torch.ones(3, 4, 4),
                torch.device("cpu"),
                [image_path],
                1,
                "fp32",
            )
        self.assertEqual(valid, [image_path])
        self.assertEqual(tuple(model.batch.shape), (1, 3, 4, 4))
        np.testing.assert_allclose(vectors, [[0.6, 0.8]], atol=1e-6)

    def test_shared_configs_add_wemm_without_changing_default(self):
        config_root = Path(__file__).resolve().parents[4] / "configs"
        app = yaml.safe_load((config_root / "app.yaml").read_text(encoding="utf-8"))
        embeddings = yaml.safe_load(
            (config_root / "embeddings.yaml").read_text(encoding="utf-8")
        )
        indexing = yaml.safe_load(
            (config_root / "indexing.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(app["semantic"]["default_model_key"], "pe-core")
        wemm = {
            entry["model_key"]: entry for entry in app["semantic"]["models"]
        }["wemm_embedding_2b"]
        self.assertEqual(wemm["backend"], "wemm_embedding")
        self.assertEqual(
            wemm["model_extra"]["revision"],
            "bbd6cd4bf52cfc6716f752a2df80b2706720bd95",
        )
        indexed = {entry["model_key"]: entry for entry in indexing["models"]}
        self.assertEqual(indexed["wemm_embedding_2b"]["index"]["dimension"], 2048)
        extraction = {entry["key"]: entry for entry in embeddings["models"]}
        self.assertFalse(extraction["wemm_embedding_2b"]["enabled"])
        self.assertEqual(
            extraction["wemm_embedding_2b"]["revision"],
            "bbd6cd4bf52cfc6716f752a2df80b2706720bd95",
        )


if __name__ == "__main__":
    unittest.main()
