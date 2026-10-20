from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from src.embedding_extraction.models.embedder import encode_keyframe_images
from src.embedding_extraction.models.registry import MODEL_PRESETS, load_model


class _FakeSentenceTransformer:
    def __init__(self, model_id, **kwargs):
        self.model_id = model_id
        self.kwargs = kwargs
        self.calls = []

    def eval(self):
        return self

    def get_sentence_embedding_dimension(self):
        return 2048

    def encode(self, values, **kwargs):
        self.calls.append((values, kwargs))
        result = np.zeros((len(values), 2048), dtype=np.float32)
        result[:, 0] = 1.0
        return result


class Qwen3VLEmbeddingTests(unittest.TestCase):
    def test_vllm_mode_uses_remote_encoder_without_loading_sentence_transformer(self):
        from src.embedding_extraction.models.backends import qwen3_vl_embedding

        with patch.object(qwen3_vl_embedding, "resolve_engine", return_value="vllm"), patch.object(
            qwen3_vl_embedding, "QwenVllmEncoder"
        ) as encoder:
            loaded = load_model(
                "Qwen/Qwen3-VL-Embedding-2B", precision="fp32", device_name="cpu", engine="vllm"
            )
            self.assertIs(loaded.model, encoder.return_value)
            self.assertEqual(loaded.embedding_dim, 2048)
            self.assertTrue(loaded.supports_text)

    def test_registry_path_images_and_text_queries_share_model(self):
        fake_package = types.ModuleType("sentence_transformers")
        fake_package.SentenceTransformer = _FakeSentenceTransformer
        with patch.dict(sys.modules, {"sentence_transformers": fake_package}):
            loaded = load_model("Qwen/Qwen3-VL-Embedding-2B", precision="fp32", device_name="cpu", engine="sentence_transformer")
            self.assertEqual(MODEL_PRESETS["Qwen/Qwen3-VL-Embedding-2B"]["backend"], "qwen3_vl_embedding")
            self.assertEqual(loaded.embedding_dim, 2048)
            self.assertTrue(loaded.supports_text)
            self.assertIsNone(loaded.preprocess)

            paths = [Path("/frames/a.jpg"), Path("/frames/b.jpg")]
            images, valid = encode_keyframe_images(
                loaded.model, loaded.preprocess, loaded.device, paths, 2, loaded.precision
            )
            self.assertEqual(valid, paths)
            self.assertEqual(images.shape, (2, 2048))
            self.assertEqual(loaded.model.model.calls[0][0], [{"image": str(path)} for path in paths])
            self.assertTrue(loaded.model.model.calls[0][1]["normalize_embeddings"])

            queries = loaded.model.encode_text(["bicycle", "river"])
            self.assertEqual(queries.shape, torch.Size([2, 2048]))
            self.assertEqual(loaded.model.model.calls[1][0], ["bicycle", "river"])
            self.assertTrue(loaded.model.model.calls[1][1]["normalize_embeddings"])


if __name__ == "__main__":
    unittest.main()
