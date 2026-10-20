from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import numpy as np

from src.embedding_extraction.models.backends.qwen_vllm import (
    MODEL_ID,
    QwenVllmEncoder,
    resolve_engine,
)


class QwenVllmTests(unittest.TestCase):
    def test_engine_selection_is_explicit_and_falls_back_only_during_initialization(self):
        with patch("httpx.get") as get:
            self.assertEqual(resolve_engine("sentence_transformer"), "sentence_transformer")
            get.assert_not_called()
            get.return_value.json.return_value = {"data": [{"id": MODEL_ID}]}
            self.assertEqual(resolve_engine("auto"), "vllm")
            get.side_effect = httpx.ConnectError("offline")
            self.assertEqual(resolve_engine("auto"), "sentence_transformer")
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                resolve_engine("vllm")

    def test_image_and_text_use_vllm_chat_embedding_contract(self):
        seen = []

        def handle(request):
            body = json.loads(request.content)
            seen.append(body)
            vector = [0.0] * 2048
            vector[0] = 1.0
            return httpx.Response(200, json={"data": [{"embedding": vector}]})

        encoder = QwenVllmEncoder()
        encoder.client.close()
        encoder.client = httpx.Client(transport=httpx.MockTransport(handle))
        try:
            with tempfile.TemporaryDirectory() as folder:
                image = Path(folder) / "frame.jpg"
                image.write_bytes(b"test image payload")
                vectors = encoder.encode_image_paths([image, image])
            self.assertEqual(vectors.shape, (2, 2048))
            self.assertEqual(vectors.dtype, np.float32)
            self.assertTrue(seen[0]["messages"][1]["content"][0]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
            text = encoder.encode_text(["bicycle"])
            self.assertEqual(tuple(text.shape), (1, 2048))
            self.assertEqual(seen[-1]["messages"][1]["content"], [{"type": "text", "text": "bicycle"}])
            self.assertTrue(all(item["model"] == MODEL_ID and item["continue_final_message"] for item in seen))
        finally:
            encoder.close()

    def test_invalid_embedding_does_not_become_index_artifact(self):
        def handle(_request):
            return httpx.Response(200, json={"data": [{"embedding": [1.0, 0.0]}]})

        encoder = QwenVllmEncoder()
        encoder.client.close()
        encoder.client = httpx.Client(transport=httpx.MockTransport(handle))
        try:
            with self.assertRaisesRegex(ValueError, "Invalid Qwen vLLM embedding"):
                encoder.encode_text(["query"])
        finally:
            encoder.close()


if __name__ == "__main__":
    unittest.main()
