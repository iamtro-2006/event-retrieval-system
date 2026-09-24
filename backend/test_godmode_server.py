from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import godmode_server


class GodModeRelayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        godmode_server.DB_PATH = Path(self.temp.name) / "relay.sqlite3"
        godmode_server.connections.clear()
        self.payload = godmode_server.VerifiedSubmit(
            dres_url="https://dres.example",
            session_id="session",
            evaluation_id="evaluation",
            task="kis",
            items=[{"video_id": "L01_V001", "frame_id": 42, "timestamp": 1.5}],
            result={"image_url": "http://localhost/frame.jpg"},
        )

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def response(verdict):
        response = Mock(ok=True, status_code=200, text=json.dumps({"submission": verdict}))
        response.json.return_value = {"submission": verdict}
        return response

    def test_only_correct_verdict_is_persisted(self):
        with patch("godmode_server.requests.post", return_value=self.response("WRONG")):
            result = asyncio.run(godmode_server.submit(self.payload))
        self.assertEqual(result["status"], "wrong")
        self.assertEqual(godmode_server.read_results("evaluation"), [])

        with patch("godmode_server.requests.post", return_value=self.response("CORRECT")):
            result = asyncio.run(godmode_server.submit(self.payload))
        self.assertEqual(result["status"], "correct")
        saved = godmode_server.read_results("evaluation")
        self.assertEqual(len(saved), 1)
        self.assertTrue(saved[0]["godmode_verified"])
        self.assertEqual(saved[0]["frame_id"], 42)


if __name__ == "__main__":
    unittest.main()
