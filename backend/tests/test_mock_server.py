"""Run: python -m unittest discover -s tests -p test_mock_server.py -v"""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

import mock_server
from src.mock_tools import MockCatalog, MockTools
from src import mock_codex
from src.mock_workers import stream_parallel


class MockServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        for dataset, color in (("V3C", "red"), ("LHE", "blue")):
            folder = cls.root / "keyframes" / dataset / "same_video"
            folder.mkdir(parents=True)
            for index in range(6):
                Image.new("RGB", (24, 16), color).save(folder / f"{index:06}.jpg")
        cls.override = patch.object(mock_server, "DATA_ROOT", cls.root)
        cls.override.start()
        cls.client = TestClient(mock_server.app)
        cls.client.__enter__()
        # Static mounts are defined at module import; test catalog image bytes via
        # inspect_frames while live smoke tests exercise actual mounted files.

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)
        cls.override.stop()
        cls.temp.cleanup()

    def test_all_search_modes_and_dataset_isolation(self):
        for dataset in ("v3c", "lhe"):
            for mode in ("semantic", "temporal", "ocr", "asr", "auto", "fusion", "color", "advanced"):
                response = self.client.post(f"/api/search/{mode}", json={"dataset": dataset, "query": "test", "top_k": 4})
                self.assertEqual(response.status_code, 200, response.text)
                rows = response.json()["results"]
                self.assertEqual(len(rows), 4)
                self.assertEqual({row["dataset"] for row in rows}, {dataset})
                self.assertTrue(all(f"/keyframes/{dataset.upper()}/" in row["image_url"] for row in rows))
                self.assertTrue(all("keyframe_path" not in row for row in rows))

    def test_filters_and_invalid_inputs(self):
        response = self.client.post("/api/search", json={"dataset": "lhe", "video_ids": ["does-not-exist"]})
        self.assertEqual(response.json()["results"], [])
        for body in ({"dataset": "other"}, {"dataset": "lhe", "top_k": 501}, {"dataset": "lhe", "search_mode": "shell"}):
            self.assertEqual(self.client.post("/api/search", json=body).status_code, 422)

    def test_neighbors_and_similarity(self):
        response = self.client.get("/api/surrounding-frames", params={"dataset": "lhe", "video_id": "same_video", "keyframe_id": 2, "radius": 1})
        self.assertEqual([row["frame_id"] for row in response.json()["frames"]], [1, 2, 3])
        response = self.client.post("/api/similarity-search", json={"dataset": "lhe", "video_id": "missing"})
        self.assertEqual(response.status_code, 404)

    def test_browse_entire_video_preserves_order_center_and_dataset(self):
        for dataset in ("v3c", "lhe"):
            response = self.client.get("/api/surrounding-frames", params={
                "dataset": dataset, "video_id": "same_video", "keyframe_id": 3,
                "radius": 1, "all_frames": "true"})
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual([row["frame_id"] for row in data["frames"]], list(range(6)))
            self.assertEqual(data["center_frame_id"], 3)
            self.assertEqual(data["count"], 6)
            self.assertEqual({row["dataset"] for row in data["frames"]}, {dataset})
        response = self.client.get("/api/surrounding-frames", params={
            "dataset": "lhe", "video_id": "missing", "keyframe_id": 0, "all_frames": "true"})
        self.assertEqual(response.status_code, 404)

    def test_shared_tools_validate_ids_and_return_actual_pixels(self):
        tools = MockTools(self.client.app.state.catalog, "lhe")
        data, _ = tools.execute("search", {"query": "test", "top_k": 2})
        identifiers = [row["id"] for row in data["results"]]
        inspected, content = tools.execute("inspect_frames", {"frame_ids": identifiers})
        self.assertEqual(inspected["count"], 2)
        self.assertEqual(len([item for item in content if item["type"] == "inputImage"]), 2)
        self.assertTrue(content[1]["imageUrl"].startswith("data:image/jpeg;base64,"))
        with self.assertRaises(ValueError):
            tools.execute("present_results", {"frame_ids": ["v3c:same_video/000001.jpg"]})
        with self.assertRaises(ValueError):
            tools.execute("shell", {})
        with self.assertRaises(ValueError):
            tools.execute("search", {"query": "test", "dataset": "v3c"})

    def test_demo_stream_and_followup_session(self):
        body = {"dataset": "lhe", "provider": "demo", "message": "test chat", "top_k": 2}
        response = self.client.post("/api/agent/chat", json=body)
        events = [json.loads(line) for line in response.text.splitlines()]
        self.assertEqual(events[-1]["type"], "done")
        self.assertTrue(any(event["type"] == "delta" for event in events))
        self.assertEqual([event["name"] for event in events if event["type"] == "tool_start"],
                         ["list_capabilities", "search", "inspect_frames"])
        rows = next(event["results"] for event in events if event["type"] == "results")
        self.assertEqual({row["dataset"] for row in rows}, {"lhe"})
        identifier = events[0]["session_id"]
        response = self.client.post("/api/agent/chat", json={**body, "session_id": identifier, "message": "follow up"})
        self.assertEqual(json.loads(response.text.splitlines()[-1])["type"], "done")
        response = self.client.post("/api/agent/chat", json={**body, "session_id": identifier, "dataset": "v3c"})
        self.assertEqual(response.status_code, 409)
        self.assertFalse(self.client.app.state.sessions[identifier]["busy"])

    def test_codex_failure_is_visible_without_demo_fallback(self):
        async def failing(*args):
            raise RuntimeError("test authentication failure")
            yield  # pragma: no cover
        with patch.object(mock_server, "stream_codex", failing):
            response = self.client.post("/api/agent/chat", json={"dataset": "v3c", "provider": "codex", "message": "test"})
        events = [json.loads(line) for line in response.text.splitlines()]
        self.assertEqual(events[-1], {"type": "error", "message": "test authentication failure"})
        self.assertFalse(any(event["type"] == "tool_start" for event in events))

    def test_chat_worker_limits_and_context_validation(self):
        base = {"dataset": "lhe", "provider": "demo", "message": "test"}
        for workers in (0, 4):
            self.assertEqual(self.client.post("/api/agent/chat", json={**base, "max_workers": workers}).status_code, 422)
        self.assertEqual(self.client.post("/api/agent/chat", json={**base,
            "context_frame_ids": ["v3c:same_video/000001.jpg"]}).status_code, 422)
        self.assertEqual(self.client.post("/api/agent/chat", json={**base,
            "images": ["https://remote.example/image.jpg"]}).status_code, 422)
        response = self.client.post("/api/agent/chat", json={**base,
            "context_frame_ids": ["lhe:same_video/000001.jpg"], "max_workers": 2,
            "color_cells": [{"row": 0, "col": 4, "color": "blue"}]})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.text.splitlines()[-1])["type"], "done")

    def test_empty_dataset_does_not_borrow_images(self):
        with tempfile.TemporaryDirectory() as empty:
            catalog = MockCatalog(Path(empty))
            catalog.load()
            self.assertEqual(catalog.search("lhe")["results"], [])

    def test_local_cors(self):
        response = self.client.options("/api/agent/chat", headers={"Origin": "http://127.0.0.1:5174",
                              "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["access-control-allow-origin"], "http://127.0.0.1:5174")


class CodexProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_shared_worker_slots_limit_multiple_coordinators(self):
        shared = asyncio.Semaphore(2)
        catalog = MockCatalog(Path("."))
        tools = [MockTools(catalog, "lhe", max_workers=3, worker_slots=shared) for _ in range(2)]
        active = 0
        peak = 0
        async def fake_child(*args):
            nonlocal active, peak
            active += 1
            peak = max(active, peak)
            try:
                await asyncio.sleep(.02)
                yield {"type": "delta", "text": "report"}
            finally:
                active -= 1
        async def collect(tool):
            return [event async for event in stream_parallel(Path("."), tool, "parallel_tasks", {"tasks": [
                {"instruction": "one"}, {"instruction": "two"}, {"instruction": "three"}]})]
        with patch.object(mock_codex, "stream_codex", fake_child):
            results = await asyncio.gather(*(collect(tool) for tool in tools))
        self.assertEqual(peak, 2)
        self.assertTrue(all(events[-1]["data"]["successful_workers"] == 3 for events in results))

    async def test_real_worker_dispatch_overlaps_and_preserves_frame_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "keyframes" / "LHE" / "video"
            folder.mkdir(parents=True)
            for index in range(18):
                Image.new("RGB", (16, 16), "blue").save(folder / f"{index:06}.jpg")
            catalog = MockCatalog(root)
            catalog.load()
            ids = [row["id"] for row in catalog.frames["lhe"]]
            tools = MockTools(catalog, "lhe", ids, max_workers=3)
            tools.reasoning_effort = "high"
            active = 0
            peak = 0
            closed = []
            async def fake_child(cwd, child, message, history, model):
                nonlocal active, peak
                self.assertTrue(child.is_worker)
                self.assertEqual(child.reasoning_effort, "high")
                self.assertEqual(child.dataset, "lhe")
                self.assertEqual(len(child.input_images), 6)
                active += 1
                peak = max(peak, active)
                try:
                    await asyncio.sleep(.02)
                    yield {"type": "delta", "text": "Evidence for assigned frames."}
                finally:
                    active -= 1
                    closed.append(True)
            with patch.object(mock_codex, "stream_codex", fake_child):
                events = [event async for event in stream_parallel(root, tools, "parallel_inspect", {"frame_ids": ids})]
            self.assertEqual(peak, 3)
            self.assertEqual(len(closed), 3)
            self.assertEqual(events[-1]["data"]["successful_workers"], 3)
            self.assertEqual([event["type"] for event in events].count("worker_start"), 3)
            # A lower configured limit must constrain actual worker overlap.
            tools.max_workers = 1
            peak = 0
            with patch.object(mock_codex, "stream_codex", fake_child):
                _ = [event async for event in stream_parallel(root, tools, "parallel_inspect", {"frame_ids": ids})]
            self.assertEqual(peak, 1)

    async def test_stopping_parallel_dispatch_closes_all_children(self):
        catalog = MockCatalog(Path("."))
        tools = MockTools(catalog, "lhe", max_workers=3)
        closed = []
        async def waiting_child(*args):
            try:
                await asyncio.sleep(60)
                yield {"type": "delta", "text": "never reached"}
            finally:
                closed.append(True)
        with patch.object(mock_codex, "stream_codex", waiting_child):
            source = stream_parallel(Path("."), tools, "parallel_tasks", {"tasks": [
                {"instruction": "task one"}, {"instruction": "task two"}, {"instruction": "task three"}]})
            await anext(source)
            await asyncio.sleep(.01)
            await source.aclose()
        self.assertEqual(len(closed), 3)

    async def test_worker_errors_are_reported_without_fake_success(self):
        tools = MockTools(MockCatalog(Path(".")), "lhe")
        async def failed_child(*args):
            raise RuntimeError("worker quota exhausted")
            yield
        with patch.object(mock_codex, "stream_codex", failed_child):
            events = [event async for event in stream_parallel(Path("."), tools, "parallel_tasks",
                      {"tasks": [{"instruction": "find scenes"}]})]
        self.assertEqual(events[-1]["data"]["successful_workers"], 0)
        self.assertIn("worker quota exhausted", events[-1]["data"]["reports"][0]["error"])

    async def test_disconnect_closes_codex_generator_and_releases_session(self):
        closed = []
        async def fake_stream(*args):
            try:
                yield {"type": "status", "message": "running"}
                await asyncio.sleep(60)
            finally:
                closed.append(True)
        state = SimpleNamespace(catalog=MockCatalog(Path(".")), sessions={}, codex_slots=asyncio.Semaphore(2))
        request = SimpleNamespace(app=SimpleNamespace(state=state), base_url="http://localhost/")
        with patch.object(mock_server, "stream_codex", fake_stream):
            response = await mock_server.chat(mock_server.ChatPayload(dataset="lhe", provider="codex", message="test"), request)
            await anext(response.body_iterator)  # Session event.
            await anext(response.body_iterator)  # Model status, then disconnect.
            await response.body_iterator.aclose()
        self.assertEqual(closed, [True])
        self.assertFalse(next(iter(state.sessions.values()))["busy"])
        self.assertFalse(state.codex_slots.locked())

    async def test_dynamic_tool_roundtrip_and_final_message(self):
        class Tools:
            reasoning_effort = "high"
            dataset = "lhe"
            known_ids = set()
            video_ids = []
            calls = 0
            max_calls = 12
            def specs(self): return []
            def execute(self, name, args):
                self.calls += 1
                return {"results": [{"id": "lhe:1", "dataset": "lhe"}], "count": 1}, []
        class FakeClient:
            instance = None
            def __init__(self, cwd):
                FakeClient.instance = self
                self.sent = []
                self.process = None
                self.events = iter([
                    {"id": 8, "method": "item/tool/call", "params": {"tool": "search", "arguments": {"query": "test"}}},
                    {"method": "item/agentMessage/delta", "params": {"delta": "mock reply"}},
                    {"method": "turn/completed", "params": {"turn": {"status": "completed"}}},
                ])
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            async def rpc(self, method, params):
                if method == "turn/start":
                    self.turn_params = params
                if method == "account/read": return {"account": {"type": "chatgpt"}}
                if method == "thread/start": return {"thread": {"id": "thread-test"}}
                return {"turn": {"id": "turn-test"}}
            async def receive(self): return next(self.events)
            def send(self, message): self.sent.append(message)
        with patch.object(mock_codex, "CodexProcess", FakeClient):
            events = [event async for event in mock_codex.stream_codex(Path("."), Tools(), "test", [])]
        self.assertTrue(FakeClient.instance.sent[0]["result"]["success"])
        self.assertEqual(FakeClient.instance.turn_params["effort"], "high")
        self.assertEqual(FakeClient.instance.sent[0]["result"]["contentItems"][0]["type"], "inputText")
        self.assertTrue(any(event["type"] == "results" for event in events))
        self.assertEqual(events[-1], {"type": "delta", "text": "mock reply"})
        Tools.reasoning_effort = None
        with patch.object(mock_codex, "CodexProcess", FakeClient):
            _ = [event async for event in mock_codex.stream_codex(Path("."), Tools(), "test", [])]
        self.assertNotIn("effort", FakeClient.instance.turn_params)


class MockVideoMappingTests(unittest.TestCase):
    def test_strip_and_preview_use_csv_timing_and_dataset_relative_urls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for dataset in ("V3C", "LHE"):
                folder = root / "keyframes" / dataset / "same_video"
                folder.mkdir(parents=True)
                for identifier in (2, 5, 9):
                    Image.new("RGB", (12, 8), "blue").save(folder / f"{identifier:06}.jpg")
            mapping = root / "map_keyframes" / "V3C"
            mapping.mkdir(parents=True)
            (mapping / "same_video.csv").write_text(
                "keyframe_id,video_id,frame_idx,timestamp_sec,fps\n"
                "2,same_video,25,1.25,20\n5,same_video,100,5,20\n9,same_video,190,9.5,20\n",
                encoding="utf-8")
            with patch.object(mock_server, "DATA_ROOT", root), TestClient(mock_server.app) as client:
                response = client.get("/api/video-keyframes", params={"dataset": "v3c", "video_id": "same_video"})
                self.assertEqual(response.status_code, 200)
                frames = response.json()["keyframes"]
                self.assertEqual([frame["keyframe_id"] for frame in frames], ["2", "5", "9"])
                self.assertEqual([frame["timestamp"] for frame in frames], [1.25, 5, 9.5])
                self.assertEqual(frames[0]["frame_idx"], 25)
                self.assertFalse(frames[0]["synthetic_timing"])
                self.assertEqual(frames[0]["image_rel_path"], "V3C/same_video/000002.jpg")
                self.assertIn("/keyframes/V3C/same_video/000002.jpg", frames[0]["image_url"])
                preview = client.get("/api/video-preview", params={"dataset": "v3c", "video_id": "same_video", "timestamp_ms": 2750}).json()
                self.assertEqual(preview["timestamp"], 2.75)
                self.assertEqual(preview["frame_idx"], 55)
                self.assertEqual(preview["raw"]["frame_idx"], 55)
                by_frame = client.get("/api/video-preview", params={"dataset": "v3c", "video_id": "same_video", "frame_id": 100}).json()
                self.assertEqual(by_frame["timestamp"], 5)
                self.assertEqual(by_frame["keyframe_id"], "5")
                fallback = client.get("/api/video-keyframes", params={"dataset": "lhe", "video_id": "same_video"}).json()["keyframes"]
                self.assertEqual(fallback[0]["timestamp"], 2)
                self.assertTrue(fallback[0]["synthetic_timing"])
                self.assertIn("/keyframes/LHE/", fallback[0]["image_url"])
                self.assertEqual(client.get("/api/video-keyframes", params={"dataset": "lhe", "video_id": "missing"}).status_code, 404)


if __name__ == "__main__":
    unittest.main()
