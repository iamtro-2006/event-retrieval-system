import test from "node:test";
import assert from "node:assert/strict";
import { centeredKeyframeOffset, findActiveKeyframe, initialVideoMilliseconds, normalizeVideoKeyframes } from "../src/utils/videoKeyframes.js";
import { createMediaUrlResolver } from "../src/api/mediaUrls.js";

test("playback selects the preceding keyframe, updates at boundaries and handles backward seeks", () => {
  const items = [{ timestamp: 1 }, { timestamp: 4 }, { timestamp: 9 }];
  for (const [seconds, expected] of [[0, 0], [1, 0], [3.99, 0], [4, 1], [8, 1], [9, 2], [100, 2], [2, 0]]) {
    assert.equal(findActiveKeyframe(items, seconds), expected);
  }
  assert.equal(findActiveKeyframe([], 0), -1);
});

test("centering clamps naturally at the first/last keyframe without empty padding", () => {
  assert.equal(centeredKeyframeOffset(0, 400, 10), 0);
  assert.equal(centeredKeyframeOffset(4, 400, 10), 318);
  assert.equal(centeredKeyframeOffset(9, 400, 10), 752);
  assert.equal(centeredKeyframeOffset(0, 400, 1), 0);
});

test("keyframe labels retain sparse identities rather than decoded frame numbers", () => {
  const items = normalizeVideoKeyframes([
    { keyframe_id: "000019", frame_idx: 5000, timestamp: 200 },
    { keyframe_id: 3, frame_idx: 100, timestamp: 4 },
    { keyframe_id: 3, frame_idx: 100, timestamp: 4 },
    { keyframe_id: 7, timestamp: null },
    { keyframe_id: 8, timestamp: -1 },
  ]);
  assert.deepEqual(items.map((item) => item.keyframe_id), ["3", "000019"]);
  assert.equal(initialVideoMilliseconds({ is_preview_frame: true, frame_idx: 5000, raw: { fps: 25 } }), 200000);
});

test("both backend URL styles resolve on the configured media host within their dataset", () => {
  const resolve = createMediaUrlResolver({ apiBaseUrl: "https://api.example", mediaBases: { keyframes: "https://media.example/keyframes" } });
  assert.equal(resolve("v3c", "keyframes", "V3C/00001/000003.jpg", "/static/v3c/keyframes/V3C/00001/000003.jpg"), "https://media.example/keyframes/V3C/00001/000003.jpg");
  assert.equal(resolve("lhe", "keyframes", "LHE/LHE01/000003.jpg", "/keyframes/LHE/LHE01/000003.jpg"), "https://media.example/keyframes/LHE/LHE01/000003.jpg");
  const local = createMediaUrlResolver({ apiBaseUrl: "http://127.0.0.1:8001" });
  assert.equal(local("v3c", "keyframes", "V3C/video/3.jpg", "/keyframes/V3C/video/3.jpg"), "http://127.0.0.1:8001/keyframes/V3C/video/3.jpg");
});
