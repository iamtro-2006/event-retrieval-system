import assert from "node:assert/strict";
import test from "node:test";
import { createMediaUrlResolver } from "./mediaUrls.js";

test("resolves each media type and dataset from independent sources", () => {
  const resolve = createMediaUrlResolver({
    apiBaseUrl: "http://localhost:8000",
    sharedBases: {
      keyframes: "http://localhost:3174/{DATASET}/keyframes/",
      videos: "https://videos.example/{dataset}/videos",
      mapKeyframes: "http://localhost:3174/{DATASET}/map-keyframes",
    },
  });

  assert.equal(resolve("aic", "keyframes", "L21/L21_V001/000001.jpg"), "http://localhost:3174/AIC/keyframes/L21/L21_V001/000001.jpg");
  assert.equal(resolve("cam", "videos", "N001/N001-V001.mov"), "https://videos.example/cam/videos/N001/N001-V001.mov");
  assert.equal(resolve("cam", "mapKeyframes", "N001/N001-V001.csv"), "http://localhost:3174/CAM/map-keyframes/N001/N001-V001.csv");
});

test("dataset override affects only its configured media type", () => {
  const resolve = createMediaUrlResolver({
    apiBaseUrl: "http://localhost:8000",
    sharedBases: { keyframes: "http://local/{DATASET}/keyframes", videos: "https://shared/{DATASET}/videos" },
    datasetBases: { cam: { videos: "https://cam-host/videos/" } },
  });

  assert.equal(resolve("cam", "videos", "N001/clip.mov"), "https://cam-host/videos/N001/clip.mov");
  assert.equal(resolve("cam", "keyframes", "N001/clip/1.jpg"), "http://local/CAM/keyframes/N001/clip/1.jpg");
  assert.equal(resolve("aic", "videos", "L21/clip.mp4"), "https://shared/AIC/videos/L21/clip.mp4");
});

test("uses relative paths in older God Mode URLs when the relay omits them", () => {
  const resolve = createMediaUrlResolver({
    apiBaseUrl: "http://localhost:8000",
    sharedBases: { keyframes: "http://local/{DATASET}/keyframes" },
  });

  assert.equal(resolve("aic", "keyframes", "", "http://old-api/static/aic/keyframes/L21/clip/1.jpg"), "http://local/AIC/keyframes/L21/clip/1.jpg");
  assert.equal(resolve("cam", "keyframes", "", "http://old-api/static/aic/keyframes/L21/clip/1.jpg"), "http://old-api/static/aic/keyframes/L21/clip/1.jpg");
});

test("keeps the old data root and API fallback when no per-media source is set", () => {
  const fromData = createMediaUrlResolver({ apiBaseUrl: "http://localhost:8000", dataBaseUrl: "http://localhost:3174/data" });
  const fromApi = createMediaUrlResolver({ apiBaseUrl: "http://localhost:8000" });

  assert.equal(fromData("aic", "videos", "L21/clip.mp4"), "http://localhost:3174/data/AIC/videos/L21/clip.mp4");
  assert.equal(fromApi("cam", "videos", "", "/static/cam/videos/N001/clip.mov"), "http://localhost:8000/static/cam/videos/N001/clip.mov");
});
