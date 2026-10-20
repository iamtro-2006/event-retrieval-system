import assert from "node:assert/strict";
import test from "node:test";
import { createMediaUrlResolver } from "./mediaUrls.js";

test("appends dataset after each independent media root", () => {
  const resolve = createMediaUrlResolver({
    apiBaseUrl: "http://localhost:8000",
    mediaBases: {
      keyframes: "https://images.example:8000/keyframes/",
      videos: "https://videos.example:8000/videos",
      mapKeyframes: "https://maps.example:8000/mapkeyframes",
    },
  });

  assert.equal(resolve("aic", "keyframes", "L21/L21_V001/000001.jpg"), "https://images.example:8000/keyframes/AIC/L21/L21_V001/000001.jpg");
  assert.equal(resolve("cam", "videos", "N001/N001-V001.mov"), "https://videos.example:8000/videos/CAM/N001/N001-V001.mov");
  assert.equal(resolve("cam", "mapKeyframes", "N001/N001-V001.csv"), "https://maps.example:8000/mapkeyframes/CAM/N001/N001-V001.csv");
});

test("does not append dataset twice when an older relative path includes it", () => {
  const resolve = createMediaUrlResolver({
    apiBaseUrl: "http://localhost:8000",
    mediaBases: { keyframes: "http://localhost:3174/keyframes" },
  });

  assert.equal(resolve("aic", "keyframes", "/AIC/L21/clip/1.jpg"), "http://localhost:3174/keyframes/AIC/L21/clip/1.jpg");
});

test("uses relative paths in old API URLs when the relay omits them", () => {
  const resolve = createMediaUrlResolver({
    apiBaseUrl: "http://localhost:8000",
    mediaBases: { keyframes: "http://localhost:3174/keyframes" },
  });

  assert.equal(resolve("aic", "keyframes", "", "http://old-api/static/aic/keyframes/L21/clip/1.jpg"), "http://localhost:3174/keyframes/AIC/L21/clip/1.jpg");
  assert.equal(resolve("cam", "keyframes", "", "http://old-api/static/aic/keyframes/L21/clip/1.jpg"), "http://old-api/static/aic/keyframes/L21/clip/1.jpg");
});

test("reads dataset after the media directory in a hosted fallback URL", () => {
  const resolve = createMediaUrlResolver({
    apiBaseUrl: "http://localhost:8000",
    mediaBases: { mapKeyframes: "https://maps.example/data/mapkeyframes" },
  });

  assert.equal(resolve("aic", "mapKeyframes", "", "https://old.example/mapkeyframes/AIC/L21/clip.csv"), "https://maps.example/data/mapkeyframes/AIC/L21/clip.csv");
});

test("keeps local API fallback when a media root is not configured", () => {
  const resolve = createMediaUrlResolver({
    apiBaseUrl: "http://localhost:8000",
    mediaBases: { keyframes: "http://localhost:3174/keyframes" },
  });

  assert.equal(resolve("cam", "videos", "N001/clip.mov", "/static/cam/videos/N001/clip.mov"), "http://localhost:8000/static/cam/videos/N001/clip.mov");
  assert.equal(resolve("cam", "mapKeyframes", "", "http://localhost:8000/static/cam/map-keyframes/N001/clip.csv"), "http://localhost:8000/static/cam/map-keyframes/N001/clip.csv");
  assert.equal(resolve("cam", "videos", "", ""), "#");
});

test("supports other dataset names without hardcoding AIC or CAM", () => {
  const resolve = createMediaUrlResolver({
    apiBaseUrl: "http://localhost:8000",
    mediaBases: { keyframes: "https://images.example/keyframes" },
  });

  assert.equal(resolve("vbs", "keyframes", "", "/static/vbs/keyframes/day1/clip/1.jpg"), "https://images.example/keyframes/VBS/day1/clip/1.jpg");
});
