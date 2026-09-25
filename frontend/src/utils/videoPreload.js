const preloads = new Map();
const MAX_PRELOADED_VIDEOS = 6;

export function preloadVideoMetadata(url) {
  if (!url || typeof document === "undefined") return;
  const normalizedUrl = String(url).split("#", 1)[0];
  if (!normalizedUrl || preloads.has(normalizedUrl)) return;

  const video = document.createElement("video");
  video.preload = "metadata";
  video.muted = true;
  video.src = normalizedUrl;
  video.load();
  preloads.set(normalizedUrl, video);

  while (preloads.size > MAX_PRELOADED_VIDEOS) {
    const [oldestUrl, oldestVideo] = preloads.entries().next().value;
    oldestVideo.pause();
    oldestVideo.removeAttribute("src");
    oldestVideo.load();
    preloads.delete(oldestUrl);
  }
}
