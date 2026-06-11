export function normalizeFrameId(frameId) {
  const number = Number(frameId);

  if (Number.isNaN(number) || number < 0) {
    return 0;
  }

  return Math.floor(number);
}

export function padFrame(frameId, padding = 5, extension = "jpg") {
  const normalized = normalizeFrameId(frameId);
  return `${String(normalized).padStart(padding, "0")}.${extension}`;
}

export function getNeighborFrames(frameId, step = 1) {
  const current = normalizeFrameId(frameId);

  return {
    prevId: Math.max(current - step, 0),
    currentId: current,
    nextId: current + step,

    prev: padFrame(Math.max(current - step, 0)),
    current: padFrame(current),
    next: padFrame(current + step),
  };
}

export function secondsToTimecode(seconds) {
  const value = Number(seconds);

  if (Number.isNaN(value) || value < 0) {
    return "00:00.00";
  }

  const minutes = Math.floor(value / 60);
  const remainSeconds = value % 60;

  return `${String(minutes).padStart(2, "0")}:${remainSeconds
    .toFixed(2)
    .padStart(5, "0")}`;
}

export function buildVideoTimestampUrl(videoUrl, timestamp) {
  if (!videoUrl || videoUrl === "#") {
    return "#";
  }

  const time = Math.max(Number(timestamp) || 0, 0);
  return `${videoUrl}#t=${time.toFixed(2)}`;
}