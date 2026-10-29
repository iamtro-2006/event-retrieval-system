export const KEYFRAME_SLOT_WIDTH = 116;
export const KEYFRAME_CARD_WIDTH = 108;

export function normalizeVideoKeyframes(items = []) {
  const seen = new Set();
  return items.filter((item) => {
    const timestamp = Number(item.timestamp);
    const id = item.keyframe_id;
    if (item.timestamp == null || !Number.isFinite(timestamp) || timestamp < 0 || id == null || seen.has(String(id))) return false;
    seen.add(String(id));
    return true;
  }).map((item) => ({ ...item, keyframe_id: String(item.keyframe_id), timestamp: Number(item.timestamp) }))
    .sort((left, right) => left.timestamp - right.timestamp);
}

// A keyframe stays active until the next keyframe's timestamp, including seeks backwards.
export function findActiveKeyframe(items, seconds) {
  if (!items.length) return -1;
  let low = 0;
  let high = items.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (items[middle].timestamp <= seconds) low = middle + 1;
    else high = middle;
  }
  return Math.max(0, low - 1);
}

export function centeredKeyframeOffset(index, viewportWidth, count) {
  const totalWidth = Math.max(0, count * KEYFRAME_SLOT_WIDTH - (KEYFRAME_SLOT_WIDTH - KEYFRAME_CARD_WIDTH));
  const center = index * KEYFRAME_SLOT_WIDTH + KEYFRAME_CARD_WIDTH / 2;
  return Math.max(0, Math.min(totalWidth - viewportWidth, center - viewportWidth / 2));
}

export function initialVideoMilliseconds(result) {
  const playbackStart = result?.playback_start ?? result?.raw?.playback_start;
  if (playbackStart != null && Number.isFinite(Number(playbackStart))) {
    return Math.max(0, Math.round(Number(playbackStart) * 1000));
  }
  const fps = Number(result?.fps ?? result?.raw?.fps);
  if (result?.is_preview_frame && fps > 0 && result?.frame_idx != null) {
    return Math.max(0, Math.round(Number(result.frame_idx) / fps * 1000));
  }
  return Math.max(0, Math.round(Number(result?.timestamp || 0) * 1000));
}
