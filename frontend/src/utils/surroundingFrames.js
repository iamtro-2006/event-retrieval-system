export function buildSurroundingFrames(result, radius = 5) {
  if (!result?.image_url || result.image_url === "#") {
    return [];
  }

  const match = result.image_url.match(/(.*\/)(\d+)(\.[a-zA-Z0-9]+)$/);

  if (!match) {
    return [];
  }

  const [, baseUrl, frameNumberText, ext] = match;
  const currentFrame = Number(frameNumberText);
  const padding = frameNumberText.length;

  if (!Number.isFinite(currentFrame)) {
    return [];
  }

  const frames = [];

  for (let offset = -radius; offset <= radius; offset += 1) {
    const frameId = Math.max(currentFrame + offset, 0);
    const frameName = `${String(frameId).padStart(padding, "0")}${ext}`;

    frames.push({
      offset,
      frameId,
      frameName,
      imageUrl: `${baseUrl}${frameName}`,
      isCurrent: offset === 0,
    });
  }

  return frames;
}