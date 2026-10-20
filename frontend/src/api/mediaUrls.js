const MEDIA_DIRECTORIES = {
  keyframes: "keyframes",
  videos: "videos",
  mapKeyframes: "map-keyframes",
};

function cleanBase(base) {
  return String(base || "").trim().replace(/\/+$/, "");
}

function cleanRelativePath(path) {
  return String(path || "").replaceAll("\\", "/").replace(/^\/+/, "");
}

function pathWithinDataset(path, dataset) {
  const relative = cleanRelativePath(path);
  const [first, ...rest] = relative.split("/");
  return first.toLowerCase() === dataset ? rest.join("/") : relative;
}

function relativePathFromUrl(url, dataset, directory, apiBaseUrl) {
  if (!url || url === "#") return "";
  try {
    const parts = new URL(url, `${cleanBase(apiBaseUrl)}/`).pathname.split("/").filter(Boolean);
    const directories = directory === "map-keyframes" ? [directory, "mapkeyframes", "map_keyframes"] : [directory];
    for (let i = 0; i < parts.length - 2; i += 1) {
      const left = parts[i].toLowerCase();
      const right = parts[i + 1].toLowerCase();
      if ((left === dataset && directories.includes(right)) || (directories.includes(left) && right === dataset)) {
        return parts.slice(i + 2).join("/");
      }
    }
  } catch { /* Keep the original URL when it cannot be parsed. */ }
  return "";
}

export function createMediaUrlResolver({ apiBaseUrl, mediaBases = {} }) {
  return (dataset, kind, relPath, fallbackUrl) => {
    const collection = String(dataset || "").trim().toLowerCase();
    const directory = MEDIA_DIRECTORIES[kind];
    if (!directory) throw new Error(`Unknown media type: ${kind}`);

    const path = pathWithinDataset(relPath || relativePathFromUrl(fallbackUrl, collection, directory, apiBaseUrl), collection);
    const base = cleanBase(mediaBases[kind]);
    if (base && collection && path) return `${base}/${collection.toUpperCase()}/${path}`;
    if (!fallbackUrl || fallbackUrl === "#") return "#";
    if (/^https?:\/\//i.test(fallbackUrl)) return fallbackUrl;
    return `${cleanBase(apiBaseUrl)}/${cleanRelativePath(fallbackUrl)}`;
  };
}
