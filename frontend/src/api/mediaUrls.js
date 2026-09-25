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

function configuredBase(base, dataset) {
  return cleanBase(base)
    .replaceAll("{dataset}", dataset)
    .replaceAll("{DATASET}", dataset.toUpperCase());
}

function relativePathFromUrl(url, dataset, directory, apiBaseUrl) {
  if (!url || url === "#") return "";
  try {
    const path = new URL(url, `${cleanBase(apiBaseUrl)}/`).pathname;
    const match = path.match(/\/(?:static\/)?(aic|cam)\/(keyframes|videos|map-keyframes)\/(.+)$/i);
    if (match && match[1].toLowerCase() === dataset && match[2].toLowerCase() === directory) return match[3];
  } catch { /* Keep the original URL when it cannot be parsed. */ }
  return "";
}

export function createMediaUrlResolver({ apiBaseUrl, dataBaseUrl = "", sharedBases = {}, datasetBases = {} }) {
  return (dataset, kind, relPath, fallbackUrl) => {
    const collection = String(dataset || "").trim().toLowerCase();
    const directory = MEDIA_DIRECTORIES[kind];
    if (!directory) throw new Error(`Unknown media type: ${kind}`);

    const path = cleanRelativePath(relPath || relativePathFromUrl(fallbackUrl, collection, directory, apiBaseUrl));
    const base = configuredBase(datasetBases[collection]?.[kind] || sharedBases[kind], collection);
    if (base && path) return `${base}/${path}`;
    if (dataBaseUrl && collection && path) {
      return `${cleanBase(dataBaseUrl)}/${collection.toUpperCase()}/${directory}/${path}`;
    }
    if (!fallbackUrl || fallbackUrl === "#") return "#";
    if (/^https?:\/\//i.test(fallbackUrl)) return fallbackUrl;
    return `${cleanBase(apiBaseUrl)}/${cleanRelativePath(fallbackUrl)}`;
  };
}
