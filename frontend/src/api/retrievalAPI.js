const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

export async function checkBackendHealth() {
  const response = await fetch(`${API_BASE_URL}/api/health`);

  if (!response.ok) {
    throw new Error("Backend health check failed");
  }

  return response.json();
}

export async function getBackendConfig() {
  const response = await fetch(`${API_BASE_URL}/api/config`);

  if (!response.ok) {
    throw new Error("Cannot load backend config");
  }

  return response.json();
}

export async function searchRetrieval({
  query,
  topK = 20,
  candidateMultiplier,
  useSplit = true,
  useTranslate = true,
}) {
  const payload = {
    query,
    top_k: topK,
    candidate_multiplier: candidateMultiplier,
    use_split: useSplit,
    use_translate: useTranslate,
  };

  const response = await fetch(`${API_BASE_URL}/api/search`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText || "Search request failed");
  }

  const data = await response.json();

  return {
    query: data.query,
    originalQuery: data.original_query,
    translatedQuery: data.translated_query,
    useTranslate: data.use_translate,
    useSplit: data.use_split,
    subQueries: data.sub_queries ?? [],
    latencyMs: data.latency_ms ?? null,
    count: data.count ?? 0,
    results: normalizeResults(data.results ?? []),
  };
}

function normalizeResults(results) {
  return results.map((item, index) => {
    const frameId = Number(item.frame_id ?? 0);
    const similarity = Number(item.similarity ?? 0);

    return {
      id: item.id || `result-${index}`,
      video_id: item.video_id || "unknown_video",
      frame_id: Number.isFinite(frameId) ? frameId : 0,
      frame_name:
        item.frame_name ||
        `${String(Number.isFinite(frameId) ? frameId : index).padStart(
          6,
          "0"
        )}.jpg`,
      path: item.path || "",
      keyframe_path: item.keyframe_path || "",
      image_url: toAbsoluteUrl(item.image_url),
      video_url: toAbsoluteUrl(item.video_url),
      timestamp: Number(item.timestamp ?? 0),
      similarity: Number.isFinite(similarity) ? similarity : 0,
      caption: item.caption || "",
      rank: Number(item.rank ?? index + 1),
      raw: item.raw || {},
    };
  });
}

function toAbsoluteUrl(url) {
  if (!url || url === "#") {
    return "#";
  }

  if (url.startsWith("http://") || url.startsWith("https://")) {
    return url;
  }

  return `${API_BASE_URL}${url}`;
}