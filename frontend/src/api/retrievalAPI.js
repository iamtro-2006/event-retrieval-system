const RAW_API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

const API_BASE_URL = RAW_API_BASE_URL.replace(/\/+$/, "");

const NGROK_HEADER = { "ngrok-skip-browser-warning": "true" };

function apiUrl(path) {
  return `${API_BASE_URL}/${String(path).replace(/^\/+/, "")}`;
}

export async function checkBackendHealth() {
  const response = await fetch(apiUrl("/api/health"), {
    headers: NGROK_HEADER,
  });

  if (!response.ok) {
    throw new Error("Backend health check failed");
  }

  return response.json();
}

export async function getBackendConfig() {
  const response = await fetch(apiUrl("/api/config"), {
    headers: NGROK_HEADER,
  });

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
  searchMode = "semantic",
  durationLimit = -1,
}) {
  const payload = {
    query,
    top_k: topK,
    candidate_multiplier: candidateMultiplier,
    use_split: useSplit,
    use_translate: useTranslate,
    search_mode: searchMode,
    duration_limit: durationLimit,
  };

  const response = await fetch(apiUrl("/api/search"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...NGROK_HEADER,
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
    searchMode: data.search_mode ?? searchMode,
    durationLimit: data.duration_limit ?? durationLimit,
    results: normalizeResults(data.results ?? []),
  };
}

function normalizeResults(results) {
  return results.map((item, index) => {
    const frameId = Number(item.frame_id ?? 0);
    const similarity = Number(item.similarity ?? 0);
    const raw = item.raw || {};

    const matchedSequence = normalizeMatchedSequence(
      item.matched_sequence ?? raw.matched_sequence ?? []
    );

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
      timestamp: safeNumber(item.timestamp, 0),
      similarity: Number.isFinite(similarity) ? similarity : 0,
      caption: item.caption || "",
      rank: safeNumber(item.rank, index + 1),
      matched_sequence: matchedSequence,
      temporal: {
        video_score: safeNumber(item.temporal?.video_score ?? raw.video_score, 0),
        start_time: safeNumber(
          item.temporal?.start_time ?? raw.temporal_start_time ?? item.timestamp,
          0
        ),
        end_time: safeNumber(
          item.temporal?.end_time ?? raw.temporal_end_time ?? item.timestamp,
          0
        ),
        duration_sec: safeNumber(
          item.temporal?.duration_sec ?? raw.temporal_duration_sec,
          0
        ),
        avg_score: safeNumber(item.temporal?.avg_score ?? raw.avg_score, similarity),
      },
      raw,
    };
  });
}

function normalizeMatchedSequence(sequence) {
  if (!Array.isArray(sequence)) {
    return [];
  }

  return sequence.map((item, index) => {
    const frameId = Number(item.keyframe_id ?? item.frame_id ?? item.frame_idx ?? 0);
    const timestamp = Number(item.timestamp_sec ?? item.timestamp ?? 0);
    const score = Number(item.score ?? item.candidate_score ?? 0);

    return {
      ...item,
      sub_query_idx: safeNumber(item.sub_query_idx, index),
      sub_query: item.sub_query || "",
      keyframe_id: Number.isFinite(frameId) ? frameId : 0,
      frame_id: Number.isFinite(frameId) ? frameId : 0,
      frame_name: `${String(Number.isFinite(frameId) ? frameId : index).padStart(
        6,
        "0"
      )}.jpg`,
      keyframe_path: item.keyframe_path || "",
      image_url: toAbsoluteUrl(item.image_url || makeKeyframeUrl(item.keyframe_path)),
      timestamp_sec: Number.isFinite(timestamp) ? timestamp : 0,
      score: Number.isFinite(score) ? score : 0,
    };
  });
}

function makeKeyframeUrl(keyframePath) {
  if (!keyframePath) {
    return "#";
  }

  const normalizedPath = String(keyframePath).replaceAll("\\", "/");
  const marker = "data/processed/keyframes/";
  const idx = normalizedPath.indexOf(marker);

  if (idx === -1) {
    return "#";
  }

  const rel = normalizedPath.slice(idx + marker.length);

  return `/static/keyframes/${rel}`;
}

function toAbsoluteUrl(url) {
  if (!url || url === "#") {
    return "#";
  }

  if (url.startsWith("http://") || url.startsWith("https://")) {
    return url;
  }

  return apiUrl(url);
}

function safeNumber(value, fallback = 0) {
  const numberValue = Number(value);
  return Number.isFinite(numberValue) ? numberValue : fallback;
}