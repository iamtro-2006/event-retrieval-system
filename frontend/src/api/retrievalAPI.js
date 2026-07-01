const RAW_API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

const API_BASE_URL = RAW_API_BASE_URL.replace(/\/+$/, "");

const KEYFRAMES_BASE_URL = (
  import.meta.env.VITE_KEYFRAMES_BASE_URL || ""
).replace(/\/+$/, "");

const VIDEOS_BASE_URL = (
  import.meta.env.VITE_VIDEOS_BASE_URL || ""
).replace(/\/+$/, "");

const MAP_KEYFRAMES_BASE_URL = (
  import.meta.env.VITE_MAP_KEYFRAMES_BASE_URL || ""
).replace(/\/+$/, "");

const NGROK_HEADER = { "ngrok-skip-browser-warning": "true" };

let activeSearchController = null;
let activeSearchRequestId = 0;

function apiUrl(path) {
  return `${API_BASE_URL}/${String(path).replace(/^\/+/, "")}`;
}

function joinBaseUrl(baseUrl, relPath) {
  if (!baseUrl || !relPath) {
    return "";
  }

  const cleanedRelPath = String(relPath)
    .replaceAll("\\", "/")
    .replace(/^\/+/, "");

  return `${baseUrl}/${cleanedRelPath}`;
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
  if (activeSearchController) {
    activeSearchController.abort();
  }

  const controller = new AbortController();
  activeSearchController = controller;

  const requestId = ++activeSearchRequestId;

  const payload = {
    query,
    top_k: topK,
    candidate_multiplier: candidateMultiplier,
    use_split: useSplit,
    use_translate: useTranslate,
    search_mode: searchMode,
    duration_limit: durationLimit,
  };

  const t0 = performance.now();

  console.log(`[SEARCH ${requestId}] payload`, payload);

  try {
    const response = await fetch(apiUrl("/api/search"), {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...NGROK_HEADER,
      },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });

    const t1 = performance.now();

    if (requestId !== activeSearchRequestId) {
      throw createStaleSearchError(requestId);
    }

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(errorText || "Search request failed");
    }

    const data = await response.json();
    const t2 = performance.now();

    if (requestId !== activeSearchRequestId) {
      throw createStaleSearchError(requestId);
    }

    const normalizedResults = normalizeResults(data.results ?? []);
    const t3 = performance.now();

    if (requestId !== activeSearchRequestId) {
      throw createStaleSearchError(requestId);
    }

    const backendMs = Number(data.latency_ms ?? 0);

    const timing = {
      requestId,
      backendMs,
      fetchMs: Number((t1 - t0).toFixed(2)),
      jsonMs: Number((t2 - t1).toFixed(2)),
      normalizeMs: Number((t3 - t2).toFixed(2)),
      apiTotalMs: Number((t3 - t0).toFixed(2)),
      overheadMs: Number(((t3 - t0) - backendMs).toFixed(2)),
      resultCount: normalizedResults.length,
    };

    console.log(`[SEARCH ${requestId}] done`);
    console.table(timing);

    return {
      query: data.query,
      originalQuery: data.original_query,
      translatedQuery: data.translated_query,
      useTranslate: data.use_translate,
      useSplit: data.use_split,
      subQueries: data.sub_queries ?? [],
      events: data.events ?? [],
      eventQueries: data.event_queries ?? [],
      latencyMs: data.latency_ms ?? null,
      timing,
      count: data.count ?? 0,
      searchMode: data.search_mode ?? searchMode,
      durationLimit: data.duration_limit ?? durationLimit,
      results: normalizedResults,
    };
  } catch (error) {
    if (error.name === "AbortError") {
      console.log(`[SEARCH ${requestId}] aborted`);
    } else if (error.name === "StaleSearchError") {
      console.log(`[SEARCH ${requestId}] stale ignored`);
    }

    throw error;
  } finally {
    if (activeSearchController === controller) {
      activeSearchController = null;
    }
  }
}

function createStaleSearchError(requestId) {
  const error = new Error(`Stale search ignored: ${requestId}`);
  error.name = "StaleSearchError";
  return error;
}

function normalizeResults(results) {
  if (!Array.isArray(results)) {
    return [];
  }

  return results.map((item, index) => {
    const frameId = Number(item.frame_id ?? 0);
    const similarity = Number(item.similarity ?? 0);
    const raw = item.raw || {};

    const imageUrl =
      joinBaseUrl(KEYFRAMES_BASE_URL, item.image_rel_path) ||
      toAbsoluteUrl(item.image_url || makeKeyframeUrl(item.keyframe_path));

    const videoUrl =
      joinBaseUrl(VIDEOS_BASE_URL, item.video_rel_path) ||
      toAbsoluteUrl(item.video_url);

    const mapUrl =
      joinBaseUrl(MAP_KEYFRAMES_BASE_URL, item.map_rel_path) ||
      toAbsoluteUrl(item.map_url ?? raw.map_url);

      
    const matchedSequence = normalizeMatchedSequence(
      item.matched_sequence ?? raw.matched_sequence ?? []
    );

    const baseId =
      item.id ||
      `${item.video_id || "unknown_video"}_${String(
        Number.isFinite(frameId) ? frameId : index
      ).padStart(6, "0")}`;
    /*
    console.log("KEYFRAMES_BASE_URL =", KEYFRAMES_BASE_URL);
    console.log("image_rel_path =", item.image_rel_path);
    console.log(
      "resolved =",
      joinBaseUrl(KEYFRAMES_BASE_URL, item.image_rel_path)
    );
    */
      return {
      id: `${baseId}-${index}`,
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
      image_rel_path: item.image_rel_path || "",
      video_rel_path: item.video_rel_path || "",
      map_rel_path: item.map_rel_path || "",
      image_url: imageUrl,
      video_url: videoUrl,
      map_url: mapUrl,
      timestamp: safeNumber(item.timestamp, 0),
      similarity: Number.isFinite(similarity) ? similarity : 0,
      caption: item.caption || "",
      rank: safeNumber(item.rank, index + 1),
      matched_sequence: matchedSequence,
      // OCR/ASR-only: the on-screen text or transcript snippet that matched
      // the query. Empty for semantic/temporal results.
      matched_texts: Array.isArray(item.matched_texts ?? raw.matched_texts)
        ? (item.matched_texts ?? raw.matched_texts)
        : [],
      ocr_score: item.ocr_score ?? raw.ocr_score ?? null,
      asr_score: item.asr_score ?? raw.asr_score ?? null,
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

    const imageUrl =
      joinBaseUrl(KEYFRAMES_BASE_URL, item.image_rel_path) ||
      toAbsoluteUrl(item.image_url || makeKeyframeUrl(item.keyframe_path));

    const baseId = `${item.video_id || "unknown_video"}_${String(
      Number.isFinite(frameId) ? frameId : index
    ).padStart(6, "0")}`;

    return {
      ...item,
      id: `${baseId}-${index}`,
      sub_query_idx: safeNumber(item.sub_query_idx, index),
      sub_query: item.sub_query || "",
      keyframe_id: Number.isFinite(frameId) ? frameId : 0,
      frame_id: Number.isFinite(frameId) ? frameId : 0,
      frame_name: `${String(Number.isFinite(frameId) ? frameId : index).padStart(
        6,
        "0"
      )}.jpg`,
      keyframe_path: item.keyframe_path || "",
      image_rel_path: item.image_rel_path || "",
      image_url: imageUrl,
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

export async function getSurroundingFrames(videoId, keyframeId, radius = 10) {
  const params = new URLSearchParams({
    video_id: videoId,
    keyframe_id: String(keyframeId),
    radius: String(radius),
  });

  const response = await fetch(apiUrl(`/api/surrounding-frames?${params}`), {
    headers: NGROK_HEADER,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || "Cannot load surrounding frames");
  }

  const data = await response.json();

  return normalizeResults(data.frames ?? []);
}

export async function similaritySearch({
  videoId,
  frameId,
  topK = 20,
}) {
  const payload = {
    video_id: videoId,
    frame_id: Number(frameId),
    top_k: topK,
  };

  const t0 = performance.now();

  const response = await fetch(apiUrl("/api/similarity-search"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...NGROK_HEADER,
    },
    body: JSON.stringify(payload),
  });

  const t1 = performance.now();

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText || "Similarity search failed");
  }

  const data = await response.json();
  const normalizedResults = normalizeResults(data.results ?? []);

  console.table({
    mode: "similarity",
    backendMs: data.latency_ms,
    fetchMs: Number((t1 - t0).toFixed(2)),
    resultCount: normalizedResults.length,
  });

  return {
    query: data.query,
    source: data.source,
    latencyMs: data.latency_ms ?? null,
    count: data.count ?? 0,
    searchMode: "similarity",
    results: normalizedResults,
  };
}

export async function getFrameInfo(videoId, keyframeId) {
  const params = new URLSearchParams({
    video_id: videoId,
    keyframe_id: String(keyframeId),
  });

  const response = await fetch(apiUrl(`/api/frame-info?${params}`), {
    headers: NGROK_HEADER,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || "Cannot load frame info");
  }

  const data = await response.json();

  return normalizeResults([data])[0];
}

export async function transcribeSpeech(blob) {
  const formData = new FormData();
  formData.append("file", blob, "speech.webm");

  const response = await fetch(apiUrl("/api/speech/transcribe"), {
    method: "POST",
    headers: {
      ...NGROK_HEADER,
    },
    body: formData,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || "Speech transcription failed");
  }

  return response.json();
}
// ─────────────────────────────────────────────────────────────────────────────
// RERANK API — gửi danh sách kết quả hiện tại lên backend để VLM chấm điểm lại
// ─────────────────────────────────────────────────────────────────────────────
export async function rerankResults({
  results,
  query,
  searchMode = "semantic",
  topCandidate = 1.0,
  topK = 20,
}) {
  const payload = {
    results,
    query,
    search_mode: searchMode,
    top_candidate: topCandidate,
    top_k: topK,
  };

  const t0 = performance.now();
  console.log("[RERANK] payload", { query, searchMode, topK, topCandidate, count: results.length });

  const response = await fetch(apiUrl("/api/rerank"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...NGROK_HEADER,
    },
    body: JSON.stringify(payload),
  });

  const t1 = performance.now();

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText || "Rerank request failed");
  }

  const data = await response.json();
  const normalizedResults = normalizeResults(data.results ?? []);

  console.table({
    mode: "rerank",
    backendMs: data.latency_ms,
    fetchMs: Number((t1 - t0).toFixed(2)),
    resultCount: normalizedResults.length,
  });

  return {
    query: data.query,
    latencyMs: data.latency_ms ?? null,
    count: data.count ?? 0,
    results: normalizedResults,
  };
}
