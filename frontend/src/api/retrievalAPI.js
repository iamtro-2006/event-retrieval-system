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

function explicitQueryPlan(query) {
  const pieces = String(query || "").split(/\b(AND|THEN)\b/);
  const clauses = [String(pieces[0] || "").trim()];
  const connectors = [];
  const events = [[]];
  if (clauses[0]) events[0].push(clauses[0]);
  for (let index = 1; index < pieces.length; index += 2) {
    const connector = pieces[index];
    const clause = String(pieces[index + 1] || "").trim();
    connectors.push(connector);
    clauses.push(clause);
    if (connector === "THEN") events.push([]);
    if (clause) events.at(-1).push(clause);
  }
  return { clauses, connectors, events: events.filter((event) => event.length) };
}

function logQueryInput(label, requestId, { query, mode, translate, multimodalClauses }) {
  if (import.meta.env.VITE_DEBUG_API_RESPONSES === "false") return;
  const plan = explicitQueryPlan(query);
  console.groupCollapsed(`[QUERY ${label} ${requestId}] explicit plan`);
  console.log("original_query", query);
  console.log("mode_requested", mode);
  console.log("connectors", plan.connectors);
  console.log("clauses_before_translation", plan.clauses);
  console.log("events_before_translation", plan.events);
  console.log("translation", {
    enabled: Boolean(translate),
    requests: translate ? 1 : 0,
    batch_size: translate ? plan.clauses.filter(Boolean).length : 0,
  });
  console.log("reasoning", "disconnected");
  if (multimodalClauses) console.table(multimodalClauses);
  console.groupEnd();
}

// Temporary diagnostic trace for the search contract.  Keep this at the API
// boundary so every POST response can be inspected before normalization and
// compared with the shape consumed by the UI (frame vs sequence, including
// ASR).  Disable with VITE_DEBUG_API_RESPONSES=false.
function logApiResponse(label, requestId, data) {
  if (import.meta.env.VITE_DEBUG_API_RESPONSES === "false") return;
  const results = Array.isArray(data?.results) ? data.results : [];
  console.groupCollapsed(`[API ${label} ${requestId}] response`);
  console.log("meta", {
    mode: data?.mode ?? data?.search_mode,
    count: data?.count,
    temporal: data?.temporal,
    use_asr: data?.use_asr,
    use_ocr: data?.use_ocr,
    latency_ms: data?.latency_ms,
  });
  console.table(results.map((item, index) => ({
    index,
    video_id: item?.video_id,
    frame_id: item?.frame_id,
    rank: item?.rank,
    matched_sequence: Array.isArray(item?.matched_sequence) ? item.matched_sequence.length : 0,
    temporal_start: item?.temporal?.start_time ?? item?.temporal_start_time,
    temporal_end: item?.temporal?.end_time ?? item?.temporal_end_time,
    search_mode: item?.search_mode,
    matched_texts: Array.isArray(item?.matched_texts) ? item.matched_texts.join(" | ") : "",
  })));
  console.log("raw", data);
  console.groupCollapsed("diagnostic: explicit query plan");
  console.log("original_query", data?.original_query ?? "not returned");
  console.log("translated_query", data?.translated_query ?? "not translated");
  console.log("effective_mode", data?.mode ?? data?.search_mode ?? "not returned");
  console.log("weights", data?.weights ?? data?.debug?.weights ?? "not returned");
  console.log("events", data?.events ?? data?.query_plan?.events ?? data?.debug?.events ?? "not returned");
  console.log("event_queries", data?.event_queries ?? data?.debug?.event_queries ?? "not returned");
  console.log("candidate_pool", {
    candidate_k: data?.candidate_k ?? data?.debug?.candidate_k ?? "not returned",
    universe_size: data?.candidate_universe_size ?? data?.debug?.candidate_universe_size ?? "not returned",
    per_event: data?.event_candidate_counts ?? data?.debug?.event_candidate_counts ?? "not returned",
  });
  console.log("focused_queries", data?.focused_queries ?? data?.debug?.focused_queries ?? "not returned");
  console.log("reasoning", "disconnected");
  console.groupEnd();
  console.groupEnd();
}

function assertModelIsolation(data, requestedModelKey, label) {
  if (!requestedModelKey) return;

  const responseModelKey = String(data?.model_key || "");
  if (responseModelKey !== requestedModelKey) {
    throw new Error(
      `${label}: backend trả model '${responseModelKey || "không xác định"}' ` +
      `thay vì model đã chọn '${requestedModelKey}'. Hãy restart cả backend và frontend.`
    );
  }

  const mismatched = (Array.isArray(data?.results) ? data.results : []).find((item) => {
    const actual = String(item?.model_key || item?.raw?.model_key || "");
    return actual !== requestedModelKey;
  });
  if (mismatched) {
    const actual = mismatched?.model_key || mismatched?.raw?.model_key || "không xác định";
    throw new Error(
      `${label}: phát hiện kết quả từ model '${actual}' trong nhánh '${requestedModelKey}'.`
    );
  }
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

export async function getAvailableModels() {
  const response = await fetch(apiUrl("/api/search/models"), {
    headers: NGROK_HEADER,
  });

  if (!response.ok) {
    throw new Error("Cannot load available models");
  }

  const data = await response.json();
  return Array.isArray(data.models) ? data.models : [];
}

export async function searchRetrieval({
  query,
  topK = 20,
  candidateMultiplier,
  useSplit = true,
  useTranslate = true,
  searchMode = "semantic",
  modelKey,
  durationLimit = -1,
  reasoning = false,
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
    reasoning,
    use_translate: useTranslate,
    translate_api_key: import.meta.env.VITE_GOOGLE_TRANSLATE,
    search_mode: searchMode,
    model_key: modelKey,
    duration_limit: durationLimit,
  };

  const t0 = performance.now();

  logQueryInput("search", requestId, {
    query,
    mode: searchMode,
    translate: useTranslate,
  });
  console.log(`[SEARCH ${requestId}] payload`, { ...payload, translate_api_key: payload.translate_api_key ? "[configured]" : null });

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

    logApiResponse("search", requestId, data);
    if (["semantic", "temporal", "auto"].includes(searchMode)) {
      assertModelIsolation(data, modelKey, "Search");
    }

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

export async function searchMultimodalRetrieval({
  query = "",
  clauses = [],
  queryConnectors = [],
  clauseImages = [],
  topK = 20,
  candidateMultiplier,
  useSplit = true,
  useTranslate = true,
  searchMode = "semantic",
  durationLimit = -1,
  modelKey,
}) {
  if (activeSearchController) activeSearchController.abort();
  const controller = new AbortController();
  activeSearchController = controller;
  const requestId = ++activeSearchRequestId;
  const formData = new FormData();
  const flatImages = [];
  const clausePayload = clauses.map((text, clauseIndex) => {
    const imageIndices = [];
    for (const attachment of clauseImages[clauseIndex] ?? []) {
      imageIndices.push(flatImages.length);
      flatImages.push(attachment.file);
    }
    return {
      text: String(text || "").trim(),
      image_indices: imageIndices,
      connector_before: clauseIndex > 0 ? (queryConnectors[clauseIndex - 1] || "AND") : null,
    };
  });

  formData.append("request_json", JSON.stringify({
    query,
    clauses: clausePayload,
    top_k: topK,
    candidate_multiplier: candidateMultiplier,
    use_split: useSplit,
    use_translate: useTranslate,
    translate_api_key: import.meta.env.VITE_GOOGLE_TRANSLATE,
    search_mode: searchMode,
    duration_limit: durationLimit,
    model_key: modelKey,
  }));
  flatImages.forEach((file) => formData.append("images", file, file.name || "query-image"));

  const t0 = performance.now();
  let eventIndex = 0;
  const multimodalDiagnostic = clausePayload.flatMap((clause, clauseIndex) => {
    if (clauseIndex > 0 && clause.connector_before === "THEN") eventIndex += 1;
    const rows = [{
      event: eventIndex + 1,
      position: clauseIndex + 1,
      connector_before: clause.connector_before || "START",
      type: "text",
      value: clause.text || "(empty)",
    }];
    clause.image_indices.forEach((imageIndex) => rows.push({
      event: eventIndex + 1,
      position: clauseIndex + 1,
      connector_before: "same clause",
      type: "image",
      value: `image-${imageIndex + 1}`,
    }));
    return rows;
  });
  logQueryInput("multimodal", requestId, {
    query,
    mode: searchMode,
    translate: useTranslate,
    multimodalClauses: multimodalDiagnostic,
  });
  try {
    const response = await fetch(apiUrl("/api/search/multimodal"), {
      method: "POST",
      headers: NGROK_HEADER,
      body: formData,
      signal: controller.signal,
    });
    if (requestId !== activeSearchRequestId) throw createStaleSearchError(requestId);
    if (!response.ok) {
      let message = await response.text();
      try { message = JSON.parse(message)?.detail || message; } catch { /* keep response text */ }
      throw new Error(message || "Multimodal search failed");
    }
    const data = await response.json();
    if (requestId !== activeSearchRequestId) throw createStaleSearchError(requestId);
    assertModelIsolation(data, modelKey, "Multimodal search");
    logApiResponse("multimodal", requestId, data);
    const results = normalizeResults(data.results ?? []);
    return {
      query: data.query || query || "Image query",
      subQueries: data.sub_queries ?? [],
      latencyMs: data.latency_ms ?? Math.round(performance.now() - t0),
      count: data.count ?? results.length,
      searchMode: data.search_mode ?? searchMode,
      durationLimit: data.duration_limit ?? durationLimit,
      results,
    };
  } finally {
    if (activeSearchController === controller) activeSearchController = null;
  }
}

function createStaleSearchError(requestId) {
  const error = new Error(`Stale search ignored: ${requestId}`);
  error.name = "StaleSearchError";
  return error;
}

export async function searchFusion({
  query,
  topK = 20,
  candidateMultiplier,
  useSplit = true,
  useTranslate = true,
  fusionConfig,
  reasoning = false,
}) {
  if (activeSearchController) {
    activeSearchController.abort();
  }

  const controller = new AbortController();
  activeSearchController = controller;

  const requestId = ++activeSearchRequestId;

  const semanticModels = (fusionConfig?.semanticModels ?? []).map((m) => m.key);

  const payload = {
    query,
    semantic_models: semanticModels,
    temporal: Boolean(fusionConfig?.temporal),
    use_ocr: Boolean(fusionConfig?.useOcr),
    use_asr: Boolean(fusionConfig?.useAsr),
    top_k: topK,
    candidate_multiplier: candidateMultiplier,
    use_split: useSplit,
    weights: fusionConfig?.weights ?? undefined,
    semantic_lambda: Number(fusionConfig?.semanticLambda ?? 0.5),
    reasoning,
    use_translate: useTranslate,
    translate_api_key: import.meta.env.VITE_GOOGLE_TRANSLATE,
    duration_limit: fusionConfig?.temporal ? Number(fusionConfig?.durationLimit ?? -1) : -1,
  };

  const t0 = performance.now();
  logQueryInput("fusion", requestId, {
    query,
    mode: fusionConfig?.temporal ? "temporal fusion" : "fusion",
    translate: useTranslate,
  });
  console.log(`[FUSION SEARCH ${requestId}] payload`, { ...payload, translate_api_key: payload.translate_api_key ? "[configured]" : null });
  console.groupCollapsed(`[FUSION SEARCH ${requestId}] source configuration`);
  console.log("reasoning", "disconnected");
  console.log("weights sent", payload.weights);
  console.log("enabled sources", { semantic: semanticModels.length > 0, ocr: payload.use_ocr, asr: payload.use_asr, temporal: payload.temporal });
  console.groupEnd();

  try {
    const response = await fetch(apiUrl("/api/search/fusion"), {
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
      throw new Error(errorText || "Fusion search request failed");
    }

    const data = await response.json();
    const t2 = performance.now();

    logApiResponse("fusion", requestId, data);

    if (requestId !== activeSearchRequestId) {
      throw createStaleSearchError(requestId);
    }

    const normalizedResults = normalizeResults(data.results ?? []);
    const t3 = performance.now();

    if (requestId !== activeSearchRequestId) {
      throw createStaleSearchError(requestId);
    }

    const backendMs = Number(data.latency_ms ?? 0);

    console.table({
      requestId,
      mode: "fusion",
      backendMs,
      fetchMs: Number((t1 - t0).toFixed(2)),
      jsonMs: Number((t2 - t1).toFixed(2)),
      normalizeMs: Number((t3 - t2).toFixed(2)),
      apiTotalMs: Number((t3 - t0).toFixed(2)),
      resultCount: normalizedResults.length,
    });

    return {
      query: data.query,
      originalQuery: data.original_query,
      translatedQuery: data.translated_query,
      useTranslate: data.use_translate,
      useSplit: data.use_split,
      latencyMs: data.latency_ms ?? null,
      count: data.count ?? 0,
      searchMode: "fusion",
      durationLimit: data.duration_limit ?? -1,
      reasoning: Boolean(data.reasoning),
      weights: data.weights ?? {},
      events: data.events ?? [],
      eventQueries: data.event_queries ?? [],
      focusedQueries: data.focused_queries ?? {},
      results: normalizedResults,
    };
  } catch (error) {
    if (error.name === "AbortError") {
      console.log(`[FUSION SEARCH ${requestId}] aborted`);
    } else if (error.name === "StaleSearchError") {
      console.log(`[FUSION SEARCH ${requestId}] stale ignored`);
    }

    throw error;
  } finally {
    if (activeSearchController === controller) {
      activeSearchController = null;
    }
  }
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
      frame_idx: safeNumber(item.frame_idx ?? raw.frame_idx, Number.isFinite(frameId) ? frameId : 0),
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
      model_key: item.model_key || raw.model_key || "",
      search_mode: item.search_mode || raw.search_mode || "",
      source_models: item.source_models ?? raw.source_models ?? [],
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
  modelKey,
}) {
  const payload = {
    video_id: videoId,
    frame_id: Number(frameId),
    top_k: topK,
    model_key: modelKey,
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
  assertModelIsolation(data, modelKey, "Similarity search");
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

export async function getVideoPreview(videoId, { frameId, timestampMs } = {}) {
  const params = new URLSearchParams({ video_id: videoId });
  if (frameId !== undefined && frameId !== "") params.set("frame_id", String(frameId));
  if (timestampMs !== undefined && timestampMs !== "") params.set("timestamp_ms", String(timestampMs));
  const response = await fetch(apiUrl(`/api/video-preview?${params}`), { headers: NGROK_HEADER });
  if (!response.ok) throw new Error((await response.text()) || "Cannot load video preview");
  return normalizeResults([await response.json()])[0];
}

export async function getVideoKeyframes(videoId) {
  const response = await fetch(apiUrl(`/api/video-keyframes?video_id=${encodeURIComponent(videoId)}`), { headers: NGROK_HEADER });
  if (!response.ok) throw new Error(`Cannot load video keyframes (${response.status})`);
  return response.json();
}

export async function getFrameIdxAtTimestamp(videoId, timestampMs) {
  const result = await getVideoPreview(videoId, { timestampMs });
  return Number(result.frame_idx ?? result.raw?.frame_idx ?? 0);
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
