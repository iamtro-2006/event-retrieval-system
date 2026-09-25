import { useRef, useState } from "react";
import { useVideoFilter } from "../components/VideoFilter";
import { searchRetrieval, searchFusion, searchMultimodalRetrieval, similaritySearch, searchColorRetrieval } from "../api/retrievalAPI";

export function useRetrievalSearch() {
  const { activeIds } = useVideoFilter();
  const requestIdRef = useRef(0);

  const [results, setResults] = useState([]);
  const [latency, setLatency] = useState(null);
  const [subQueries, setSubQueries] = useState([]);
  const [count, setCount] = useState(0);
  const [lastQuery, setLastQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [searchMode, setSearchMode] = useState("semantic");
  const [durationLimit, setDurationLimit] = useState(-1);

  async function search({
    dataset,
    query,
    topK = 20,
    candidateMultiplier,
    useSplit = true,
    useTranslate = true,
    searchMode: requestedSearchMode = "semantic",
    modelKey,
    durationLimit: requestedDurationLimit = -1,
    reasoning = false,
    videoIds = activeIds,
  }) {
    const cleanQuery = typeof query === "string" ? query.trim() : "";

    if (!cleanQuery) {
      return;
    }

    const requestId = ++requestIdRef.current;

    setLoading(true);
    setError("");

    try {
      const data = await searchRetrieval({
        dataset,
        videoIds,
        query: cleanQuery,
        topK,
        candidateMultiplier,
        useSplit,
        useTranslate,
        searchMode: requestedSearchMode,
        modelKey,
        durationLimit: requestedDurationLimit,
        reasoning,
      });

      if (requestId !== requestIdRef.current) {
        return;
      }

      setResults(data.results ?? []);
      setLatency(data.latencyMs ?? null);
      setSubQueries(data.subQueries ?? []);
      setCount(data.count ?? 0);
      setLastQuery(data.query ?? cleanQuery);
      setSearchMode(data.searchMode ?? requestedSearchMode);
      setDurationLimit(data.durationLimit ?? requestedDurationLimit);
    } catch (err) {
      if (err.name === "AbortError" || err.name === "StaleSearchError") {
        return;
      }

      if (requestId !== requestIdRef.current) {
        return;
      }

      setResults([]);
      setLatency(null);
      setSubQueries([]);
      setCount(0);
      setError(err.message || "Search failed");
    } finally {
      if (requestId === requestIdRef.current) {
        setLoading(false);
      }
    }
  }

  async function searchWithFusion({
    dataset,
    query,
    topK = 20,
    candidateMultiplier,
    useSplit = true,
    useTranslate = true,
    fusionConfig,
    reasoning = false,
  }) {
    const cleanQuery = typeof query === "string" ? query.trim() : "";

    if (!cleanQuery) {
      return;
    }

    const requestId = ++requestIdRef.current;

    setLoading(true);
    setError("");

    try {
      const data = await searchFusion({
        dataset,
        videoIds: activeIds,
        query: cleanQuery,
        topK,
        candidateMultiplier,
        useSplit,
        useTranslate,
        fusionConfig,
        reasoning,
      });

      if (requestId !== requestIdRef.current) {
        return;
      }

      setResults(data.results ?? []);
      setLatency(data.latencyMs ?? null);
      setSubQueries([]);
      setCount(data.count ?? 0);
      setLastQuery(data.query ?? cleanQuery);
      setSearchMode("fusion");
      setDurationLimit(data.durationLimit ?? -1);
    } catch (err) {
      if (err.name === "AbortError" || err.name === "StaleSearchError") {
        return;
      }

      if (requestId !== requestIdRef.current) {
        return;
      }

      setResults([]);
      setLatency(null);
      setSubQueries([]);
      setCount(0);
      setError(err.message || "Fusion search failed");
    } finally {
      if (requestId === requestIdRef.current) {
        setLoading(false);
      }
    }
  }

  async function searchMultimodal(options) {
    const requestId = ++requestIdRef.current;
    setLoading(true);
    setError("");
    try {
      const data = await searchMultimodalRetrieval({ ...options, videoIds: activeIds });
      if (requestId !== requestIdRef.current) return;
      setResults(data.results ?? []);
      setLatency(data.latencyMs ?? null);
      setSubQueries(data.subQueries ?? []);
      setCount(data.count ?? 0);
      setLastQuery(data.query ?? "Image query");
      setSearchMode(data.searchMode ?? options.searchMode);
      setDurationLimit(data.durationLimit ?? options.durationLimit ?? -1);
    } catch (err) {
      if (err.name === "AbortError" || err.name === "StaleSearchError") return;
      if (requestId !== requestIdRef.current) return;
      setResults([]);
      setLatency(null);
      setSubQueries([]);
      setCount(0);
      setError(err.message || "Multimodal search failed");
      throw err;
    } finally {
      if (requestId === requestIdRef.current) setLoading(false);
    }
  }

  async function searchSimilar({
    dataset,
    videoId,
    frameId,
    topK = 20,
  }) {
    const requestId = ++requestIdRef.current;

    setLoading(true);
    setError("");

    try {
      const data = await similaritySearch({
        dataset,
        videoIds: activeIds,
        videoId,
        frameId,
        topK,
      });

      if (requestId !== requestIdRef.current) {
        return;
      }

      setResults(data.results ?? []);
      setLatency(data.latencyMs ?? null);
      setSubQueries([]);
      setCount(data.count ?? 0);
      setLastQuery(data.query ?? `similarity:${videoId}/${frameId}`);
      setSearchMode("similarity");
      setDurationLimit(-1);
    } catch (err) {
      if (requestId !== requestIdRef.current) {
        return;
      }

      setResults([]);
      setLatency(null);
      setSubQueries([]);
      setCount(0);
      setError(err.message || "Similarity search failed");
    } finally {
      if (requestId === requestIdRef.current) {
        setLoading(false);
      }
    }
  }

  async function searchByColor({ dataset, cells, topK = 20 }) {
    if (!cells?.length) return;
    const requestId = ++requestIdRef.current;
    setLoading(true); setError("");
    try {
      const data = await searchColorRetrieval({ dataset, cells, topK, videoIds: activeIds });
      if (requestId !== requestIdRef.current) return;
      setResults(data.results ?? []);
      setLatency(data.latencyMs ?? null);
      setSubQueries([]);
      setCount(data.count ?? 0);
      setLastQuery(data.query);
      setSearchMode("color");
      setDurationLimit(-1);
    } catch (err) {
      if (err.name === "AbortError" || err.name === "StaleSearchError" || requestId !== requestIdRef.current) return;
      setResults([]); setLatency(null); setCount(0); setError(err.message || "Color search failed");
      throw err;
    } finally {
      if (requestId === requestIdRef.current) setLoading(false);
    }
  }

  function reset() {
    requestIdRef.current += 1;

    setResults([]);
    setLatency(null);
    setSubQueries([]);
    setCount(0);
    setLastQuery("");
    setError("");
    setSearchMode("semantic");
    setDurationLimit(-1);
    setLoading(false);
  }

  return {
    results,
    latency,
    subQueries,
    count,
    lastQuery,
    loading,
    error,
    searchMode,
    durationLimit,
    search,
    searchMultimodal,
    searchWithFusion,
    searchSimilar,
    searchByColor,
    reset,
  };
}
