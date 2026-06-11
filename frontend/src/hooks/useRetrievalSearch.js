import { useState } from "react";
import { searchRetrieval } from "../api/retrievalApi";

export function useRetrievalSearch() {
  const [results, setResults] = useState([]);
  const [latency, setLatency] = useState(null);
  const [subQueries, setSubQueries] = useState([]);
  const [count, setCount] = useState(0);
  const [lastQuery, setLastQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function search({
    query,
    topK = 20,
    candidateMultiplier,
    useSplit = true,
    useTranslate = true,
  }) {
    const cleanQuery = query?.trim();

    if (!cleanQuery) {
      return;
    }

    setLoading(true);
    setError("");

    try {
      const data = await searchRetrieval({
        query: cleanQuery,
        topK,
        candidateMultiplier,
        useSplit,
        useTranslate,
      });

      setResults(data.results ?? []);
      setLatency(data.latencyMs ?? null);
      setSubQueries(data.subQueries ?? []);
      setCount(data.count ?? 0);
      setLastQuery(data.query ?? cleanQuery);
    } catch (err) {
      setResults([]);
      setLatency(null);
      setSubQueries([]);
      setCount(0);
      setError(err.message || "Search failed");
    } finally {
      setLoading(false);
    }
  }

  function reset() {
    setResults([]);
    setLatency(null);
    setSubQueries([]);
    setCount(0);
    setLastQuery("");
    setError("");
  }

  return {
    results,
    latency,
    subQueries,
    count,
    lastQuery,
    loading,
    error,
    search,
    reset,
  };
}