import { Plus, Search, Mic } from "lucide-react";
import { useState } from "react";

export default function SearchBar({
  model,
  mode,
  loading,
  disabled = false,
  durationLimit = -1,
  onModelChange,
  onModeChange,
  onDurationLimitChange,
  onSearch,
}) {
  const [query, setQuery] = useState("");

  const isTemporal = mode === "temporal";

  function handleSubmit(e) {
    e.preventDefault();

    const cleanQuery = query.trim();
    if (!cleanQuery || loading || disabled) return;

    onSearch({
      query: cleanQuery,
      searchMode: isTemporal ? "temporal" : "semantic",
      durationLimit: isTemporal ? Number(durationLimit) : -1,
    });
  }

  return (
    <form
      className={`search-wrapper ${loading ? "searching-active" : ""}`}
      onSubmit={handleSubmit}
    >
      <div className="search-inner">
        <button type="button" className="search-icon-button">
          <Plus size={20} />
        </button>

        <input
          className="search-input"
          value={query}
          placeholder={
            isTemporal
              ? "Ví dụ: person opens box then reads label..."
              : "Nhập truy vấn retrieval..."
          }
          onChange={(e) => setQuery(e.target.value)}
        />

        <select
          className="search-select"
          value={model}
          onChange={(e) => onModelChange(e.target.value)}
        >
          <option value="OpenCLIP">OpenCLIP</option>
          <option value="SigLIP">SigLIP</option>
          <option value="DINOv3">DINOv3</option>
          <option value="Hybrid">Hybrid</option>
        </select>

        <select
          className="search-select"
          value={mode}
          onChange={(e) => onModeChange(e.target.value)}
        >
          <option value="text">Semantic</option>
          <option value="temporal">Temporal</option>
          <option value="hybrid">Hybrid</option>
        </select>

        {isTemporal && (
          <input
            className="search-duration-input"
            type="number"
            value={durationLimit}
            min={-1}
            step={1}
            title="-1 = quét toàn video; >0 = giới hạn số giây"
            onChange={(e) => onDurationLimitChange?.(Number(e.target.value))}
          />
        )}

        <button type="button" className="search-icon-button">
          <Mic size={18} />
        </button>

        <button type="submit" className="search-submit" disabled={loading || disabled}>
          <Search size={18} />
        </button>
      </div>
    </form>
  );
}