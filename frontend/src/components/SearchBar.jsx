import { Plus, Search, Mic } from "lucide-react";
import { useEffect, useRef, useState, useCallback } from "react";

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
  const textareaRef = useRef(null);

  const isTemporal = mode === "temporal";
  const isExpanded = query.includes("\n") || query.length > 80;

  // Auto-resize textarea — uses requestAnimationFrame for perf
  const resizeTextarea = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    requestAnimationFrame(() => {
      el.style.height = "auto";
      el.style.height = `${Math.min(el.scrollHeight, 220)}px`;
    });
  }, []);

  useEffect(() => {
    resizeTextarea();
  }, [query, resizeTextarea]);

  function resolveSearchMode() {
    if (mode === "temporal") return "temporal";
    if (mode === "auto")     return "auto";
    return "semantic";
  }

  function handleSubmit(e) {
    e?.preventDefault();
    const cleanQuery = query.trim();
    if (!cleanQuery || loading || disabled) return;

    const searchMode = resolveSearchMode();
    onSearch({
      query: cleanQuery,
      searchMode,
      durationLimit: searchMode === "temporal" ? Number(durationLimit) : -1,
    });
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && e.shiftKey) return;
    if (e.key === "Enter") {
      e.preventDefault();
      handleSubmit();
    }
  }

  return (
    <form
      className={[
        "search-wrapper",
        loading    ? "searching-active" : "",
        isExpanded ? "expanded"         : "",
      ].filter(Boolean).join(" ")}
      onSubmit={handleSubmit}
    >
      <div className="search-inner">
        <textarea
          ref={textareaRef}
          className="search-chat-textarea"
          value={query}
          rows={1}
          placeholder={
            isTemporal
              ? "Ví dụ: person opens box; reads label..."
              : "Nhập truy vấn retrieval..."
          }
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
        />

        <div className="search-chat-footer">
          <button type="button" className="search-icon-button" aria-label="Attach">
            <Plus size={20} />
          </button>

          <div className="search-chat-controls">
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
              <option value="auto">Auto</option>
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

            <button type="button" className="search-icon-button" aria-label="Voice">
              <Mic size={18} />
            </button>

            <button
              type="submit"
              className="search-submit"
              disabled={loading || disabled}
              aria-label="Search"
            >
              <Search size={18} />
            </button>
          </div>
        </div>
      </div>
    </form>
  );
}
