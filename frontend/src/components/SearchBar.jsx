import { Plus, Search, Mic } from "lucide-react";
import { useState } from "react";

export default function SearchBar({
  model,
  mode,
  loading,
  onModelChange,
  onModeChange,
  onSearch,
}) {
  const [query, setQuery] = useState("");

  function handleSubmit(e) {
    e.preventDefault();
    onSearch(query);
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
          placeholder="Nhập truy vấn retrieval..."
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
          <option value="text">Text</option>
          <option value="image">Image</option>
          <option value="temporal">Temporal</option>
          <option value="hybrid">Hybrid</option>
        </select>

        <button type="button" className="search-icon-button">
          <Mic size={18} />
        </button>

        <button type="submit" className="search-submit" disabled={loading}>
          <Search size={18} />
        </button>
      </div>
    </form>
  );
}