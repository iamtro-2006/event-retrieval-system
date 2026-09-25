import { useMemo, useState } from "react";
import { ChevronDown, ChevronUp, LoaderCircle, Search, X } from "lucide-react";
import ResultCard from "./ResultCard";
import TemporalSequence from "./TemporalSequence";
import { groupByVideoSorted } from "../utils/groupByVideo";

function VideoSearchControl({ videoId, topK, onSearch, loading }) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const resultLimit = Math.max(1, Math.floor(Number(topK || 20) / 2));

  function submit(event) {
    event.preventDefault();
    const cleanQuery = query.trim();
    if (!cleanQuery || loading) return;
    onSearch(videoId, cleanQuery);
    setOpen(false);
  }

  return (
    <div className="video-group-search">
      {open && (
        <form className="video-group-search-form" onSubmit={submit}>
          <input
            autoFocus
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Semantic Search"
            aria-label={`Semantic search within video ${videoId}`}
          />
          <button type="submit" disabled={!query.trim() || loading} aria-label="Search this video" title="Search this video">
            {loading ? <LoaderCircle size={14} className="spin" /> : <Search size={14} />}
          </button>
          <button type="button" onClick={() => setOpen(false)} aria-label="Close video search" title="Close">
            <X size={14} />
          </button>
        </form>
      )}
      {!open && (
        <button
          type="button"
          className="video-group-search-toggle"
          onClick={() => setOpen(true)}
          disabled={loading}
          title={`Search every keyframe in ${videoId} and replace its group with the top ${resultLimit} results`}
        >
          <Search size={13} /> <span>Search video · top {resultLimit}</span>
        </button>
      )}
    </div>
  );
}

export default function GroupedResults({
  results,
  columns,
  selectedId,
  onSelect,
  onSubmit,
  onPlay,
  onSimilaritySearch,
  onSurroundingImages,
  onSearchVideo,
  topK = 20,
  searchingVideoIds,
  localQueries = {},
  groupOrder = [],
  query,
}) {
  const verified = useMemo(() => results.filter((item) => item.godmode_verified), [results]);
  const groups = useMemo(() => {
    const order = new Map(groupOrder.map((videoId, index) => [videoId, index]));
    return groupByVideoSorted(results.filter((item) => !item.godmode_verified)).sort((a, b) =>
      (order.get(a.videoId) ?? Number.MAX_SAFE_INTEGER) - (order.get(b.videoId) ?? Number.MAX_SAFE_INTEGER)
    );
  }, [groupOrder, results]);
  const [expandedByVideo, setExpandedByVideo] = useState({});

  function toggleExpanded(videoId) {
    setExpandedByVideo((previous) => ({ ...previous, [videoId]: !previous[videoId] }));
  }

  return (
    <div className="grouped-results">
      {verified.length > 0 && (
        <section className="video-strip-group">
          <div className="video-strip-label"><span>RERANK</span></div>
          <div className="video-strip-main">
            <div className="video-strip-topbar"><span>{verified.length} assets</span></div>
            <div className="video-group-viewport is-expanded">
              <div className="video-group-items" style={{ gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` }}>
                {verified.map((result) => <ResultCard key={result.id} result={result} selected={result.id === selectedId}
                  onSelect={onSelect} onSubmit={onSubmit} onPlay={onPlay} onSimilaritySearch={onSimilaritySearch}
                  onSurroundingImages={onSurroundingImages} query={query} />)}
              </div>
            </div>
          </div>
        </section>
      )}
      {groups.map(({ videoId, items }) => {
        const hasTemporal = items.some(
          (item) => Array.isArray(item.matched_sequence) && item.matched_sequence.length > 0
        );
        const isExpanded = Boolean(expandedByVideo[videoId]);
        const visibleItems = isExpanded ? items : items.slice(0, Math.max(1, columns));
        const cardGridStyle = hasTemporal
          ? { "--sequence-cols": columns }
          : { gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` };

        return (
          <section key={videoId} className="video-strip-group">
            <div className="video-strip-label">
              <span>{videoId}</span>
            </div>

            <div className="video-strip-main">
              <div className="video-strip-topbar">
                <span>
                  {items.length} {hasTemporal ? "temporal sequences" : "assets"}
                  {localQueries[videoId] ? ` · ${localQueries[videoId]}` : ""}
                </span>
                <div className="video-strip-actions">
                  <VideoSearchControl videoId={videoId} topK={topK} onSearch={onSearchVideo} loading={Boolean(searchingVideoIds?.[videoId])} />
                </div>
              </div>

              <div className={`video-group-viewport ${isExpanded ? "is-expanded" : ""}`}>
                <div className={`video-group-items ${hasTemporal ? "temporal-sequences-grid grouped-temporal-sequences-grid" : ""}`}
                  style={cardGridStyle}>
                  {visibleItems.map((result, index) => hasTemporal ? (
                    <TemporalSequence key={result.id} result={result} sequenceIndex={index}
                      selectedId={selectedId} onSelect={onSelect} onSubmit={onSubmit} onPlay={onPlay}
                      onSimilaritySearch={onSimilaritySearch} onSurroundingImages={onSurroundingImages} query={query} />
                  ) : (
                    <ResultCard key={result.id} result={result} selected={result.id === selectedId}
                      onSelect={onSelect} onSubmit={onSubmit} onPlay={onPlay} onSimilaritySearch={onSimilaritySearch}
                      onSurroundingImages={onSurroundingImages} query={query} />
                  ))}
                </div>
              </div>
              {items.length > Math.max(1, columns) && (
                <div className="video-group-expand-footer">
                  <button type="button" className="video-group-expand-toggle"
                    onClick={() => toggleExpanded(videoId)} aria-expanded={isExpanded}
                    aria-label={`${isExpanded ? "Collapse" : "Expand"} results for ${videoId}`}
                    title={isExpanded ? "Collapse results" : "Expand results"}>
                    {isExpanded ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
                  </button>
                </div>
              )}
            </div>
          </section>
        );
      })}
    </div>
  );
}
