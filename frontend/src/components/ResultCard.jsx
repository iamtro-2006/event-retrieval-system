import { memo, useCallback } from "react";
import { Play, Send, Search, Images, ListPlus } from "lucide-react";
import { VideoVotes } from "./VideoFilter";
import HighlightedSnippet from "./HighlightedSnippet";

const ResultCard = memo(function ResultCard({
  result,
  index = 0,
  selected,
  onSelect,
  onSubmit,
  onPlay,
  onSimilaritySearch,
  onSurroundingImages,
  query,
}) {
  const score    = (result.similarity * 100).toFixed(1);
  const label    = `${result.video_id}/${String(result.frame_id).padStart(6, "0")}`;
  const sequence = result.matched_sequence || [];
  // OCR-only: on-screen text that matched the query (ASR results don't reach
  // this component — they carry a matched_sequence and render via
  // TemporalSequence instead).
  const matchedTexts = Array.isArray(result.matched_texts) ? result.matched_texts : [];

  // Memoize handlers to prevent child re-renders
  const openVideoAt = useCallback((timestamp) => {
    onPlay?.({ ...result, timestamp: Number(timestamp) || 0 });
  }, [onPlay, result]);

  const handlePlay = useCallback((e) => {
    e.stopPropagation();
    openVideoAt(result.temporal?.start_time ?? result.timestamp);
  }, [openVideoAt, result.temporal?.start_time, result.timestamp]);

  const handleSubmit = useCallback((e) => {
    e.stopPropagation();
    onSubmit?.(result);
  }, [onSubmit, result]);

  const handleQueue = useCallback((e) => {
    e.stopPropagation();
    onSubmit?.(result, "queue");
  }, [onSubmit, result]);

  const handleSimilaritySearch = useCallback((e) => {
    e.stopPropagation();
    onSimilaritySearch?.(result);
  }, [onSimilaritySearch, result]);

  const handleSurroundingImages = useCallback((e) => {
    e.stopPropagation();
    onSurroundingImages?.(result);
  }, [onSurroundingImages, result]);

  // Image lazy-load fade
  function onImgLoad(e) {
    e.currentTarget.classList.add("loaded");
  }

  return (
    <article
      className={`result-card${selected ? " selected" : ""}${result.godmode_verified ? " godmode-verified" : ""}`}
      style={{ "--stagger-index": index }}
      onClick={() => onSelect?.(result)}
    >
      <div className="thumbnail-box">
        <img
          src={result.image_url}
          alt={label}
          loading="lazy"
          decoding="async"
          onLoad={onImgLoad}
        />

        <span className="score-badge">{score}%</span>
        {result.godmode_verified && <span className="godmode-badge">DRES ✓</span>}

        <VideoVotes videoId={result.video_id} />

        <button className="play-button" type="button" onClick={handlePlay}
          aria-label={`Play ${result.video_id} from ${Number(result.temporal?.start_time ?? result.timestamp ?? 0).toFixed(2)} seconds`}>
          <Play size={16} fill="currentColor" />
        </button>
      </div>

      {matchedTexts.length > 0 && (
        <div className="ocr-matched-text">
          <HighlightedSnippet
            text={matchedTexts.join(" · ")}
            query={query}
            maxLength={140}
          />
        </div>
      )}

      {sequence.length > 0 && (
        <div className="temporal-sequence">
          {sequence.map((item, idx) => (
            <button
              key={`${item.video_id || result.video_id}-${item.keyframe_id}-${idx}`}
              className="temporal-step"
              type="button"
              onClick={(e) => { e.stopPropagation(); openVideoAt(item.timestamp_sec); }}
              title={item.sub_query}
            >
              <img
                src={item.image_url}
                alt={`Q${idx + 1}`}
                className="temporal-step-img"
                loading="lazy"
                decoding="async"
              />
              <div className="temporal-step-info">
                <span>Q{idx + 1}</span>
                <span>{Number(item.timestamp_sec).toFixed(2)}s</span>
              </div>
            </button>
          ))}
        </div>
      )}

      <div className="card-footer">
        <div className="result-footer">
          <div className="result-frame-label">{label}</div>

          <div className="result-actions-row">
            <button
              type="button"
              className="result-mini-btn"
              title="Similarity search"
              aria-label="Similarity search"
              onClick={handleSimilaritySearch}
            >
              <Search size={15} strokeWidth={2.4} />
              <span className="action-label">Similar</span>
            </button>

            <button
              type="button"
              className="result-mini-btn"
              title="Surrounding images"
              aria-label="Surrounding images"
              onClick={handleSurroundingImages}
            >
              <Images size={15} strokeWidth={2.4} />
              <span className="action-label">Surround</span>
            </button>

            <button
              type="button"
              className="result-mini-btn"
              title="Add to submission queue"
              aria-label="Add to submission queue"
              onClick={handleQueue}
            >
              <ListPlus size={15} strokeWidth={2.4} />
              <span className="action-label">Queue</span>
            </button>
            <button type="button" className="result-submit-btn" title="Submit KIS" aria-label="Submit KIS" onClick={handleSubmit}>
              <Send size={15} strokeWidth={2.4} />
              <span className="action-label">Submit</span>
            </button>
          </div>
        </div>
      </div>
    </article>
  );
});

export default ResultCard;
