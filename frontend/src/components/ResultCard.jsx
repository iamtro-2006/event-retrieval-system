import { Play, ThumbsDown, ThumbsUp, Send, Search, Images } from "lucide-react";

export default function ResultCard({
  result,
  selected,
  onSelect,
  onSubmit,
  onSimilaritySearch,
  onSurroundingImages,
}) {
  const score = (result.similarity * 100).toFixed(1);
  const label = `${result.video_id}/${String(result.frame_id).padStart(6, "0")}`;
  const sequence = result.matched_sequence || [];

  function openVideoAt(timestamp) {
    if (result.video_url && result.video_url !== "#") {
      window.open(`${result.video_url}#t=${Number(timestamp).toFixed(2)}`, "_blank");
    } else {
      alert(`Play video tại timestamp ${Number(timestamp).toFixed(2)}s`);
    }
  }

  function handlePlay(e) {
    e.stopPropagation();

    const startTime = result.temporal?.start_time ?? result.timestamp;
    openVideoAt(startTime);
  }

  function handleSubmit(e) {
    e.stopPropagation();
    onSubmit?.(result);
  }

  function handleSimilaritySearch(e) {
    e.stopPropagation();
    onSimilaritySearch?.(result);
  }

  function handleSurroundingImages(e) {
    e.stopPropagation();
    console.log("CARD");
    onSurroundingImages?.(result);
  }

  return (
    <article
      className={`result-card ${selected ? "selected" : ""}`}
      onClick={() => onSelect?.(result)}
    >
      <div className="thumbnail-box">
        <img src={result.image_url} alt={label} loading="lazy" decoding="async" />

        <span className="score-badge">{score}%</span>

        <button className="vote-button like" type="button" onClick={(e) => e.stopPropagation()}>
          <ThumbsUp size={12} />
        </button>

        <button className="vote-button dislike" type="button" onClick={(e) => e.stopPropagation()}>
          <ThumbsDown size={12} />
        </button>

        <button className="play-button" type="button" onClick={handlePlay}>
          <Play size={16} fill="currentColor" />
        </button>
      </div>

      {sequence.length > 0 && (
        <div className="temporal-sequence">
          {sequence.map((item, idx) => (
            <button
              key={`${item.video_id || result.video_id}-${item.keyframe_id}-${idx}`}
              className="temporal-step"
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                openVideoAt(item.timestamp_sec);
              }}
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
              onClick={handleSimilaritySearch}
            >
              <Search size={12} strokeWidth={2.4} />
              <span>Similar</span>
            </button>

            <button
              type="button"
              className="result-mini-btn"
              title="Surrounding images"
              onClick={handleSurroundingImages}
            >
              <Images size={12} strokeWidth={2.4} />
              <span>Surround</span>
            </button>

            <button
              type="button"
              className="result-submit-btn"
              onClick={handleSubmit}
            >
              <Send size={12} strokeWidth={2.4} />
              <span>Submit</span>
            </button>
          </div>
        </div>
      </div>
    </article>
  );
}