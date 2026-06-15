export default function TemporalSequence({
  sequence,
  onFrameSelect,
}) {
  return (
    <div className="temporal-sequence-block">
      <div className="temporal-sequence-grid">
        {sequence.map((frame, idx) => (
          <article
            key={frameResult.id}
            className={`temporal-frame-card ${isSelected ? "selected" : ""}`}
            onClick={() => handleSelectFrame(frame, idx)}
          >
            <div className="temporal-frame-thumb">
              <img
                src={frameResult.image_url}
                alt={`Q${idx + 1}`}
                loading="lazy"
                decoding="async"
              />

              <span className="temporal-query-badge">Q{idx + 1}</span>

              <span className="temporal-score-badge">
                {(Number(frame.score ?? 0) * 100).toFixed(1)}%
              </span>
            </div>

            <div className="temporal-frame-footer">
              <strong>
                {frameResult.video_id}/{String(frameResult.frame_id ?? 0).padStart(6, "0")}
              </strong>
              <span>{Number(frame.timestamp_sec ?? 0).toFixed(2)}s</span>
            </div>

            <div className="temporal-frame-actions-row">
              <button
                type="button"
                title="Similarity search"
                onClick={(e) => handleSimilaritySearch(e, frame, idx)}
              >
                Similar
              </button>

              <button
                type="button"
                title="Surrounding images"
                onClick={(e) => handleSurroundingImages(e, frame, idx)}
              >
                Surround
              </button>

              <button
                type="button"
                title="Submit frame"
                onClick={(e) => {
                  e.stopPropagation();
                  onSubmit?.(frameResult);
                }}
              >
                Submit
              </button>
            </div>

            {frame.sub_query && (
              <div className="temporal-frame-query" title={frame.sub_query}>
                {frame.sub_query}
              </div>
            )}
          </article>
        ))}
      </div>
    </div>
  );
}