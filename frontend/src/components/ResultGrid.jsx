import {
  Play,
  Search,
  Images,
  Send,
} from "lucide-react";
import ResultCard from "./ResultCard";

function TemporalFlatSequence({
  result,
  sequenceIndex,
  selectedId,
  onSelect,
  onSubmit,
  onSimilaritySearch,
  onSurroundingImages,
}) {
  const sequence = result.matched_sequence || [];

  function makeFrameResult(frame, idx) {
    const frameId = frame.frame_id ?? Number(frame.keyframe_id ?? 0) ?? idx;
    const frameResultId = `${result.id}_q${frame.sub_query_idx ?? idx}`;

    return {
      ...result,
      ...frame,
      id: frameResultId,
      video_id: frame.video_id || result.video_id,
      video_url: frame.video_url || result.video_url,
      frame_id: frameId,
      frame_name: frame.frame_name || `${String(frameId).padStart(6, "0")}.jpg`,
      image_url: frame.image_url,
      timestamp: frame.timestamp_sec,
      similarity: frame.score,
      temporal: {
        ...result.temporal,
        start_time: frame.timestamp_sec,
        end_time: frame.timestamp_sec,
      },
      matched_sequence: [],
    };
  }

  function handleSelectFrame(frame, idx) {
    onSelect?.(makeFrameResult(frame, idx));
  }

  function handlePlay(e) {
    e.stopPropagation();

    const startTime = result.temporal?.start_time ?? result.timestamp ?? 0;

    if (result.video_url && result.video_url !== "#") {
      window.open(`${result.video_url}#t=${Number(startTime).toFixed(2)}`, "_blank");
    }
  }

  function handleSubmitSequence(e) {
    e.stopPropagation();
    onSubmit?.(result);
  }

  function handleSimilaritySearch(e, frame, idx) {
    e.stopPropagation();
    onSimilaritySearch?.(makeFrameResult(frame, idx));
  }

  function handleSurroundingImages(e, frame, idx) {
    e.stopPropagation();
    onSurroundingImages?.(makeFrameResult(frame, idx));
  }

  function handleSubmitFrame(e, frame, idx) {
    e.stopPropagation();
    onSubmit?.(makeFrameResult(frame, idx));
  }

  return (
    <section
      className="temporal-flat-sequence"
      style={{
        "--seq-count": sequence.length,
      }}
    >
      <div className="temporal-flat-header">
        <span>Sequence #{sequenceIndex + 1}</span>

        <span>
          {result.video_id} ·{" "}
          {Number(result.temporal?.duration_sec ?? 0).toFixed(2)}s
        </span>

        <button type="button" onClick={handlePlay} title="Play sequence">
          <Play size={13} fill="currentColor" />
          Play
        </button>

        <button
          type="button"
          onClick={handleSubmitSequence}
          title="Submit sequence start frame"
        >
          <Send size={13} />
          Submit
        </button>
      </div>

      <div
        className="temporal-flat-frames"
        style={{
          "--seq-count": sequence.length,
        }}
      >
        {sequence.map((frame, idx) => {
          const frameResult = makeFrameResult(frame, idx);
          const isSelected = selectedId === frameResult.id;

          return (
            <article
              key={frameResult.id}
              className={`temporal-flat-frame-card ${isSelected ? "selected" : ""}`}
              onClick={() => handleSelectFrame(frame, idx)}
            >
              <div className="temporal-flat-thumb">
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

              <div className="temporal-flat-footer">
                <strong>
                  {frameResult.video_id}/
                  {String(frameResult.frame_id ?? 0).padStart(6, "0")}
                </strong>
                <span>{Number(frame.timestamp_sec ?? 0).toFixed(2)}s</span>
              </div>

              <div className="temporal-frame-actions-row">
                <button
                  type="button"
                  title="Similarity search"
                  onClick={(e) => handleSimilaritySearch(e, frame, idx)}
                >
                  <Search size={12} />
                  <span>Similar</span>
                </button>

                <button
                  type="button"
                  title="Surrounding images"
                  onClick={(e) => handleSurroundingImages(e, frame, idx)}
                >
                  <Images size={12} />
                  <span>Surround</span>
                </button>

                <button
                  type="button"
                  title="Submit frame"
                  onClick={(e) => handleSubmitFrame(e, frame, idx)}
                >
                  <Send size={12} />
                  <span>Submit</span>
                </button>
              </div>

              {frame.sub_query && (
                <div className="temporal-frame-query" title={frame.sub_query}>
                  {frame.sub_query}
                </div>
              )}
            </article>
          );
        })}
      </div>
    </section>
  );
}

export default function ResultGrid({
  results,
  columns,
  selectedId,
  onSelect,
  onSubmit,
  onSimilaritySearch,
  onSurroundingImages,
}) {
  return (
    <div
      className="result-grid"
      style={{
        gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))`,
      }}
    >
      {results.map((result, index) => {
        const hasTemporal =
          Array.isArray(result.matched_sequence) &&
          result.matched_sequence.length > 0;

        if (hasTemporal) {
          return (
            <TemporalFlatSequence
              key={result.id}
              result={result}
              sequenceIndex={index}
              selectedId={selectedId}
              onSelect={onSelect}
              onSubmit={onSubmit}
              onSimilaritySearch={onSimilaritySearch}
              onSurroundingImages={onSurroundingImages}
            />
          );
        }

        return (
          <ResultCard
            key={result.id}
            result={result}
            selected={result.id === selectedId}
            onSelect={onSelect}
            onSubmit={onSubmit}
            onSimilaritySearch={onSimilaritySearch}
            onSurroundingImages={onSurroundingImages}
          />
        );
      })}
    </div>
  );
}