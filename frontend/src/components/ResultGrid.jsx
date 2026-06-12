import { Play } from "lucide-react";
import ResultCard from "./ResultCard";

function TemporalFlatSequence({ result, sequenceIndex, selectedId, onSelect }) {
  const sequence = result.matched_sequence || [];

  function handleSelectFrame(frame, idx) {
    const frameResultId = `${result.id}_q${frame.sub_query_idx ?? idx}`;

    onSelect({
      ...result,
      ...frame,
      id: frameResultId,
      frame_id: frame.frame_id,
      frame_name: frame.frame_name,
      image_url: frame.image_url,
      timestamp: frame.timestamp_sec,
      similarity: frame.score,
    });
  }

  function handlePlay(e) {
    e.stopPropagation();

    const startTime = result.temporal?.start_time ?? result.timestamp ?? 0;

    if (result.video_url && result.video_url !== "#") {
      window.open(`${result.video_url}#t=${Number(startTime).toFixed(2)}`, "_blank");
    }
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
      </div>

      <div
        className="temporal-flat-frames"
        style={{
          "--seq-count": sequence.length,
        }}
      >
        {sequence.map((frame, idx) => {
          const frameResultId = `${result.id}_q${frame.sub_query_idx ?? idx}`;
          const isSelected = selectedId === frameResultId;

          return (
            <article
              key={frameResultId}
              className={`temporal-flat-frame-card ${isSelected ? "selected" : ""}`}
              onClick={() => handleSelectFrame(frame, idx)}
            >
              <div className="temporal-flat-thumb">
                <img src={frame.image_url} alt={`Q${idx + 1}`} />

                <span className="temporal-query-badge">Q{idx + 1}</span>

                <span className="temporal-score-badge">
                  {(Number(frame.score ?? 0) * 100).toFixed(1)}%
                </span>
              </div>

              <div className="temporal-flat-footer">
                <span>
                  {result.video_id}/{String(frame.frame_id ?? 0).padStart(6, "0")}
                </span>
                <span>{Number(frame.timestamp_sec ?? 0).toFixed(2)}s</span>
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
            />
          );
        }

        return (
          <ResultCard
            key={result.id}
            result={result}
            selected={result.id === selectedId}
            onSelect={onSelect}
          />
        );
      })}
    </div>
  );
}