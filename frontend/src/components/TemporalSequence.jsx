import { memo, useMemo } from "react";
import { Images, ListPlus, Play, Search, Send } from "lucide-react";
import HighlightedSnippet from "./HighlightedSnippet";
import { VideoVotes } from "./VideoFilter";

function makeFrameResult(result, frame, idx) {
  const frameId = Number(frame.frame_id ?? frame.keyframe_id ?? frame.keyframe_id_int ?? idx);
  const timestamp = Number(frame.timestamp_sec ?? frame.timestamp ?? result.timestamp ?? 0);
  const score = Number(frame.score ?? frame.similarity ?? frame.candidate_score ?? result.similarity ?? 0);
  return {
    ...result,
    ...frame,
    search_mode: frame.search_mode ?? result.search_mode,
    matched_texts: frame.matched_texts ?? result.matched_texts ?? [],
    caption: ["asr", "caption"].includes(result.search_mode)
      ? (result.matched_texts?.join(" ") || result.caption || frame.caption || "")
      : (frame.caption || result.caption || ""),
    id: `${result.id}_q${frame.sub_query_idx ?? idx}_${frameId}`,
    video_id: frame.video_id || result.video_id,
    video_url: frame.video_url || result.video_url,
    frame_id: frameId,
    frame_name: frame.frame_name || `${String(frameId).padStart(6, "0")}.jpg`,
    image_url: frame.image_url || result.image_url,
    timestamp,
    similarity: score,
    matched_sequence: [],
  };
}

const TemporalSequence = memo(function TemporalSequence({
  result,
  index = 0,
  selectedId,
  onSelect,
  onSubmit,
  onPlay,
  onSimilaritySearch,
  onSurroundingImages,
  query,
}) {
  const sequence = useMemo(
    () => Array.isArray(result?.matched_sequence) ? result.matched_sequence : [],
    [result?.matched_sequence]
  );
  // ASR and caption hits share ONE segment text across every frame in the
  // sequence and show it once in the header below. Temporal hits don't set
  // `matched_texts`.
  const hasSharedSegmentText = Array.isArray(result?.matched_texts) && result.matched_texts.length > 0;
  const frameResults = useMemo(
    () => sequence.map((frame, idx) => makeFrameResult(result, frame, idx)),
    [result, sequence]
  );

  if (!sequence.length) return null;

  return (
    <section className={`temporal-sequence-block temporal-events-${Math.min(sequence.length, 6)}`} style={{ "--sequence-event-total": sequence.length, "--stagger-index": index }}>
      <div className="temporal-sequence-toolbar temporal-sequence-header">
        <div>
          <strong>{result.video_id}</strong>
        </div>
        <span className="temporal-sequence-meta">
          {Number(result.temporal?.start_time ?? result.timestamp ?? 0).toFixed(2)}s
          → {Number(result.temporal?.end_time ?? result.timestamp ?? 0).toFixed(2)}s
        </span>
        <div className="temporal-sequence-actions">
          <button type="button" className="temporal-pill-btn" onClick={() => onPlay?.(result)}>
            <Play size={13} fill="currentColor" /><span className="action-label">Play</span>
          </button>
          <button type="button" className="temporal-pill-btn" onClick={() => onSubmit?.(result, "queue")}>
            <ListPlus size={13} /><span className="action-label">Queue</span>
          </button>
          <button type="button" className="temporal-pill-btn temporal-submit-btn" onClick={() => onSubmit?.(result)}>
            <Send size={13} /><span className="action-label">Submit</span>
          </button>
        </div>
      </div>

      <div className="temporal-frame-row">
          {sequence.map((frame, idx) => {
            const frameResult = frameResults[idx];
            const isSelected = frameResult.id === selectedId;
            return (
              <article
                key={frameResult.id}
                className={`temporal-frame-card ${isSelected ? "selected" : ""}`}
                onClick={() => onSelect?.(frameResult)}
              >
                <div className="temporal-frame-thumb">
                  <VideoVotes videoId={frameResult.video_id} />
                  <img src={frameResult.image_url} alt={`Event ${idx + 1}`} loading="lazy" decoding="async" />
                  <span className="temporal-query-badge">E{idx + 1}</span>
                  <span className="temporal-score-badge">{Number(frameResult.similarity).toFixed(3)}</span>
                  <button
                    className="play-button"
                    type="button"
                    aria-label={`Play ${frameResult.video_id} from ${frameResult.timestamp.toFixed(2)} seconds`}
                    onClick={(event) => {
                      event.stopPropagation();
                      onPlay?.(frameResult);
                    }}
                  >
                    <Play size={18} fill="currentColor" />
                  </button>
                </div>
                <div className="temporal-frame-footer">
                  <strong>{frameResult.video_id}/{String(frameResult.frame_id).padStart(6, "0")}</strong>
                  <span>{frameResult.timestamp.toFixed(2)}s</span>
                </div>
                <div className="temporal-frame-actions-row">
                  <button type="button" onClick={(e) => { e.stopPropagation(); onSimilaritySearch?.(frameResult); }}>
                    <Search size={12} /><span className="action-label">Similar</span>
                  </button>
                  <button type="button" onClick={(e) => { e.stopPropagation(); onSurroundingImages?.(frameResult); }}>
                    <Images size={12} /><span className="action-label">Surround</span>
                  </button>
                  <button type="button" onClick={(e) => { e.stopPropagation(); onSubmit?.(frameResult, "queue"); }}>
                    <ListPlus size={12} /><span className="action-label">Queue</span>
                  </button>
                  <button type="button" onClick={(e) => { e.stopPropagation(); onSubmit?.(frameResult); }}>
                    <Send size={12} /><span className="action-label">Submit</span>
                  </button>
                </div>
                {!hasSharedSegmentText && frame.sub_query && (
                  <div className="temporal-frame-query">
                    <HighlightedSnippet
                      text={frame.sub_query}
                      query={query}
                      maxLength={90}
                      highlight={false}
                    />
                  </div>
                )}
              </article>
            );
          })}
      </div>
    </section>
  );
});

export default TemporalSequence;
