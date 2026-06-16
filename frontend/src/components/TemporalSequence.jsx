import { useMemo, useState } from "react";

function normalizeFrameResult(frame) {
  const frameId = Number(
    frame.frame_id ??
      frame.keyframe_id ??
      frame.keyframe_id_int ??
      0
  );

  const videoId = String(frame.video_id ?? "");

  return {
    id: frame.id ?? `${videoId}_${String(frameId).padStart(6, "0")}`,
    video_id: videoId,
    frame_id: frameId,
    frame_name: frame.frame_name ?? `${String(frameId).padStart(6, "0")}.jpg`,
    image_url: frame.image_url ?? "",
    video_url: frame.video_url ?? "#",
    timestamp: Number(frame.timestamp ?? frame.timestamp_sec ?? 0),
    similarity: Number(frame.score ?? frame.similarity ?? frame.candidate_score ?? 0),
    raw: frame.raw ?? frame,
  };
}

export default function TemporalSequence({
  sequence = [],
  sequenceIndex = 0,
  videoId = "",
  startTime = 0,
  onPlay,
  onSubmitSequence,
  onFrameSelect,
  onSimilaritySearch,
  onSurroundingImages,
  onSubmit,
}) {
  const [selectedIdx, setSelectedIdx] = useState(null);

  const normalizedSequence = useMemo(
    () => sequence.map((frame) => normalizeFrameResult(frame)),
    [sequence]
  );

  if (!Array.isArray(sequence) || sequence.length === 0) return null;

  return (
    <div className="temporal-sequence-block">
      <div className="temporal-sequence-header">
        <div className="temporal-sequence-title">
          Sequence #{sequenceIndex + 1}
        </div>

        <div className="temporal-sequence-center">
          <span>
            {videoId} · {Number(startTime).toFixed(2)}s
          </span>

          <button
            type="button"
            className="temporal-play-btn"
            onClick={(e) => {
              e.stopPropagation();
              onPlay?.();
            }}
          >
            ▶ Play
          </button>
        </div>

        <div className="temporal-sequence-right">
          <button
            type="button"
            className="temporal-submit-btn"
            onClick={(e) => {
              e.stopPropagation();
              onSubmitSequence?.();
            }}
          >
            ✈ Submit
          </button>
        </div>
      </div>

      <div className="temporal-sequence-grid">
        {sequence.map((frame, idx) => {
          const frameResult = normalizedSequence[idx];
          const isSelected = selectedIdx === idx;

          return (
            <article
              key={frameResult.id}
              className={`temporal-frame-card ${isSelected ? "selected" : ""}`}
              onClick={() => {
                setSelectedIdx(idx);
                onFrameSelect?.(frameResult, frame, idx);
              }}
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
                  {(Number(frame.score ?? frameResult.similarity ?? 0) * 100).toFixed(1)}%
                </span>
              </div>

              <div className="temporal-frame-footer">
                <strong>
                  {frameResult.video_id}/{String(frameResult.frame_id ?? 0).padStart(6, "0")}
                </strong>
                <span>
                  {Number(frame.timestamp_sec ?? frameResult.timestamp ?? 0).toFixed(2)}s
                </span>
              </div>

              <div className="temporal-frame-actions-row">
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    onSimilaritySearch?.(frameResult, frame, idx);
                  }}
                >
                  Similar
                </button>

                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    onSurroundingImages?.(frameResult, frame, idx);
                  }}
                >
                  Surround
                </button>

                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    onSubmit?.(frameResult, frame, idx);
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
          );
        })}
      </div>
    </div>
  );
}