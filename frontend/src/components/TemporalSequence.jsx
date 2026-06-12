export default function TemporalSequence({
  sequence,
  onFrameSelect,
}) {
  return (
    <div className="temporal-sequence-block">
      <div className="temporal-sequence-grid">
        {sequence.map((frame, idx) => (
          <div
            key={`${frame.keyframe_id}-${idx}`}
            className="temporal-frame-card"
            onClick={() => onFrameSelect(frame)}
          >
            <img
              src={frame.image_url}
              alt={`Q${idx + 1}`}
            />

            <div className="temporal-frame-info">
              <span>Q{idx + 1}</span>
              <span>
                {frame.timestamp_sec.toFixed(2)}s
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}