import { useEffect, useRef } from "react";
import { X } from "lucide-react";
import ResultCard from "./ResultCard";

export default function SurroundingFramesModal({
  open,
  centerResult,
  frames = [],
  loading = false,
  columns = 5,
  onColumnsChange,
  onClose,
  onSelect,
  onSubmit,
  onSimilaritySearch,
  onSurroundingImages,
  layer = 1,
}) {
  const centerRef = useRef(null);

  useEffect(() => {
    if (!open) return;

    requestAnimationFrame(() => {
      centerRef.current?.scrollIntoView({
        behavior: "smooth",
        block: "center",
        inline: "center",
      });
    });
  }, [open, centerResult?.video_id, centerResult?.frame_id, frames.length]);

  if (!open || !centerResult) return null;

  return (
    <div className="surround-modal-backdrop" style={{ zIndex: 3000 + layer }} onClick={onClose}>
      <div className="surround-modal" onClick={(e) => e.stopPropagation()}>
        <div className="surround-modal-header">
          <div>
            <h2>Surrounding Frames</h2>
            <p>
              Center: {centerResult.video_id}/
              {String(centerResult.frame_id).padStart(6, "0")}
            </p>
          </div>

          <div className="surround-modal-controls">
            <label>
              Số cột
              <input
                type="range"
                min="2"
                max="8"
                value={columns}
                onChange={(e) => onColumnsChange?.(Number(e.target.value))}
              />
              <span>{columns}</span>
            </label>

            <button className="modal-close-btn" type="button" onClick={onClose}>
              <X size={18} />
            </button>
          </div>
        </div>

        {loading && (
          <div className="surround-loading">
            Loading surrounding frames...
          </div>
        )}

        <div
          className="surround-grid"
          style={{
            gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))`,
          }}
        >
          {frames.map((frame, index) => {
            const isCenter =
              frame.is_surround_center ||
              (
                frame.video_id === centerResult.video_id &&
                Number(frame.frame_id) === Number(centerResult.frame_id)
              );

            return (
              <div
                key={`${frame.video_id}-${frame.frame_id}-${index}`}
                ref={isCenter ? centerRef : null}
                className={isCenter ? "surround-item is-center" : "surround-item"}
              >
                <ResultCard
                  result={frame}
                  selected={isCenter}
                  onSelect={onSelect}
                  onSubmit={onSubmit}
                  onSimilaritySearch={onSimilaritySearch}
                  onSurroundingImages={onSurroundingImages}
                />
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}