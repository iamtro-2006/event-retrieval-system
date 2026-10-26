import { useEffect, useMemo, useRef, useState } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
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
  onPlay,
  onSimilaritySearch,
  onSurroundingImages,
  layer = 1,
}) {
  "use no memo";
  const viewportRef = useRef(null);
  const [viewportWidth, setViewportWidth] = useState(1000);
  const rowCount = Math.ceil(frames.length / columns);
  // Fixed rows reserve image and footer space before image download/decoding.
  const cardWidth = Math.max(1, (viewportWidth - 32 - (columns - 1) * 8) / columns);
  const rowHeight = Math.ceil(cardWidth * 9 / 16 + 68);
  // The mutable virtualizer stays local; this component opts out of compiler memoization.
  // eslint-disable-next-line react-hooks/incompatible-library
  const virtualizer = useVirtualizer({
    count: rowCount,
    getScrollElement: () => viewportRef.current,
    estimateSize: () => rowHeight,
    overscan: 2,
    gap: 8,
    paddingStart: 16,
    paddingEnd: 16,
    enabled: open && !loading,
  });
  const centerIndex = useMemo(() => frames.findIndex((frame) => frame.video_id === centerResult?.video_id &&
    Number(frame.frame_id) === Number(centerResult?.frame_id)), [frames, centerResult?.video_id, centerResult?.frame_id]);

  useEffect(() => {
    const viewport = viewportRef.current;
    if (!open || !viewport) return;
    const observer = new ResizeObserver(([entry]) => setViewportWidth(entry.contentRect.width));
    observer.observe(viewport);
    return () => observer.disconnect();
  }, [open]);

  useEffect(() => {
    if (!open || loading || centerIndex < 0) return;

    const tick = requestAnimationFrame(() => {
      virtualizer.measure();
      virtualizer.scrollToIndex(Math.floor(centerIndex / columns), { align: "center", behavior: "auto" });
    });
    return () => cancelAnimationFrame(tick);
  }, [open, loading, centerResult, centerIndex, columns, rowHeight, virtualizer]);

  if (!open || !centerResult) return null;

  return (
    <div className="surround-modal-backdrop" style={{ zIndex: 3000 + layer }} onClick={onClose}>
      <div className="surround-modal video-frames-browser" onClick={(e) => e.stopPropagation()}>
        <div className="surround-modal-header">
          <div>
            <h2>Surrounding Frames</h2>
            <p>
              Center: {centerResult.video_id}/
              {String(centerResult.frame_id).padStart(6, "0")} · {loading ? "Loading…" : `${frames.length} frames · entire video`}
            </p>
          </div>

          <div className="surround-modal-controls">
            <label>
              Columns
              <input
                type="range"
                min="2"
                max="8"
                value={columns}
                onChange={(e) => onColumnsChange?.(Number(e.target.value))}
              />
              <span>{columns}</span>
            </label>

            <button className="modal-close-btn" type="button" aria-label="Close surrounding frames" onClick={onClose}>
              <X size={18} />
            </button>
          </div>
        </div>

        {loading && (
          <div className="surround-loading">
            Loading all video frames...
          </div>
        )}

        <div className="video-frames-viewport" ref={viewportRef}>
          <div className="video-frames-spacer" style={{ height: virtualizer.getTotalSize() }}>
            {virtualizer.getVirtualItems().map((row) => (
              <div key={row.key} className="video-frame-row" data-row-index={row.index}
                style={{ height: rowHeight, transform: `translateY(${row.start}px)`,
                  gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` }}>
                {frames.slice(row.index * columns, (row.index + 1) * columns).map((frame) => {
                  const isCenter = frame.video_id === centerResult.video_id &&
                    Number(frame.frame_id) === Number(centerResult.frame_id);
                  return <div key={`${frame.video_id}-${frame.frame_id}`}
                    className={isCenter ? "surround-item is-center" : "surround-item"}>
                    <ResultCard result={frame} selected={isCenter}
                      onSelect={onSelect} onSubmit={onSubmit} onPlay={onPlay}
                      onSimilaritySearch={onSimilaritySearch} onSurroundingImages={onSurroundingImages} />
                  </div>;
                })}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
