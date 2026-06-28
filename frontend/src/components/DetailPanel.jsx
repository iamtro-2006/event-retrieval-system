import { useEffect, useMemo, useRef, useState } from "react";
import {
  X,
  Info,
  Play,
  Copy,
  ExternalLink,
  FileImage,
  Clock,
  Video,
  Gauge,
  Star,
  Send,
} from "lucide-react";
import { getNeighborFrames } from "../utils/frameUtils";
import { buildSurroundingFrames } from "../utils/surroundingFrames";
import VideoModal from "./VideoModal";
import { getFrameInfo } from "../api/retrievalAPI";

export default function DetailPanel({ result, onClose, onSubmit }) {
  const [videoOpen, setVideoOpen] = useState(false);
  const [activeResult, setActiveResult] = useState(result);
  const [timelineLoadingId, setTimelineLoadingId] = useState(null);
  const centerRef = useRef(null);

  useEffect(() => {
    centerRef.current?.scrollIntoView({
      behavior: "smooth",
      inline: "center",
      block: "nearest",
    });
  }, [activeResult?.id, activeResult?.frame_id]);

  const surroundingFrames = useMemo(() => {
    if (!activeResult) return [];
    return buildSurroundingFrames(activeResult, 5);
  }, [activeResult]);

  if (!activeResult) return null;

  const similarity = Number(activeResult.similarity ?? 0);
  const scoreNumber = Math.max(0, Math.min(similarity * 100, 100));
  const scoreText = scoreNumber.toFixed(1);

  const keyframeId = Number(
    activeResult.raw?.keyframe_id_int ??
      activeResult.raw?.keyframe_id ??
      activeResult.frame_id ??
      0
  );

  const frameIdx = Number(activeResult.raw?.frame_idx ?? activeResult.frame_id ?? 0);
  const { prev, current, next } = getNeighborFrames(keyframeId);

  const timestamp = Number(activeResult.timestamp ?? 0);

  async function handleCopyPath() {
    try {
      await navigator.clipboard.writeText(activeResult.path ?? "");
    } catch {
      console.error("Cannot copy path");
    }
  }

  async function handleTimelineClick(frame) {
    const clickedKeyframeId = Number(frame.keyframeId ?? frame.frameId ?? 0);

    if (!Number.isFinite(clickedKeyframeId)) {
      return;
    }

    setTimelineLoadingId(clickedKeyframeId);

    try {
      const nextResult = await getFrameInfo(
        activeResult.video_id,
        clickedKeyframeId
      );

      setActiveResult({
        ...nextResult,
        similarity: activeResult.similarity,
        caption: nextResult.caption || activeResult.caption || "",
      });
    } catch (err) {
      console.error("Cannot load frame info:", err);
    } finally {
      setTimelineLoadingId(null);
    }
  }

  function handleSubmit() {
    onSubmit?.(activeResult);
  }

  return (
    <>
      <aside className="detail-panel">
        <div className="detail-header">
          <h3>
            <Info size={16} />
            Asset Metadata
          </h3>

          <button className="detail-close-button" onClick={onClose}>
            <X size={16} />
          </button>
        </div>

        <div className="detail-preview">
          <img
            src={activeResult.image_url}
            alt={activeResult.path ?? "selected frame"}
            onError={(e) => {
              e.currentTarget.style.display = "none";
            }}
          />

          <button
            className="detail-play-button"
            onClick={() => setVideoOpen(true)}
          >
            <Play size={20} fill="currentColor" />
          </button>
        </div>

        <div className="surrounding-timeline">
          {surroundingFrames.map((frame) => {
            const clickedKeyframeId = Number(frame.keyframeId ?? frame.frameId ?? 0);
            const isLoading = timelineLoadingId === clickedKeyframeId;

            return (
              <button
                key={`${frame.frameId}-${frame.offset}`}
                ref={frame.isCurrent ? centerRef : null}
                type="button"
                className={
                  frame.isCurrent
                    ? "timeline-frame current center"
                    : "timeline-frame"
                }
                title={`${frame.frameName} (${frame.offset})`}
                disabled={isLoading}
                onClick={() => handleTimelineClick(frame)}
              >
                <img src={frame.imageUrl} alt={frame.frameName} />

                {frame.isCurrent && (
                  <span className="timeline-center-star">
                    <Star size={12} fill="currentColor" />
                  </span>
                )}

                <span className="timeline-offset">
                  {isLoading ? "..." : frame.offset === 0 ? "0" : frame.offset}
                </span>
              </button>
            );
          })}
        </div>

        <div className="detail-actions detail-actions-4">
          <button type="button" onClick={() => setVideoOpen(true)}>
            <Play size={14} />
            Play
          </button>

          <button type="button" onClick={handleCopyPath}>
            <Copy size={14} />
            Copy Path
          </button>

          {activeResult.image_url ? (
            <a href={activeResult.image_url} target="_blank" rel="noreferrer">
              <ExternalLink size={14} />
              Open Image
            </a>
          ) : (
            <button type="button" disabled>
              <ExternalLink size={14} />
              No Image
            </button>
          )}

          <button
            type="button"
            className="detail-submit-inline"
            onClick={handleSubmit}
          >
            <Send size={14} />
            Submit
          </button>
        </div>

        <div className="detail-section">
          <h4>Core Information</h4>

          <MetadataRow
            icon={<FileImage size={14} />}
            label="Keyframe"
            value={activeResult.frame_name ?? current}
          />

          <MetadataRow
            icon={<FileImage size={14} />}
            label="Frame idx"
            value={frameIdx}
          />

          <MetadataRow
            icon={<Video size={14} />}
            label="Video"
            value={activeResult.video_id ?? "Unknown"}
          />

          <MetadataRow
            icon={<Clock size={14} />}
            label="Timestamp"
            value={`${timestamp.toFixed(2)}s`}
          />

          <MetadataRow
            icon={<Gauge size={14} />}
            label="Similarity"
            value={`${scoreText}%`}
            strong
          />

          <div className="detail-score-track">
            <div style={{ width: `${scoreNumber}%` }} />
          </div>
        </div>

        <div className="detail-section">
          <h4>Path</h4>

          <p className="detail-path" title={activeResult.path}>
            {activeResult.path ?? "No path available"}
          </p>
        </div>

        <div className="detail-section">
          <h4>Neighbor Keyframes</h4>

          <div className="neighbor-grid">
            <NeighborFrame label="Previous" value={prev} />
            <NeighborFrame label="Current" value={current} active />
            <NeighborFrame label="Next" value={next} />
          </div>
        </div>

        <div className="detail-section">
          <h4>Caption</h4>

          <p className="detail-caption">
            {activeResult.caption || "No caption available for this frame."}
          </p>
        </div>
      </aside>

      <VideoModal
        key={activeResult?.id}
        open={videoOpen}
        result={activeResult}
        onClose={() => setVideoOpen(false)}
        onSubmit={onSubmit}
      />
    </>
  );
}

function MetadataRow({ icon, label, value, strong = false }) {
  return (
    <div className="metadata-row">
      <div className="metadata-label">
        {icon}
        <span>{label}</span>
      </div>

      <span className={strong ? "metadata-value strong" : "metadata-value"}>
        {value}
      </span>
    </div>
  );
}

function NeighborFrame({ label, value, active = false }) {
  return (
    <div className={active ? "neighbor-frame active" : "neighbor-frame"}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}