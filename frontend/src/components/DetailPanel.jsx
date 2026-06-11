import { useState } from "react";
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
} from "lucide-react";
import { getNeighborFrames } from "../utils/frameUtils";
import { buildSurroundingFrames } from "../utils/surroundingFrames";
import VideoModal from "./VideoModal";

export default function DetailPanel({ result, onClose }) {
  const [videoOpen, setVideoOpen] = useState(false);

  if (!result) return null;

  const similarity = Number(result.similarity ?? 0);
  const scoreNumber = Math.max(0, Math.min(similarity * 100, 100));
  const scoreText = scoreNumber.toFixed(1);

  const frameId = Number(result.frame_id ?? 0);
  const { prev, current, next } = getNeighborFrames(frameId);

  const timestamp = Number(result.timestamp ?? 0);
  const surroundingFrames = buildSurroundingFrames(result, 5);

  async function handleCopyPath() {
    try {
      await navigator.clipboard.writeText(result.path ?? "");
      alert("Đã copy path frame.");
    } catch {
      alert("Không thể copy path.");
    }
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
            src={result.image_url}
            alt={result.path ?? "selected frame"}
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
          {surroundingFrames.map((frame) => (
            <div
              key={frame.offset}
              className={
                frame.isCurrent
                  ? "timeline-frame current"
                  : "timeline-frame"
              }
              title={`${frame.frameName} (${frame.offset})`}
            >
              <img src={frame.imageUrl} alt={frame.frameName} />
              <span>{frame.offset === 0 ? "0" : frame.offset}</span>
            </div>
          ))}
        </div>

        <div className="detail-actions">
          <button type="button" onClick={() => setVideoOpen(true)}>
            <Play size={14} />
            Play
          </button>

          <button type="button" onClick={handleCopyPath}>
            <Copy size={14} />
            Copy Path
          </button>

          {result.image_url ? (
            <a href={result.image_url} target="_blank" rel="noreferrer">
              <ExternalLink size={14} />
              Open Image
            </a>
          ) : (
            <button type="button" disabled>
              <ExternalLink size={14} />
              No Image
            </button>
          )}
        </div>

        <div className="detail-section">
          <h4>Core Information</h4>

          <MetadataRow
            icon={<FileImage size={14} />}
            label="Frame"
            value={result.frame_name ?? current}
          />

          <MetadataRow
            icon={<Video size={14} />}
            label="Video"
            value={result.video_id ?? "Unknown"}
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

          <p className="detail-path" title={result.path}>
            {result.path ?? "No path available"}
          </p>
        </div>

        <div className="detail-section">
          <h4>Neighbor Frames</h4>

          <div className="neighbor-grid">
            <NeighborFrame label="Previous" value={prev} />
            <NeighborFrame label="Current" value={current} active />
            <NeighborFrame label="Next" value={next} />
          </div>
        </div>

        <div className="detail-section">
          <h4>Caption</h4>

          <p className="detail-caption">
            {result.caption || "No caption available for this frame."}
          </p>
        </div>
      </aside>

      <VideoModal
        open={videoOpen}
        result={result}
        onClose={() => setVideoOpen(false)}
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