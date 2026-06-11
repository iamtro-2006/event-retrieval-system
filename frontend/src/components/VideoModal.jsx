import { X } from "lucide-react";

export default function VideoModal({ open, result, onClose }) {
  if (!open || !result) return null;

  return (
    <div className="video-modal-backdrop" onClick={onClose}>
      <div className="video-modal" onClick={(e) => e.stopPropagation()}>
        <div className="video-modal-header">
          <div>
            <h3>{result.video_id}</h3>
            <p>
              Frame {result.frame_name} · {result.timestamp.toFixed(2)}s
            </p>
          </div>

          <button onClick={onClose}>
            <X size={18} />
          </button>
        </div>

        {result.video_url && result.video_url !== "#" ? (
          <video
            className="video-player"
            src={`${result.video_url}#t=${result.timestamp}`}
            controls
            autoPlay
          />
        ) : (
          <div className="video-missing">
            Video file not available.
          </div>
        )}
      </div>
    </div>
  );
}