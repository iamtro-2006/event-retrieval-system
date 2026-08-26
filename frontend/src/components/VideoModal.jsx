import { useEffect, useMemo, useRef, useState } from "react";
import { ChevronLeft, ChevronRight, Copy, Send, X } from "lucide-react";
import { getFrameIdxAtTimestamp } from "../api/retrievalAPI";

export default function VideoModal({ open, result, onClose, onSubmit, layer = 40, autoPlay = true }) {
  const videoRef = useRef(null);
  const initialMs = useMemo(() => Math.max(0, Math.round(Number(result?.timestamp ?? 0) * 1000)), [result]);
  const [currentMs, setCurrentMs] = useState(initialMs);
  const [durationMs, setDurationMs] = useState(Math.max(initialMs + 1000, 1000));
  const [stepMs, setStepMs] = useState(500);
  const [copyState, setCopyState] = useState("");

  useEffect(() => {
    if (!open) return;
    setCurrentMs(initialMs);
    setCopyState("");
  }, [open, initialMs]);

  if (!open || !result) return null;

  function seekTo(nextMs) {
    const bounded = Math.max(0, Math.min(Number(nextMs) || 0, durationMs));
    setCurrentMs(Math.round(bounded));
    if (videoRef.current) videoRef.current.currentTime = bounded / 1000;
  }

  function submitAtCurrentTime() {
    onSubmit?.({
      ...result,
      timestamp: currentMs / 1000,
      submit_timestamp_ms: currentMs,
      raw: { ...result.raw, submit_timestamp_ms: currentMs },
    });
  }

  async function copyCurrentFrameId() {
    const playbackMs = videoRef.current && Number.isFinite(videoRef.current.currentTime)
      ? videoRef.current.currentTime * 1000
      : currentMs;
    const timestampMs = Math.max(0, Math.round(playbackMs));
    setCurrentMs(timestampMs);

    try {
      const frameIdx = await getFrameIdxAtTimestamp(result.video_id, timestampMs);
      await navigator.clipboard.writeText(`${result.video_id}, ${frameIdx}`);
      setCopyState(`Copied ${frameIdx}`);
    } catch (error) {
      console.error("Cannot copy current frame id", error);
      setCopyState("Copy failed");
    }
    window.setTimeout(() => setCopyState(""), 1800);
  }

  return (
    <div className="video-modal-backdrop" style={{ zIndex: 3000 + layer }} onClick={onClose}>
      <div className="video-modal" onClick={(e) => e.stopPropagation()}>
        <div className="video-modal-header">
          <div>
            <h3>{result.video_id}</h3>
            <p>Frame {result.frame_name} · {(currentMs / 1000).toFixed(3)}s · {currentMs} ms</p>
          </div>
          <button type="button" onClick={onClose}><X size={18} /></button>
        </div>

        {result.video_url && result.video_url !== "#" ? (
          <video
            ref={videoRef}
            className="video-player"
            src={`${result.video_url}#t=${initialMs / 1000}`}
            controls
            autoPlay={autoPlay}
            onLoadedMetadata={(e) => {
              const ms = Math.max(1000, Math.round((e.currentTarget.duration || 0) * 1000));
              setDurationMs(ms);
              e.currentTarget.currentTime = initialMs / 1000;
            }}
            onTimeUpdate={(e) => setCurrentMs(Math.round(e.currentTarget.currentTime * 1000))}
          />
        ) : <div className="video-missing">Video file not available.</div>}

        <div className="video-submit-controls">
          <div className="video-seek-row">
            <button type="button" onClick={() => seekTo(currentMs - stepMs)}><ChevronLeft size={18} /> -{stepMs} ms</button>
            <input type="range" min="0" max={durationMs} step="1" value={Math.min(currentMs, durationMs)} onChange={(e) => seekTo(Number(e.target.value))} />
            <button type="button" onClick={() => seekTo(currentMs + stepMs)}>+{stepMs} ms <ChevronRight size={18} /></button>
          </div>
          <div className="video-submit-row">
            <label>Seek step
              <select value={stepMs} onChange={(e) => setStepMs(Number(e.target.value))}>
                <option value="100">100 ms</option><option value="250">250 ms</option><option value="500">500 ms</option><option value="1000">1000 ms</option><option value="5000">5000 ms</option>
              </select>
            </label>
            <label>Exact ms
              <input type="number" min="0" max={durationMs} value={currentMs} onChange={(e) => seekTo(Number(e.target.value))} />
            </label>
            <button type="button" className="video-copy-id-button" onClick={copyCurrentFrameId}><Copy size={16} /> {copyState || "Copy ID"}</button>
            <button type="button" className="video-submit-button" onClick={submitAtCurrentTime}><Send size={16} /> Submit at {currentMs} ms</button>
          </div>
        </div>
      </div>
    </div>
  );
}
