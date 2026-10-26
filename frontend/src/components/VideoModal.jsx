import { useEffect, useMemo, useRef, useState } from "react";
import { ChevronLeft, ChevronRight, Copy, ListPlus, Send, X } from "lucide-react";
import { getFrameIdxAtTimestamp, getVideoKeyframes, getVideoPreview } from "../api/retrievalAPI";
import { findActiveKeyframe, initialVideoMilliseconds, normalizeVideoKeyframes } from "../utils/videoKeyframes";
import VideoKeyframeStrip from "./VideoKeyframeStrip";
import "./VideoModal.css";

export default function VideoModal({ open, result, dataset, ...props }) {
  if (!open || !result) return null;
  const initialMs = initialVideoMilliseconds(result);
  return <VideoPlayback key={`${dataset}:${result.video_id}:${initialMs}`} result={result} dataset={dataset} initialMs={initialMs} {...props} />;
}

function VideoPlayback({ result, dataset, initialMs, onClose, onSubmit, layer = 40, autoPlay = true }) {
  const videoRef = useRef(null);
  const copyTimer = useRef(null);
  const [currentMs, setCurrentMs] = useState(initialMs);
  const [mediaDurationMs, setMediaDurationMs] = useState(null);
  const [stepMs, setStepMs] = useState(500);
  const [strip, setStrip] = useState({ keyframes: [], loading: true, error: "" });
  const [copyState, setCopyState] = useState("");
  const [actionBusy, setActionBusy] = useState(false);
  const [error, setError] = useState("");
  const videoId = result.video_id;

  useEffect(() => {
    const controller = new AbortController();
    getVideoKeyframes(dataset, videoId, { signal: controller.signal }).then((data) => {
      if (!controller.signal.aborted) setStrip({ keyframes: normalizeVideoKeyframes(data.keyframes), loading: false, error: "" });
    }).catch((err) => {
      if (!controller.signal.aborted) setStrip({ keyframes: [], loading: false, error: err.message });
    });
    return () => controller.abort();
  }, [dataset, videoId]);

  useEffect(() => {
    const closeOnEscape = (event) => { if (event.key === "Escape") onClose(); };
    window.addEventListener("keydown", closeOnEscape);
    return () => {
      window.removeEventListener("keydown", closeOnEscape);
      window.clearTimeout(copyTimer.current);
    };
  }, [onClose]);

  const activeIndex = useMemo(() => findActiveKeyframe(strip.keyframes, currentMs / 1000), [strip.keyframes, currentMs]);
  const activeFrame = strip.keyframes[activeIndex];
  const durationMs = mediaDurationMs ?? Math.max(initialMs + 1000, (strip.keyframes.at(-1)?.timestamp || 0) * 1000 + 1000);
  const fps = Number(activeFrame?.fps ?? result.fps ?? result.raw?.fps);
  const frameIdx = fps > 0 ? Math.round(currentMs / 1000 * fps) : activeFrame?.frame_idx ?? result.frame_idx ?? result.raw?.frame_idx;
  const videoUrl = result.video_url && result.video_url !== "#" ? result.video_url : "";

  function seekTo(value) {
    const nextMs = Math.round(Math.max(0, Math.min(Number(value) || 0, durationMs)));
    setCurrentMs(nextMs);
    if (videoRef.current) videoRef.current.currentTime = nextMs / 1000;
  }

  async function sendCurrent(action) {
    setActionBusy(true);
    setError("");
    try {
      const preview = await getVideoPreview(dataset, videoId, { timestampMs: currentMs });
      const exactFrameId = Number(preview.frame_idx ?? preview.raw?.frame_idx ?? await getFrameIdxAtTimestamp(dataset, videoId, currentMs));
      await onSubmit?.({ ...result, ...preview, video_id: videoId, frame_id: exactFrameId, frame_idx: exactFrameId,
        timestamp: currentMs / 1000, image_url: preview.image_url || result.image_url,
        raw: { ...result.raw, ...preview.raw, frame_idx: exactFrameId } }, action);
    } catch (err) {
      setError(err.message);
    } finally {
      setActionBusy(false);
    }
  }

  async function copyId() {
    try {
      const exactFrameId = await getFrameIdxAtTimestamp(dataset, videoId, currentMs);
      await navigator.clipboard.writeText(`${videoId}, ${exactFrameId}`);
      setCopyState(`Copied ${exactFrameId}`);
    } catch {
      setCopyState("Copy failed");
    }
    window.clearTimeout(copyTimer.current);
    copyTimer.current = window.setTimeout(() => setCopyState(""), 1500);
  }

  return <div className="video-modal-backdrop" style={{ zIndex: 3000 + layer }} onClick={onClose}>
    <div className="video-modal video-modal--framework" role="dialog" aria-modal="true" aria-label={`Video ${videoId}`} onClick={(event) => event.stopPropagation()}>
      <div className="video-modal-header">
        <div><h3>{videoId}</h3><p>
          {activeFrame && <>Keyframe {activeFrame.keyframe_id.padStart(6, "0")} · </>}
          {frameIdx != null && <>Frame {String(frameIdx).padStart(6, "0")} · </>}
          {(currentMs / 1000).toFixed(3)}s · {currentMs} ms
        </p></div>
        <button type="button" aria-label="Close video" onClick={onClose}><X size={18} /></button>
      </div>
      <div className="video-modal-columns"><section className="video-modal-main">
        {videoUrl ? <video ref={videoRef} className="video-player" src={videoUrl} controls autoPlay={autoPlay}
          onLoadedMetadata={(event) => {
            const duration = event.currentTarget.duration;
            if (Number.isFinite(duration) && duration > 0) {
              const milliseconds = Math.round(duration * 1000);
              const target = Math.min(initialMs, milliseconds);
              setMediaDurationMs(milliseconds);
              setCurrentMs(target);
              event.currentTarget.currentTime = target / 1000;
            }
          }}
          onTimeUpdate={(event) => setCurrentMs(Math.round(event.currentTarget.currentTime * 1000))}
          onSeeked={(event) => setCurrentMs(Math.round(event.currentTarget.currentTime * 1000))}
          onError={() => setError("Cannot play this video. Check the media URL and browser codec support.")}
        /> : <div className="video-missing">Video file not available.</div>}

        <div className="video-keyframe-row">
          <button type="button" className="video-timeline-step" aria-label={`Seek back ${stepMs} milliseconds`}
            title={`−${stepMs} ms`} onClick={() => seekTo(currentMs - stepMs)}><ChevronLeft size={20} /></button>
          {strip.loading ? <div className="video-keyframe-status" role="status">Loading keyframes…</div>
            : strip.error ? <div className="video-keyframe-status" role="alert">{strip.error}</div>
            : strip.keyframes.length ? <VideoKeyframeStrip keyframes={strip.keyframes} activeIndex={activeIndex} onSeek={seekTo} />
            : <div className="video-keyframe-status">No keyframes with valid timestamps.</div>}
          <button type="button" className="video-timeline-step" aria-label={`Seek forward ${stepMs} milliseconds`}
            title={`+${stepMs} ms`} onClick={() => seekTo(currentMs + stepMs)}><ChevronRight size={20} /></button>
        </div>
        {activeFrame?.synthetic_timing && <p className="video-timing-notice">Mock timing: this video has no matching CSV mapping.</p>}
        {error && <p className="video-playback-error" role="alert">{error}</p>}

        <div className="video-controls-row">
          <div className="video-submit-row">
            <label>Seek step<select value={stepMs} onChange={(event) => setStepMs(Number(event.target.value))}>
              {[100, 250, 500, 1000].map((step) => <option key={step} value={step}>{step} ms</option>)}
            </select></label>
            <label>Exact ms<input aria-label="Exact video milliseconds" type="number" min="0" max={durationMs} value={currentMs} onChange={(event) => seekTo(event.target.value)} /></label>
          </div>
          <div className="video-action-row">
            <button type="button" onClick={copyId}><Copy size={16} />{copyState || "Copy ID"}</button>
            <button type="button" disabled={actionBusy} onClick={() => sendCurrent("queue")}><ListPlus size={16} />Queue</button>
            <button type="button" className="video-submit-button" disabled={actionBusy} onClick={() => sendCurrent("submit")}><Send size={16} />Submit KIS</button>
          </div>
        </div>
      </section></div>
    </div>
  </div>;
}
