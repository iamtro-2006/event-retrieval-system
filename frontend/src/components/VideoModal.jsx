import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ChevronLeft, ChevronRight, Copy, ListPlus, Send, X } from "lucide-react";
import { getFrameIdxAtTimestamp, getVideoKeyframes, getVideoPreview } from "../api/retrievalAPI";

export default function VideoModal({ open, result, dataset, onClose, onSubmit, layer = 40, autoPlay = true }) {
  const videoRef = useRef(null);
  const timelineViewportRef = useRef(null);
  const zoomRef = useRef(1);
  const panRef = useRef(null);
  const panMovedRef = useRef(false);
  const manualSeekRef = useRef(false);
  const seekTimerRef = useRef(0);
  const autoLocatedRef = useRef(false);
  const initialMs = useMemo(() => {
    if (result?.is_preview_frame && Number(result?.fps) > 0 && result?.frame_idx != null) return Math.max(0, Math.round(Number(result.frame_idx) / Number(result.fps) * 1000));
    return Math.max(0, Math.round(Number(result?.timestamp ?? 0) * 1000));
  }, [result]);
  const currentMsRef = useRef(initialMs);
  const durationMsRef = useRef(1000);
  const [currentMs, setCurrentMs] = useState(initialMs), [durationMs, setDurationMs] = useState(Math.max(initialMs + 1000, 1000));
  const [stepMs, setStepMs] = useState(500), [zoom, setZoom] = useState(1);
  const [range, setRange] = useState([0, 100]), [markers, setMarkers] = useState([]);
  const [copyState, setCopyState] = useState("");
  const [dragHandle, setDragHandle] = useState(null);
  const [videoKeyframes, setVideoKeyframes] = useState([]);
  const seekTo = useCallback((value) => {
    const nextMs = Math.max(0, Math.min(Number(value) || 0, durationMs));
    manualSeekRef.current = true;
    setCurrentMs(Math.round(nextMs));
    if (videoRef.current) videoRef.current.currentTime = nextMs / 1000;
    requestAnimationFrame(() => {
      const node = timelineViewportRef.current;
      if (!node || node.scrollWidth <= node.clientWidth) return;
      const x = nextMs / Math.max(1, durationMs) * node.scrollWidth;
      const margin = Math.max(70, node.clientWidth * .18);
      if (x < node.scrollLeft + margin) node.scrollLeft = Math.max(0, x - margin);
      else if (x > node.scrollLeft + node.clientWidth - margin) node.scrollLeft = Math.min(node.scrollWidth - node.clientWidth, x - node.clientWidth + margin);
    });
    window.clearTimeout(seekTimerRef.current);
    seekTimerRef.current = window.setTimeout(() => { manualSeekRef.current = false; }, 180);
  }, [durationMs]);
  useEffect(() => {
    if (!open || !result?.video_id) return;
    let alive = true;
    getVideoKeyframes(dataset, result.video_id).then((data) => { if (alive) setVideoKeyframes(data.keyframes || []); }).catch(() => { if (alive) setVideoKeyframes([]); });
    return () => { alive = false; };
  }, [dataset, open, result?.video_id]);
  const centerTimelineAt = useCallback((ms, nextDuration) => {
    requestAnimationFrame(() => requestAnimationFrame(() => {
      const node = timelineViewportRef.current;
      if (!node) return;
      const ratio = Math.max(0, Math.min(1, ms / Math.max(1, nextDuration)));
      node.scrollLeft = Math.max(0, ratio * node.scrollWidth - node.clientWidth / 2);
    }));
  }, []);
  useEffect(() => {
    if (!open || autoLocatedRef.current || durationMs <= 1000) return;
    autoLocatedRef.current = true;
    const targetZoom = Math.max(1, Math.min(50, durationMs / 40000));
    setZoom(targetZoom);
    centerTimelineAt(initialMs, durationMs);
  }, [open, durationMs, initialMs, centerTimelineAt]);
  useEffect(() => { zoomRef.current = zoom; }, [zoom]);
  useEffect(() => { currentMsRef.current = currentMs; durationMsRef.current = durationMs; }, [currentMs, durationMs]);
  useEffect(() => {
    const node = timelineViewportRef.current;
    if (!node) return undefined;
    const blockBrowserZoom = (e) => {
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault();
        e.stopPropagation();
        setZoom((z) => {
          const next = Math.max(1, Math.min(50, z * (e.deltaY < 0 ? 1.12 : 0.89)));
          requestAnimationFrame(() => {
            const ratio = currentMsRef.current / Math.max(1, durationMsRef.current);
            node.scrollLeft = Math.max(0, ratio * node.scrollWidth - node.clientWidth / 2);
          });
          return next;
        });
      } else if (Math.abs(e.deltaY) > Math.abs(e.deltaX)) {
        e.preventDefault();
        node.scrollLeft += e.deltaY;
      }
    };
    const pauseWhileEditing = (e) => {
      if (e.target.closest(".timeline-handle, .timeline-playhead-button")) { manualSeekRef.current = true; panMovedRef.current = true; videoRef.current?.pause(); }
    };
    const finishEditing = () => { window.setTimeout(() => { manualSeekRef.current = false; }, 300); };
    const seekClickedKeyframe = (e) => {
      const marker = e.target.closest(".timeline-keyframe, .timeline-marker");
      if (!marker) return;
      const left = Number.parseFloat(marker.style.left);
      if (Number.isFinite(left)) { e.preventDefault(); e.stopPropagation(); panMovedRef.current = true; seekTo(left / 100 * durationMsRef.current); if (marker.classList.contains("timeline-marker") && videoRef.current) void videoRef.current.play(); window.setTimeout(() => { panMovedRef.current = false; }, 300); }
    };
    node.addEventListener("wheel", blockBrowserZoom, { passive: false });
    node.addEventListener("pointerdown", pauseWhileEditing, true);
    node.addEventListener("pointerup", finishEditing, true);
    node.addEventListener("pointercancel", finishEditing, true);
    node.addEventListener("click", seekClickedKeyframe, true);
    node.addEventListener("pointerdown", seekClickedKeyframe, true);
    return () => { node.removeEventListener("wheel", blockBrowserZoom); node.removeEventListener("pointerdown", pauseWhileEditing, true); node.removeEventListener("pointerup", finishEditing, true); node.removeEventListener("pointercancel", finishEditing, true); node.removeEventListener("click", seekClickedKeyframe, true); node.removeEventListener("pointerdown", seekClickedKeyframe, true); };
  }, [open, seekTo]);
  if (!open || !result) return null;
  const frameId = Math.max(0, Math.round(currentMs / 1000 * Number(result.fps || 25))), videoId = result.video_id;
  const listedKeyframeTimes = [...(result.keyframes || result.raw?.keyframes || result.matched_sequence || [])].map((item) => Number(item.timestamp ?? item.timestamp_sec ?? item.time ?? 0) * 1000).filter((ms) => ms >= 0 && ms <= durationMs);
  const hasKeyframeIdentity = !result.is_preview_timestamp && !result.is_preview_frame && (result.keyframe_id != null || result.keyframe_id_int != null || result.raw?.keyframe_id != null || result.raw?.keyframe_id_int != null);
  const primaryKeyframeMs = hasKeyframeIdentity ? Math.max(0, Math.min(initialMs, durationMs)) : null;
  const keyframeTimes = [...new Set([...videoKeyframes.map((item) => Number(item.timestamp) * 1000), ...listedKeyframeTimes, ...(hasKeyframeIdentity ? [primaryKeyframeMs] : [])])].sort((a, b) => hasKeyframeIdentity ? Math.abs(a - primaryKeyframeMs) - Math.abs(b - primaryKeyframeMs) : 0);
  const startMs = range[0] / 100 * durationMs, endMs = range[1] / 100 * durationMs;
  const rulerStepMs = (() => {
    if (zoom >= 50) return 100;
    if (zoom >= 45) return 250;
    if (zoom >= 35) return 500;
    const raw = durationMs / Math.max(12, 18 * zoom);
    const magnitude = 10 ** Math.floor(Math.log10(Math.max(1, raw)));
    const normalized = raw / magnitude;
    const nice = normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10;
    return Math.max(1, nice * magnitude);
  })();
  const rulerCount = Math.min(20000, Math.ceil(durationMs / rulerStepMs) + 1);
  const rulerLabelEvery = zoom >= 50 ? 10 : zoom >= 45 ? 5 : zoom >= 35 ? 3 : 3;
  const updateHandle = (e) => { if (!dragHandle) return; const rect = e.currentTarget.getBoundingClientRect(); const pct = Math.max(0, Math.min(100, ((e.clientX - rect.left) / rect.width) * 100)); if (dragHandle === "current") { seekTo(pct / 100 * durationMs); return; } setRange(([start, end]) => dragHandle === "start" ? [Math.min(pct, end), end] : [start, Math.max(pct, start)]); };
  const releaseHandle = () => setDragHandle(null);
  const beginPan = (e) => {
    if (e.target.closest(".timeline-handle, .timeline-playhead-button, .timeline-keyframe")) return;
    panRef.current = { x: e.clientX, scrollLeft: timelineViewportRef.current.scrollLeft };
    panMovedRef.current = false;
    e.currentTarget.setPointerCapture?.(e.pointerId);
  };
  const movePan = (e) => {
    if (!panRef.current || dragHandle || !timelineViewportRef.current) return;
    if (Math.abs(e.clientX - panRef.current.x) > 3) panMovedRef.current = true;
    timelineViewportRef.current.scrollLeft = panRef.current.scrollLeft - (e.clientX - panRef.current.x);
  };
  const endPan = () => { panRef.current = null; setTimeout(() => { panMovedRef.current = false; }, 250); };
  const boundedPlayback = (ms) => { if (manualSeekRef.current || dragHandle) return ms; if ((ms < startMs || ms >= endMs) && videoRef.current?.paused === false) { const reset = startMs; setCurrentMs(Math.round(reset)); if (videoRef.current) { videoRef.current.currentTime = reset / 1000; void videoRef.current.play(); } return reset; } return ms; };
  const addMarker = (ms = currentMs, exactFrameId = null) => setMarkers((p) => [...p, { ms, frameId: exactFrameId, type: "star" }]);
  async function sendCurrent(action) {
    const preview = await getVideoPreview(dataset, videoId, { timestampMs: currentMs });
    const exactFrameId = Number(preview.frame_idx ?? preview.raw?.frame_idx ?? await getFrameIdxAtTimestamp(dataset, videoId, currentMs));
    onSubmit?.({ ...result, ...preview, video_id: videoId, frame_id: exactFrameId, frame_idx: exactFrameId, timestamp: currentMs / 1000, image_url: preview.image_url || result.image_url, raw: { ...result.raw, ...preview.raw, frame_idx: exactFrameId } }, action);
    addMarker(currentMs, exactFrameId);
  }
  async function copyId() { try { const exactFrameId = await getFrameIdxAtTimestamp(dataset, videoId, currentMs); await navigator.clipboard.writeText(`${videoId}, ${exactFrameId}`); setCopyState(`Copied ${exactFrameId}`); } catch { setCopyState("Copy failed"); } setTimeout(() => setCopyState(""), 1500); }
  const pos = (ms) => ({ left: `${ms / durationMs * 100}%` });
  return <div className="video-modal-backdrop" style={{ zIndex: 3000 + layer }} onClick={onClose}><div className={`video-modal video-modal--framework ${hasKeyframeIdentity ? "has-primary-keyframe" : ""}`} onClick={(e) => e.stopPropagation()}>
    <div className="video-modal-header"><div><h3>{videoId}</h3><p>Frame {String(frameId).padStart(6, "0")} · {(currentMs / 1000).toFixed(3)}s · {currentMs} ms</p></div><button type="button" onClick={onClose}><X size={18} /></button></div>
    <div className="video-modal-columns"><section className="video-modal-main">{result.video_url && result.video_url !== "#" ? <video ref={videoRef} className="video-player" src={`${result.video_url}#t=${initialMs / 1000}`} controls autoPlay={autoPlay} onLoadedMetadata={(e) => { const d = Math.max(1000, Math.round(e.currentTarget.duration * 1000)); const targetZoom = Math.max(1, Math.min(50, d / 40000)); setDurationMs(d); setZoom(targetZoom); e.currentTarget.currentTime = initialMs / 1000; centerTimelineAt(initialMs, d, targetZoom); }} onTimeUpdate={(e) => setCurrentMs(Math.round(boundedPlayback(e.currentTarget.currentTime * 1000)))} /> : <div className="video-missing">Video file not available.</div>}
      <div className="timeline-shell"><div ref={timelineViewportRef} className="timeline-viewport" onPointerDown={beginPan} onPointerMove={movePan} onPointerUp={endPan} onPointerCancel={endPan}><div className="timeline-ruler timeline-ruler--zoomed" style={{ width: `${zoom * 100}%` }}>{Array.from({ length: rulerCount }, (_, i) => { const ms = Math.min(durationMs, i * rulerStepMs); const major = i % rulerLabelEvery === 0 || i === rulerCount - 1; return <span key={i} className={major ? "timeline-ruler-label" : "timeline-ruler-tick"} style={{ left: `${ms / durationMs * 100}%` }}>{major ? `${(ms / 1000).toFixed(ms < 10000 ? 2 : 1)}s` : ""}</span>; })}<span className="timeline-ruler-current" style={pos(currentMs)}>{(currentMs / 1000).toFixed(3)}s</span></div><div className="timeline-track" style={{ width: `${zoom * 100}%` }} onPointerMove={updateHandle} onPointerUp={releaseHandle} onPointerLeave={releaseHandle} onClick={(e) => { if (!dragHandle && !panMovedRef.current) seekTo((e.nativeEvent.offsetX / e.currentTarget.clientWidth) * durationMs); }}><div className="timeline-density">{Array.from({ length: 101 }, (_, i) => <i key={i} style={{ left: `${i}%` }} />)}</div><div className="timeline-selected" style={{ left: `${range[0]}%`, width: `${range[1] - range[0]}%` }} />{keyframeTimes.map((ms, i) => <span key={`keyframe-${i}`} className="timeline-keyframe" style={pos(ms)} title="Keyframe model" />)}<button type="button" aria-label="Adjust range start" className="timeline-handle timeline-handle-start" style={{ left: `${range[0]}%` }} onPointerDown={(e) => { e.stopPropagation(); setDragHandle("start"); }} /><button type="button" aria-label="Adjust range end" className="timeline-handle timeline-handle-end" style={{ left: `${range[1]}%` }} onPointerDown={(e) => { e.stopPropagation(); setDragHandle("end"); }} />{markers.map((m, i) => <span key={`${m.ms}-${i}`} className="timeline-marker" style={pos(m.ms)}>★</span>)}<button type="button" aria-label="Adjust current time" className="timeline-playhead timeline-playhead-button" style={pos(currentMs)} onPointerDown={(e) => { e.stopPropagation(); setDragHandle("current"); }} /></div></div><small>Ticks every {rulerStepMs >= 1000 ? `${rulerStepMs / 1000}s` : `${rulerStepMs}ms`} · sparse labels for readability · position {(currentMs / 1000).toFixed(3)}s · {zoom.toFixed(1)}×</small></div>
      <div className="video-seek-row"><button type="button" onClick={() => seekTo(currentMs - stepMs)}><ChevronLeft size={18} /> -{stepMs} ms</button><input type="range" min="0" max={durationMs} value={currentMs} onChange={(e) => seekTo(e.target.value)} /><button type="button" onClick={() => seekTo(currentMs + stepMs)}>+{stepMs} ms <ChevronRight size={18} /></button></div>
      <div className="video-submit-row"><label>Seek step<select value={stepMs} onChange={(e) => setStepMs(Number(e.target.value))}><option value="100">100 ms</option><option value="250">250 ms</option><option value="500">500 ms</option><option value="1000">1000 ms</option></select></label><label>Exact ms<input type="number" min="0" max={durationMs} value={currentMs} onChange={(e) => seekTo(e.target.value)} /></label></div>
      <div className="video-action-row"><button type="button" onClick={copyId}><Copy size={16} /> {copyState || "Copy ID"}</button><button type="button" onClick={() => sendCurrent("queue")}><ListPlus size={16} /> Queue</button><button type="button" className="video-submit-button" onClick={() => sendCurrent("submit")}><Send size={16} /> Submit KIS</button></div>
    </section></div>
  </div></div>;
}
