import { useState } from "react";
import { X, Video } from "lucide-react";
import VideoModal from "./VideoModal";
import { getVideoPreview } from "../api/retrievalAPI";

export default function PreviewVideoModal({ open, dataset, onClose, onSubmit }) {
  const [videoId, setVideoId] = useState("");
  const [unit, setUnit] = useState("frame");
  const [value, setValue] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  if (!open) return null;
  if (result) return <VideoModal open result={result} dataset={dataset} autoPlay={false} onClose={() => { setResult(null); }} onSubmit={onSubmit} />;

  async function handlePreview(e) {
    e.preventDefault();
    setError("");
    if (!videoId.trim() || value === "") return setError("Enter a video ID and a frame or timestamp value.");
    setLoading(true);
    try {
      const data = await getVideoPreview(dataset, videoId.trim(), unit === "frame" ? { frameId: value } : { timestampMs: value });
      setResult(data);
    } catch (err) { setError(err.message); }
    finally { setLoading(false); }
  }

  return <div className="preview-modal-backdrop" onClick={onClose}>
    <form className="preview-modal" onSubmit={handlePreview} onClick={(e) => e.stopPropagation()}>
      <header><div><Video size={18} /><strong>Preview video</strong></div><button type="button" onClick={onClose}><X size={18} /></button></header>
      <label>Video ID<input value={videoId} onChange={(e) => setVideoId(e.target.value)} placeholder="L21_V001" autoFocus /></label>
      <div className="preview-unit-row"><label>Unit<select value={unit} onChange={(e) => setUnit(e.target.value)}><option value="frame">Frame index</option><option value="ms">Milliseconds</option></select></label><label>Value<input type="number" min="0" value={value} onChange={(e) => setValue(e.target.value)} placeholder={unit === "frame" ? "125" : "789022"} /></label></div>
      {error && <p className="preview-error">{error}</p>}
      <button className="preview-submit" disabled={loading}>{loading ? "Loading..." : "Open preview"}</button>
    </form>
  </div>;
}
