import { useState } from "react";
import { X, Video } from "lucide-react";
import VideoModal from "./VideoModal";
import { getVideoPreview } from "../api/retrievalAPI";

export default function PreviewVideoModal({ open, onClose, onSubmit }) {
  const [videoId, setVideoId] = useState("");
  const [unit, setUnit] = useState("frame");
  const [value, setValue] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  if (!open) return null;
  if (result) return <VideoModal open result={result} onClose={() => { setResult(null); }} onSubmit={onSubmit} />;

  async function handlePreview(e) {
    e.preventDefault();
    setError("");
    if (!videoId.trim() || value === "") return setError("Nhập video và giá trị frame/ms.");
    setLoading(true);
    try {
      const data = await getVideoPreview(videoId.trim(), unit === "frame" ? { frameId: value } : { timestampMs: value });
      setResult(data);
    } catch (err) { setError(err.message); }
    finally { setLoading(false); }
  }

  return <div className="preview-modal-backdrop" onClick={onClose}>
    <form className="preview-modal" onSubmit={handlePreview} onClick={(e) => e.stopPropagation()}>
      <header><div><Video size={18} /><strong>Preview video</strong></div><button type="button" onClick={onClose}><X size={18} /></button></header>
      <label>Video ID<input value={videoId} onChange={(e) => setVideoId(e.target.value)} placeholder="L21_V001" autoFocus /></label>
      <div className="preview-unit-row"><label>Đơn vị<select value={unit} onChange={(e) => setUnit(e.target.value)}><option value="frame">frame_id</option><option value="ms">ms</option></select></label><label>Giá trị<input type="number" min="0" value={value} onChange={(e) => setValue(e.target.value)} placeholder={unit === "frame" ? "125" : "789022"} /></label></div>
      {error && <p className="preview-error">{error}</p>}
      <button className="preview-submit" disabled={loading}>{loading ? "Đang tải…" : "Mở preview"}</button>
    </form>
  </div>;
}
