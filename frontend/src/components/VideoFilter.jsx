import { createContext, useContext, useEffect, useMemo, useRef, useState } from "react";
import { History, Search, ThumbsDown, ThumbsUp, X } from "lucide-react";
import { getVideoIds } from "../api/retrievalAPI";

const VideoFilterContext = createContext(null);
// Shared by the composer, sidebar and every frame, including modal frames.
// eslint-disable-next-line react-refresh/only-export-components
export const useVideoFilter = () => useContext(VideoFilterContext);

export function VideoFilterProvider({ children }) {
  const [ids, setIds] = useState(() => {
    try {
      const saved = JSON.parse(localStorage.getItem("vireta-video-cache") || "[]");
      return Array.isArray(saved) ? [...new Set(saved.filter((id) => typeof id === "string" && id.trim()))] : [];
    } catch { return []; }
  });
  const [enabled, setEnabled] = useState(false);
  const [open, setOpen] = useState(false);
  const [notice, setNotice] = useState("");
  const currentIds = useRef(ids);
  const timer = useRef(null);
  const notify = (message) => {
    setNotice(message);
    clearTimeout(timer.current);
    timer.current = setTimeout(() => setNotice(""), 3500);
  };
  useEffect(() => () => clearTimeout(timer.current), []);
  const save = (value) => {
    const next = typeof value === "function" ? value(currentIds.current) : value;
    currentIds.current = next;
    setIds(next);
    try { localStorage.setItem("vireta-video-cache", JSON.stringify(next)); }
    catch { notify("The video cache could not be saved for the next page load."); }
  };
  const add = (id) => {
    if (!id) return;
    if (currentIds.current.includes(id)) { notify(`${id} is already in the cache.`); return; }
    save([...currentIds.current, id]);
    notify(`Added ${id}.`);
  };
  const remove = (id) => save(currentIds.current.filter((value) => value !== id));
  const activeIds = useMemo(() => enabled && ids.length ? ids : [], [enabled, ids]);
  return <VideoFilterContext.Provider value={{ ids, enabled, setEnabled, open, setOpen, add, remove, save, notify, activeIds }}>
    {children}
    {notice && <div className="video-cache-notice" role="status">{notice}</div>}
  </VideoFilterContext.Provider>;
}

export function VideoFilterBar({ dataset = "aic" }) {
  const { ids, enabled, setEnabled, add, save, notify, open, setOpen, remove } = useVideoFilter();
  const [catalog, setCatalog] = useState([]);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState("");
  const load = async () => {
    setLoading(true);
    setError("");
    try { const values = await getVideoIds(dataset); setCatalog(values); setLoaded(true); return values; }
    catch (err) { setError(err.message); return null; }
    finally { setLoading(false); }
  };
  const matches = useMemo(() => catalog.filter((id) => id.toLowerCase().includes(query.trim().toLowerCase())), [catalog, query]);
  const submit = (event) => {
    event.preventDefault();
    const exact = catalog.find((id) => id.toLowerCase() === query.trim().toLowerCase());
    const id = exact || (matches.length === 1 ? matches[0] : null);
    if (!id) { notify("Select an exact video ID from the metadata, then press Enter."); return; }
    add(id); setQuery("");
  };
  return <>
    <div className="video-filter-bar">
      <form onSubmit={submit}>
        <Search size={13} />
        <input aria-label="Search video ID metadata" placeholder="Search video ID…" value={query}
          onFocus={() => { if (!loaded && !loading) void load(); }}
          onChange={(event) => setQuery(event.target.value)} list="video-id-options" />
        <datalist id="video-id-options">{matches.slice(0, 100).map((id) => <option key={id} value={id} />)}</datalist>
      </form>
      <button type="button" disabled={loading} onClick={async () => {
        const values = await load();
        if (values) { save((current) => [...new Set([...current, ...values])]); notify(`Imported ${values.length} video IDs from metadata.`); }
      }}>{loading ? "Loading…" : "Import all"}</button>
      <button type="button" className={enabled ? "is-active" : ""} aria-pressed={enabled}
        title="Immediately filter visible results by the video ID cache. An empty cache shows all results."
        onClick={() => setEnabled((value) => !value)}>
        Filter{ids.length > 0 ? ` (${ids.length})` : ""}
      </button>
      <button type="button" onClick={() => save([])} disabled={!ids.length}>Clear all</button>
      <button type="button" onClick={() => setOpen(!open)} aria-expanded={open}>Cache ({ids.length})</button>
      {error && <span role="alert">{error}</span>}
    </div>
    {open && <div className="video-cache-backdrop" onClick={() => setOpen(false)}>
      <section className="video-cache-panel" role="dialog" aria-modal="true" aria-label="Video ID cache" onClick={(event) => event.stopPropagation()}
        onKeyDown={(event) => {
          if (event.key === "Escape") setOpen(false);
          if (event.key === "Tab") {
            const buttons = event.currentTarget.querySelectorAll("button");
            const first = buttons[0];
            const last = buttons[buttons.length - 1];
            if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
            else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
          }
        }}>
        <header><strong><History size={16} /> Video ID cache ({ids.length})</strong><button autoFocus type="button" onClick={() => setOpen(false)} aria-label="Close cache"><X size={18} /></button></header>
        <p>The Filter button immediately limits visible results and future searches to cached video IDs. An empty cache shows all videos.</p>
        {!ids.length ? <p>The cache is empty.</p> : <ol>{ids.map((id) => <li key={id}><span>{id}</span><button type="button" onClick={() => remove(id)} aria-label={`Remove ${id}`}><X size={14} /></button></li>)}</ol>}
      </section>
    </div>}
  </>;
}

export function VideoVotes({ videoId }) {
  const filter = useVideoFilter();
  if (!filter) return null;
  return <>
    <button className="vote-button like" type="button" aria-label={`Add ${videoId} to cache`} title="Add video to cache"
      aria-pressed={filter.ids.includes(videoId)} onClick={(event) => { event.stopPropagation(); filter.add(videoId); }}><ThumbsUp size={13} /></button>
    <button className="vote-button dislike" type="button" aria-label={`Remove ${videoId} from cache`} title="Remove video from cache"
      onClick={(event) => { event.stopPropagation(); filter.remove(videoId); }}><ThumbsDown size={13} /></button>
  </>;
}
