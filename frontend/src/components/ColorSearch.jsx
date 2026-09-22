import { useEffect, useRef, useState } from "react";
import { Redo2, RotateCcw, Search, Undo2, X } from "lucide-react";
import { COLOR_PALETTE, EMPTY_COLOR_GRID, selectedColorCells } from "../utils/colorGrid";
const COLOR_HEX = Object.fromEntries(COLOR_PALETTE);

export function ColorGridEditor({ value, onChange, compact = false, onSearch, loading = false }) {
  const [selected, setSelected] = useState("blue");
  const [undo, setUndo] = useState([]);
  const [redo, setRedo] = useState([]);
  const painting = useRef(null);
  const commit = (next) => {
    if (next.every((color, index) => color === value[index])) return;
    setUndo((history) => [...history.slice(-29), [...value]]);
    setRedo([]);
    onChange(next);
  };
  const paint = (index, color = selected, record = true) => {
    if (value[index] === color) return;
    const next = [...value]; next[index] = color;
    if (record) commit(next); else onChange(next);
  };
  const undoOnce = () => {
    const previous = undo.at(-1); if (!previous) return;
    setUndo((items) => items.slice(0, -1)); setRedo((items) => [...items, [...value]]); onChange(previous);
  };
  const redoOnce = () => {
    const next = redo.at(-1); if (!next) return;
    setRedo((items) => items.slice(0, -1)); setUndo((items) => [...items, [...value]]); onChange(next);
  };
  return <div className={`color-grid-editor ${compact ? "is-compact" : ""}`}>
    <div className="color-grid-tools">
      <button type="button" onClick={() => commit([...EMPTY_COLOR_GRID])} title="Reset"><RotateCcw size={14} /> Reset</button>
      <button type="button" onClick={undoOnce} disabled={!undo.length} title="Undo"><Undo2 size={14} /></button>
      <button type="button" onClick={redoOnce} disabled={!redo.length} title="Redo"><Redo2 size={14} /></button>
      <span>{selectedColorCells(value).length}/25 cells</span>
    </div>
    <div className="color-editor-body">
      <div className="color-palette" aria-label="Dominant color palette">
        {COLOR_PALETTE.map(([name, hex]) => <button key={name} type="button" aria-label={name}
          aria-pressed={selected === name} title={name} style={{ "--swatch": hex }} onClick={() => setSelected(name)} />)}
      </div>
      <div className="dominant-color-grid" role="grid" aria-label="5 by 5 dominant color grid"
        onPointerUp={() => { painting.current = null; }} onPointerLeave={() => { painting.current = null; }}>
        {value.map((color, index) => <button key={index} type="button" role="gridcell"
          aria-label={`Row ${Math.floor(index / 5) + 1}, column ${index % 5 + 1}${color ? `: ${color}` : ": empty"}`}
          style={{ background: color ? COLOR_HEX[color] : undefined }}
          onContextMenu={(event) => { event.preventDefault(); paint(index, null); }}
          onPointerDown={(event) => {
            if (event.button === 2) return;
            painting.current = { color: selected, snapshot: [...value] };
            paint(index, selected);
          }}
          onPointerEnter={(event) => {
            if (!painting.current || !(event.buttons & 1)) return;
            paint(index, painting.current.color, false);
          }} />)}
      </div>
    </div>
    {onSearch && <button className="color-search-submit" type="button" disabled={loading || !selectedColorCells(value).length} onClick={onSearch}>
      <Search size={16} /> {loading ? "Searching..." : "Search colors"}
    </button>}
    <small>Click/drag to paint · right-click to clear a cell</small>
  </div>;
}

export default function ColorSearchModal({ open, value, onChange, onClose, onSearch, loading }) {
  useEffect(() => {
    if (!open) return undefined;
    const handle = (event) => {
      if (event.key === "Escape") onClose();
      if (event.key === "Enter" && selectedColorCells(value).length && !loading) onSearch();
    };
    window.addEventListener("keydown", handle);
    return () => window.removeEventListener("keydown", handle);
  }, [loading, onClose, onSearch, open, value]);
  if (!open) return null;
  return <div className="color-modal-backdrop" onClick={onClose}>
    <section className="color-modal" role="dialog" aria-modal="true" aria-label="Dominant color search" onClick={(event) => event.stopPropagation()}>
      <header><div><strong>Dominant Color Search</strong><span>Paint the remembered colors on the 5 × 5 frame grid</span></div>
        <button type="button" onClick={onClose} aria-label="Close color canvas"><X size={18} /></button></header>
      <ColorGridEditor value={value} onChange={onChange} onSearch={onSearch} loading={loading} />
    </section>
  </div>;
}
