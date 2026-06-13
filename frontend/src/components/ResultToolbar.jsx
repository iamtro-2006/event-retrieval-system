import { FolderClosed, FolderOpen, Grid3X3 } from "lucide-react";

export default function ResultToolbar({
  model,
  latency,
  columns,
  grouped,
  onColumnsChange,
  onGroupedChange,
}) {
  return (
    <div className="result-toolbar">
      <div className="toolbar-left">
        <span className="latency-text">
          ⚡ Pipeline: {model} | Latency: {latency ?? "--"}ms
        </span>

        <button
          className={`group-toggle ${grouped ? "active" : ""}`}
          onClick={() => onGroupedChange(!grouped)}
        >
          {grouped ? <FolderOpen size={16} /> : <FolderClosed size={16} />}
          <span>{grouped ? "Group Video Rows" : "Flat List"}</span>
        </button>
      </div>

      <div className="column-control">
        <Grid3X3 size={16} />
        <span>Số cột:</span>

        <input
          type="range"
          min="3"
          max="8"
          value={columns}
          onChange={(e) => onColumnsChange(Number(e.target.value))}
        />

        <strong>{columns}</strong>
      </div>
    </div>
  );
} 