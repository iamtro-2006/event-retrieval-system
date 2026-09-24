import { useMemo, useState } from "react";
import { GripVertical, Send, Trash2, X } from "lucide-react";

const TASKS = ["kis", "trake", "qa"];

function mediaAnswer(item) {
  const timeMs = Math.round(Number(item.timestamp || 0) * 1000);
  return { mediaItemName: item.video_id, start: timeMs, end: timeMs };
}

export default function SubmissionQueueModal({ open, items, onClose, onChange, onSubmit, submitting }) {
  const [task, setTask] = useState("kis");
  const [answer, setAnswer] = useState("");
  const [dragIndex, setDragIndex] = useState(null);
  const preview = useMemo(() => {
    if (!items.length) return null;
    if (task === "qa") {
      const item = items[0];
      const timeMs = Math.round(Number(item.timestamp || 0) * 1000);
      return { answerSets: [{ answers: [{ text: `QA-${answer || "<ANSWER>"}-${item.video_id}-${timeMs}` }] }] };
    }
    if (task === "trake") {
      const videoId = items[0].video_id;
      const frameIds = items.map((item) => Number(item.frame_id || 0)).join(",");
      return { answerSets: [{ answers: [{ text: `TR-${videoId}-${frameIds}` }] }] };
    }
    const selected = task === "kis" ? [items[0]] : items;
    return { answerSets: [{ answers: selected.map(mediaAnswer) }] };
  }, [answer, items, task]);
  const invalidTrake = task === "trake" && new Set(items.map((item) => item.video_id)).size > 1;
  if (!open) return null;

  const move = (from, to) => {
    if (from == null || from === to) return;
    const next = [...items];
    const [item] = next.splice(from, 1);
    next.splice(to, 0, item);
    onChange(next);
  };

  return <div className="submission-queue-backdrop" onClick={onClose}>
    <section className="submission-queue-modal" onClick={(event) => event.stopPropagation()}>
      <header>
        <div><h2>Submission Queue</h2><p>{items.length} frame(s)</p></div>
        <button type="button" onClick={onClose} aria-label="Close"><X size={18} /></button>
      </header>

      <nav className="submission-task-tabs" aria-label="Submission task">
        {TASKS.map((value) => <button key={value} type="button" className={task === value ? "active" : ""} onClick={() => setTask(value)}>{value.toUpperCase()}</button>)}
      </nav>

      <div className="submission-queue-list">
        {items.map((item, index) => <article key={item.queue_id} draggable
          onDragStart={() => setDragIndex(index)} onDragOver={(event) => event.preventDefault()}
          onDrop={() => { move(dragIndex, index); setDragIndex(null); }}>
          <GripVertical size={18} className="queue-grip" />
          <img src={item.image_url} alt={`${item.video_id} frame ${item.frame_id}`} />
          <div><strong>{item.video_id}</strong><span>Frame {item.frame_id}</span><span>{Number(item.timestamp || 0).toFixed(3)}s · {Math.round(Number(item.timestamp || 0) * 1000)}ms</span></div>
          <button type="button" onClick={() => onChange(items.filter((_, itemIndex) => itemIndex !== index))} aria-label="Remove frame"><Trash2 size={16} /></button>
        </article>)}
        {!items.length && <p className="submission-empty">Add a keyframe or queue a position from the video player.</p>}
      </div>

      {task === "qa" && <label className="submission-answer">Answer<textarea value={answer} onChange={(event) => setAnswer(event.target.value)} placeholder="Enter the QA answer" /></label>}
      {invalidTrake && <p className="submission-validation-error">TRAKE chỉ chấp nhận các frame thuộc cùng một video.</p>}
      <div className="submission-preview"><strong>JSON preview</strong><pre>{preview ? JSON.stringify(preview, null, 2) : "Queue is empty"}</pre></div>
      <footer>
        <button type="button" onClick={() => onChange([])} disabled={!items.length}>Clear</button>
        <button type="button" className="submission-send" disabled={!items.length || submitting || invalidTrake || (task === "qa" && !answer.trim())}
          onClick={() => onSubmit({ task, items, answer })}><Send size={16} />{submitting ? "Submitting…" : `Submit ${task.toUpperCase()}`}</button>
      </footer>
    </section>
  </div>;
}
