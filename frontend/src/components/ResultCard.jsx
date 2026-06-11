import { Play, ThumbsDown, ThumbsUp } from "lucide-react";

export default function ResultCard({ result, selected, onSelect }) {
  const score = (result.similarity * 100).toFixed(1);
  const label = `${result.video_id}/${String(result.frame_id).padStart(6, "0")}`;

  function handlePlay(e) {
    e.stopPropagation();

    if (result.video_url && result.video_url !== "#") {
      window.open(`${result.video_url}#t=${result.timestamp}`, "_blank");
    } else {
      alert(`Play video tại timestamp ${result.timestamp.toFixed(2)}s`);
    }
  }

  function handleSubmit(e) {
    e.stopPropagation();
    alert(`Submit frame: ${label}`);
  }

  return (
    <article
      className={`result-card ${selected ? "selected" : ""}`}
      onClick={() => onSelect(result)}
    >
      <div className="thumbnail-box">
        <img src={result.image_url} alt={label} />

        <span className="score-badge">{score}%</span>

        <button className="vote-button like">
          <ThumbsUp size={12} />
        </button>

        <button className="vote-button dislike">
          <ThumbsDown size={12} />
        </button>

        <button className="play-button" onClick={handlePlay}>
          <Play size={16} fill="currentColor" />
        </button>
      </div>

      <div className="card-footer">
        <span title={label}>{label}</span>
        <button onClick={handleSubmit}>Submit</button>
      </div>
    </article>
  );
}