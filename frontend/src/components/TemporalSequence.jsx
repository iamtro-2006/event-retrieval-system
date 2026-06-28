import { memo, useEffect, useMemo, useRef, useState } from "react";
import { ChevronLeft, ChevronRight, Images, Play, Search, Send } from "lucide-react";

function makeFrameResult(result, frame, idx) {
  const frameId = Number(frame.frame_id ?? frame.keyframe_id ?? frame.keyframe_id_int ?? idx);
  const timestamp = Number(frame.timestamp_sec ?? frame.timestamp ?? result.timestamp ?? 0);
  const score = Number(frame.score ?? frame.similarity ?? frame.candidate_score ?? result.similarity ?? 0);
  return {
    ...result,
    ...frame,
    id: `${result.id}_q${frame.sub_query_idx ?? idx}_${frameId}`,
    video_id: frame.video_id || result.video_id,
    video_url: frame.video_url || result.video_url,
    frame_id: frameId,
    frame_name: frame.frame_name || `${String(frameId).padStart(6, "0")}.jpg`,
    image_url: frame.image_url || result.image_url,
    timestamp,
    similarity: score,
    matched_sequence: [],
  };
}

const TemporalSequence = memo(function TemporalSequence({
  result,
  sequenceIndex = 0,
  selectedId,
  onSelect,
  onSubmit,
  onPlay,
  onSimilaritySearch,
  onSurroundingImages,
}) {
  const sequence = useMemo(
    () => (Array.isArray(result?.matched_sequence) ? result.matched_sequence : []),
    [result]
  );
  const viewportRef = useRef(null);
  const [page, setPage] = useState(0);
  const frameResults = useMemo(
    () => sequence.map((frame, idx) => makeFrameResult(result, frame, idx)),
    [result, sequence]
  );

  useEffect(() => {
    setPage(0);
    viewportRef.current?.scrollTo({ left: 0, behavior: "auto" });
  }, [result?.id, sequence.length]);

  if (!sequence.length) return null;

  const isCarousel = sequence.length > 3;
  const visibleCount = Math.min(sequence.length, 3);
  const maxPage = Math.max(0, sequence.length - visibleCount);

  function move(direction) {
    const viewport = viewportRef.current;
    if (!viewport) return;

    const next = Math.max(0, Math.min(page + direction, maxPage));
    const cards = viewport.querySelectorAll(".temporal-frame-card");
    const target = cards[next];

    setPage(next);

    if (target) {
      const left = target.offsetLeft - viewport.offsetLeft;
      viewport.scrollTo({ left, behavior: "smooth" });
      return;
    }

    const firstCard = cards[0];
    const styles = window.getComputedStyle(viewport);
    const gap = Number.parseFloat(styles.columnGap || styles.gap || "0") || 0;
    const step = (firstCard?.getBoundingClientRect().width || viewport.clientWidth / visibleCount) + gap;
    viewport.scrollTo({ left: next * step, behavior: "smooth" });
  }

  function syncPageFromScroll() {
    const viewport = viewportRef.current;
    if (!viewport || !isCarousel) return;

    const cards = Array.from(viewport.querySelectorAll(".temporal-frame-card"));
    if (!cards.length) return;

    const currentLeft = viewport.scrollLeft;
    let nearestIndex = 0;
    let nearestDistance = Number.POSITIVE_INFINITY;

    cards.forEach((card, index) => {
      const distance = Math.abs(card.offsetLeft - viewport.offsetLeft - currentLeft);
      if (distance < nearestDistance) {
        nearestDistance = distance;
        nearestIndex = index;
      }
    });

    setPage(Math.min(nearestIndex, maxPage));
  }

  return (
    <section className={`temporal-sequence-block temporal-events-${Math.min(sequence.length, 6)}`} style={{ "--sequence-event-total": sequence.length }}>
      <div className="temporal-sequence-toolbar temporal-sequence-header">
        <div>
          <strong>Sequence #{sequenceIndex + 1}</strong>
          <span className="temporal-sequence-count">{sequence.length} events</span>
        </div>
        <span className="temporal-sequence-meta">
          {result.video_id} · {Number(result.temporal?.duration_sec ?? 0).toFixed(2)}s
        </span>
        <div className="temporal-sequence-actions">
          <button type="button" className="temporal-pill-btn" onClick={() => onPlay?.(result)}>
            <Play size={13} fill="currentColor" /><span>Play</span>
          </button>
          <button type="button" className="temporal-pill-btn temporal-submit-btn" onClick={() => onSubmit?.(result)}>
            <Send size={13} /><span>Submit</span>
          </button>
        </div>
      </div>

      <div className={`temporal-carousel-shell ${isCarousel ? "is-carousel" : "is-fluid"}`}>
        {isCarousel && (
          <button className="temporal-carousel-nav left" type="button" disabled={page === 0} onClick={() => move(-1)}>
            <ChevronLeft size={22} />
          </button>
        )}

        <div
          ref={viewportRef}
          className="temporal-frame-row"
          style={{ "--event-count": visibleCount }}
          onScroll={syncPageFromScroll}
        >
          {sequence.map((frame, idx) => {
            const frameResult = frameResults[idx];
            const isSelected = frameResult.id === selectedId;
            return (
              <article
                key={frameResult.id}
                className={`temporal-frame-card ${isSelected ? "selected" : ""}`}
                onClick={() => onSelect?.(frameResult)}
              >
                <div className="temporal-frame-thumb">
                  <img src={frameResult.image_url} alt={`Event ${idx + 1}`} loading="lazy" decoding="async" />
                  <span className="temporal-query-badge">E{idx + 1}</span>
                  <span className="temporal-score-badge">{Number(frameResult.similarity).toFixed(3)}</span>
                </div>
                <div className="temporal-frame-footer">
                  <strong>{frameResult.video_id}/{String(frameResult.frame_id).padStart(6, "0")}</strong>
                  <span>{frameResult.timestamp.toFixed(2)}s</span>
                </div>
                <div className="temporal-frame-actions-row">
                  <button type="button" onClick={(e) => { e.stopPropagation(); onSimilaritySearch?.(frameResult); }}>
                    <Search size={12} /><span>Similar</span>
                  </button>
                  <button type="button" onClick={(e) => { e.stopPropagation(); onSurroundingImages?.(frameResult); }}>
                    <Images size={12} /><span>Surround</span>
                  </button>
                  <button type="button" onClick={(e) => { e.stopPropagation(); onSubmit?.(frameResult); }}>
                    <Send size={12} /><span>Submit</span>
                  </button>
                </div>
                {frame.sub_query && <div className="temporal-frame-query" title={frame.sub_query}>{frame.sub_query}</div>}
              </article>
            );
          })}
        </div>

        {isCarousel && (
          <button className="temporal-carousel-nav right" type="button" disabled={page >= maxPage} onClick={() => move(1)}>
            <ChevronRight size={22} />
          </button>
        )}
      </div>
      {isCarousel && <div className="temporal-carousel-status">{page + 1}–{Math.min(page + visibleCount, sequence.length)} / {sequence.length}</div>}
    </section>
  );
});

export default TemporalSequence;
