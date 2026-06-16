import { memo, useMemo, useState } from "react";
import {
  ChevronLeft,
  ChevronRight,
  Play,
  Search,
  Images,
  Send,
} from "lucide-react";
import ResultCard from "./ResultCard";
import { groupByVideoSorted } from "../utils/groupByVideo";

function makeFrameResultFromSequence(result, frame, idx) {
  const frameId = Number(
    frame.frame_id ??
      frame.keyframe_id ??
      frame.keyframe_id_int ??
      idx
  );

  const timestamp = Number(frame.timestamp_sec ?? frame.timestamp ?? result.timestamp ?? 0);
  const score = Number(frame.score ?? frame.similarity ?? frame.candidate_score ?? result.similarity ?? 0);
  const frameResultId = `${result.id}_q${frame.sub_query_idx ?? idx}_${frameId}`;

  return {
    ...result,
    ...frame,
    id: frameResultId,
    video_id: frame.video_id || result.video_id,
    video_url: frame.video_url || result.video_url,
    frame_id: frameId,
    frame_name: frame.frame_name || `${String(frameId).padStart(6, "0")}.jpg`,
    image_url: frame.image_url || result.image_url,
    timestamp,
    similarity: score,
    temporal: {
      ...result.temporal,
      start_time: timestamp,
      end_time: timestamp,
    },
    matched_sequence: [],
  };
}

const TemporalSequenceBlock = memo(function TemporalSequenceBlock({
  result,
  sequenceIndex,
  selectedId,
  onSelect,
  onSubmit,
  onSimilaritySearch,
  onSurroundingImages,
}) {
  const sequence = Array.isArray(result.matched_sequence) ? result.matched_sequence : [];

  const frameResults = useMemo(
    () => sequence.map((frame, idx) => makeFrameResultFromSequence(result, frame, idx)),
    [result, sequence]
  );

  function handlePlaySequence(e) {
    e.stopPropagation();
    const startTime = result.temporal?.start_time ?? result.timestamp ?? 0;

    if (result.video_url && result.video_url !== "#") {
      window.open(`${result.video_url}#t=${Number(startTime).toFixed(2)}`, "_blank");
    }
  }

  function handleSubmitSequence(e) {
    e.stopPropagation();
    onSubmit?.(result);
  }

  return (
    <section className="temporal-sequence-block">
      <div className="temporal-sequence-toolbar temporal-sequence-header">
        <div className="temporal-sequence-title">
          Sequence #{sequenceIndex + 1}
        </div>

        <div className="temporal-sequence-center">
          <span className="temporal-sequence-meta">
            {result.video_id} · {Number(result.temporal?.duration_sec ?? 0).toFixed(2)}s
          </span>

          <button
            type="button"
            className="temporal-pill-btn temporal-play-btn"
            onClick={handlePlaySequence}
            title="Play sequence"
          >
            <Play size={13} fill="currentColor" />
            <span>Play</span>
          </button>
        </div>

        <div className="temporal-sequence-actions">
          <button
            type="button"
            className="temporal-pill-btn temporal-submit-btn"
            onClick={handleSubmitSequence}
            title="Submit sequence start frame"
          >
            <Send size={13} />
            <span>Submit</span>
          </button>
        </div>
      </div>

      <div className="temporal-frame-row">
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
                <img
                  src={frameResult.image_url}
                  alt={`Q${idx + 1}`}
                  loading="lazy"
                  decoding="async"
                  draggable="false"
                />

                <span className="temporal-query-badge">Q{idx + 1}</span>
                <span className="temporal-score-badge">
                  {(Number(frame.score ?? frameResult.similarity ?? 0) * 100).toFixed(1)}%
                </span>
              </div>

              <div className="temporal-frame-footer">
                <strong>
                  {frameResult.video_id}/{String(frameResult.frame_id ?? 0).padStart(6, "0")}
                </strong>
                <span>{Number(frame.timestamp_sec ?? frameResult.timestamp ?? 0).toFixed(2)}s</span>
              </div>

              <div className="temporal-frame-actions-row">
                <button
                  type="button"
                  title="Similarity search"
                  onClick={(e) => {
                    e.stopPropagation();
                    onSimilaritySearch?.(frameResult);
                  }}
                >
                  <Search size={12} />
                  <span>Similar</span>
                </button>

                <button
                  type="button"
                  title="Surrounding images"
                  onClick={(e) => {
                    e.stopPropagation();
                    onSurroundingImages?.(frameResult);
                  }}
                >
                  <Images size={12} />
                  <span>Surround</span>
                </button>

                <button
                  type="button"
                  title="Submit frame"
                  onClick={(e) => {
                    e.stopPropagation();
                    onSubmit?.(frameResult);
                  }}
                >
                  <Send size={12} />
                  <span>Submit</span>
                </button>
              </div>

              {frame.sub_query && (
                <div className="temporal-frame-query" title={frame.sub_query}>
                  {frame.sub_query}
                </div>
              )}
            </article>
          );
        })}
      </div>
    </section>
  );
});

export default function GroupedResults({
  results,
  columns,
  selectedId,
  onSelect,
  onSubmit,
  onSimilaritySearch,
  onSurroundingImages,
}) {
  const groups = useMemo(() => groupByVideoSorted(results), [results]);
  const [pageByVideo, setPageByVideo] = useState({});

  function getCurrentPage(videoId) {
    return pageByVideo[videoId] ?? 0;
  }

  function changePage(videoId, direction, totalPages) {
    const currentPage = getCurrentPage(videoId);
    const nextPage = Math.min(Math.max(currentPage + direction, 0), totalPages - 1);

    setPageByVideo((prev) => ({
      ...prev,
      [videoId]: nextPage,
    }));
  }

  return (
    <div className="grouped-results">
      {groups.map(({ videoId, items }) => {
        const hasTemporal = items.some(
          (item) => Array.isArray(item.matched_sequence) && item.matched_sequence.length > 0
        );

        if (hasTemporal) {
          return (
            <section key={videoId} className="video-strip-group">
              <div className="video-strip-label">
                <span>{videoId}</span>
              </div>

              <div className="video-strip-main">
                <div className="video-strip-topbar">
                  <span>{items.length} temporal sequences</span>
                </div>

                <div
                  className="temporal-sequences-grid grouped-temporal-sequences-grid"
                  style={{ "--sequence-cols": columns }}
                >
                  {items.map((result, idx) => (
                    <TemporalSequenceBlock
                      key={result.id}
                      result={result}
                      sequenceIndex={idx}
                      selectedId={selectedId}
                      onSelect={onSelect}
                      onSubmit={onSubmit}
                      onSimilaritySearch={onSimilaritySearch}
                      onSurroundingImages={onSurroundingImages}
                    />
                  ))}
                </div>
              </div>
            </section>
          );
        }

        const pageSize = Math.max(1, columns);
        const totalPages = Math.max(Math.ceil(items.length / pageSize), 1);
        const currentPage = Math.min(getCurrentPage(videoId), totalPages - 1);

        const pages = Array.from({ length: totalPages }, (_, pageIndex) => {
          const start = pageIndex * pageSize;
          return items.slice(start, start + pageSize);
        });

        return (
          <section key={videoId} className="video-strip-group">
            <div className="video-strip-label">
              <span>{videoId}</span>
            </div>

            <div className="video-strip-main">
              <div className="video-strip-topbar">
                <span>
                  {items.length} assets · page {currentPage + 1}/{totalPages}
                </span>
              </div>

              <div className="video-strip-content">
                <button
                  type="button"
                  className="group-nav-button left"
                  disabled={currentPage <= 0}
                  onClick={() => changePage(videoId, -1, totalPages)}
                >
                  <ChevronLeft size={22} />
                </button>

                <div className="video-group-viewport">
                  <div
                    className="video-group-track"
                    style={{
                      width: `${totalPages * 100}%`,
                      transform: `translateX(-${currentPage * (100 / totalPages)}%)`,
                    }}
                  >
                    {pages.map((pageItems, pageIndex) => (
                      <div
                        key={`${videoId}-page-${pageIndex}`}
                        className="video-group-page"
                        style={{
                          width: `${100 / totalPages}%`,
                          gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))`,
                        }}
                      >
                        {pageItems.map((result) => (
                          <ResultCard
                            key={result.id}
                            result={result}
                            selected={result.id === selectedId}
                            onSelect={onSelect}
                            onSubmit={onSubmit}
                            onSimilaritySearch={onSimilaritySearch}
                            onSurroundingImages={onSurroundingImages}
                          />
                        ))}
                      </div>
                    ))}
                  </div>
                </div>

                <button
                  type="button"
                  className="group-nav-button right"
                  disabled={currentPage >= totalPages - 1}
                  onClick={() => changePage(videoId, 1, totalPages)}
                >
                  <ChevronRight size={22} />
                </button>
              </div>
            </div>
          </section>
        );
      })}
    </div>
  );
}
