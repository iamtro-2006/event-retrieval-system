import { useState } from "react";
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

function TemporalSequenceBlock({
  result,
  sequenceIndex,
  selectedId,
  onSelect,
  onSubmit,
  onSimilaritySearch,
  onSurroundingImages,
}) {
  const sequence = result.matched_sequence || [];

  function makeFrameResult(frame, idx) {
    const frameId = frame.frame_id ?? Number(frame.keyframe_id ?? 0) ?? idx;
    const frameResultId = `${result.id}_q${frame.sub_query_idx ?? idx}`;

    return {
      ...result,
      ...frame,
      id: frameResultId,
      video_id: frame.video_id || result.video_id,
      video_url: frame.video_url || result.video_url,
      frame_id: frameId,
      frame_name: frame.frame_name || `${String(frameId).padStart(6, "0")}.jpg`,
      image_url: frame.image_url,
      timestamp: frame.timestamp_sec,
      similarity: frame.score,
      temporal: {
        ...result.temporal,
        start_time: frame.timestamp_sec,
        end_time: frame.timestamp_sec,
      },
      matched_sequence: [],
    };
  }

  function handleSelectFrame(frame, idx) {
    onSelect?.(makeFrameResult(frame, idx));
  }

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

  function handleSimilaritySearch(e, frame, idx) {
    e.stopPropagation();
    onSimilaritySearch?.(makeFrameResult(frame, idx));
  }

  function handleSurroundingImages(e, frame, idx) {
    e.stopPropagation();
    onSurroundingImages?.(makeFrameResult(frame, idx));
  }

  function handleSubmitFrame(e, frame, idx) {
    e.stopPropagation();
    onSubmit?.(makeFrameResult(frame, idx));
  }

  return (
    <div className="temporal-sequence-block">
      <div className="temporal-sequence-header">
        <div>
          <strong>Sequence #{sequenceIndex + 1}</strong>
          <span>
            {Number(result.temporal?.start_time ?? 0).toFixed(2)}s →{" "}
            {Number(result.temporal?.end_time ?? 0).toFixed(2)}s
          </span>
        </div>

        <button type="button" onClick={handlePlaySequence}>
          <Play size={14} fill="currentColor" />
          Play
        </button>

        <button type="button" onClick={handleSubmitSequence}>
          <Send size={13} />
          Submit
        </button>
      </div>

      <div className="temporal-frame-row">
        {sequence.map((frame, idx) => {
          const frameResult = makeFrameResult(frame, idx);
          const isSelected = frameResult.id === selectedId;

          return (
            <article
              key={frameResult.id}
              className={`temporal-frame-card ${isSelected ? "selected" : ""}`}
              onClick={() => handleSelectFrame(frame, idx)}
            >
              <div className="temporal-frame-thumb">
                <img
                  src={frameResult.image_url}
                  alt={`Q${idx + 1}`}
                  loading="lazy"
                  decoding="async"
                />

                <span className="temporal-query-badge">Q{idx + 1}</span>

                <span className="temporal-score-badge">
                  {(Number(frame.score ?? 0) * 100).toFixed(1)}%
                </span>
              </div>

              <div className="temporal-frame-footer">
                <strong>
                  {frameResult.video_id}/
                  {String(frameResult.frame_id ?? 0).padStart(6, "0")}
                </strong>
                <span>{Number(frame.timestamp_sec ?? 0).toFixed(2)}s</span>
              </div>

              <div className="temporal-frame-actions-row">
                <button
                  type="button"
                  title="Similarity search"
                  onClick={(e) => handleSimilaritySearch(e, frame, idx)}
                >
                  <Search size={12} />
                  <span>Similar</span>
                </button>

                <button
                  type="button"
                  title="Surrounding images"
                  onClick={(e) => handleSurroundingImages(e, frame, idx)}
                >
                  <Images size={12} />
                  <span>Surround</span>
                </button>

                <button
                  type="button"
                  title="Submit frame"
                  onClick={(e) => handleSubmitFrame(e, frame, idx)}
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
    </div>
  );
}

export default function GroupedResults({
  results,
  columns,
  selectedId,
  onSelect,
  onSubmit,
  onSimilaritySearch,
  onSurroundingImages,
}) {
  const groups = groupByVideoSorted(results);
  const [pageByVideo, setPageByVideo] = useState({});

  function getCurrentPage(videoId) {
    return pageByVideo[videoId] ?? 0;
  }

  function changePage(videoId, direction, totalPages) {
    const currentPage = getCurrentPage(videoId);

    const nextPage = Math.min(
      Math.max(currentPage + direction, 0),
      totalPages - 1
    );

    setPageByVideo((prev) => ({
      ...prev,
      [videoId]: nextPage,
    }));
  }

  return (
    <div className="grouped-results">
      {groups.map(({ videoId, items }) => {
        const hasTemporal = items.some(
          (item) =>
            Array.isArray(item.matched_sequence) &&
            item.matched_sequence.length > 0
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

                <div className="video-strip-content temporal-video-strip-content">
                  <div className="video-group-viewport temporal-video-viewport">
                    <div className="video-group-page temporal-page">
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
                </div>
              </div>
            </section>
          );
        }

        const pageSize = columns;
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