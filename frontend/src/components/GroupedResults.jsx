import { useState } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import ResultCard from "./ResultCard";
import { groupByVideoSorted } from "../utils/groupByVideo";

export default function GroupedResults({
  results,
  columns,
  selectedId,
  onSelect,
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
        const totalPages = Math.max(Math.ceil(items.length / columns), 1);
        const currentPage = Math.min(getCurrentPage(videoId), totalPages - 1);

        const pages = Array.from({ length: totalPages }, (_, pageIndex) => {
          const start = pageIndex * columns;
          return items.slice(start, start + columns);
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
                      transform: `translateX(-${
                        currentPage * (100 / totalPages)
                      }%)`,
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