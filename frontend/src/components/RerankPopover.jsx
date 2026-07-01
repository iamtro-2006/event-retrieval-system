import { useState, useRef, useEffect } from "react";
import { Zap, ChevronDown, Loader2 } from "lucide-react";

/**
 * RerankPopover — Nút rerank thủ công hiển thị inline ngay dưới query summary.
 *
 * Props:
 *   currentQuery   — câu query gốc (pre-fill vào VLM query field)
 *   searchMode     — "semantic" | "temporal"
 *   loading        — bool
 *   onRerank(params) — callback với { query, topCandidate, topK }
 */
export default function RerankPopover({
  currentQuery = "",
  searchMode = "semantic",
  loading = false,
  reranking = false,
  onRerank,
}) {
  const [open, setOpen] = useState(false);
  const [vlmQuery, setVlmQuery] = useState(currentQuery);
  const [topCandidate, setTopCandidate] = useState(1.0);
  const [topK, setTopK] = useState(20);

  const popoverRef = useRef(null);
  const btnRef = useRef(null);

  // Sync khi query gốc thay đổi
  useEffect(() => {
    setVlmQuery(currentQuery);
  }, [currentQuery]);

  // Đóng khi click ra ngoài
  useEffect(() => {
    if (!open) return;
    function handleClickOutside(e) {
      if (
        popoverRef.current && !popoverRef.current.contains(e.target) &&
        btnRef.current  && !btnRef.current.contains(e.target)
      ) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [open]);

  function handleSubmit(e) {
    e.preventDefault();
    const cleanQuery = vlmQuery.trim();
    if (!cleanQuery || reranking) return;
    setOpen(false);
    onRerank?.({ query: cleanQuery, topCandidate, topK });
  }

  return (
    <div className="rerank-popover-host">
      <button
        ref={btnRef}
        type="button"
        className={["rerank-manual-btn", open ? "open" : "", reranking ? "reranking" : ""].filter(Boolean).join(" ")}
        onClick={() => !reranking && setOpen((v) => !v)}
        disabled={loading || reranking}
        title="Rerank kết quả bằng VLM (InternVL)"
      >
        {reranking ? (
          <><Loader2 size={13} className="spin" /><span>Đang rerank...</span></>
        ) : (
          <><Zap size={13} /><span>Rerank VLM</span><ChevronDown size={12} className={open ? "rotated" : ""} /></>
        )}
      </button>

      {open && (
        <div ref={popoverRef} className="rerank-popover">
          <p className="rerank-popover-title">Cấu hình Rerank (InternVL)</p>

          <form onSubmit={handleSubmit}>
            {/* VLM query */}
            <label className="rerank-popover-label">
              VLM Query
              <textarea
                className="rerank-popover-input rerank-popover-textarea"
                value={vlmQuery}
                rows={2}
                placeholder="Nhập mô tả để VLM chấm điểm..."
                onChange={(e) => setVlmQuery(e.target.value)}
                autoFocus
              />
            </label>

            {/* top_candidate */}
            <label className="rerank-popover-label">
              Candidate range
              <div className="rerank-popover-row">
                <input
                  className="rerank-popover-input rerank-popover-number"
                  type="number"
                  value={topCandidate}
                  min={0.1}
                  max={200}
                  step={0.1}
                  title="1.0 = toàn bộ kết quả; 0.5 = 50% đầu; >1 = số cứng"
                  onChange={(e) => setTopCandidate(Number(e.target.value))}
                />
                <span className="rerank-popover-hint">
                  {topCandidate <= 1 ? `${Math.round(topCandidate * 100)}% kết quả` : `${topCandidate} frame đầu`}
                </span>
              </div>
            </label>

            {/* top_k output */}
            <label className="rerank-popover-label">
              Số kết quả trả về
              <input
                className="rerank-popover-input rerank-popover-number"
                type="number"
                value={topK}
                min={1}
                max={200}
                step={1}
                onChange={(e) => setTopK(Number(e.target.value))}
              />
            </label>

            <button
              type="submit"
              className="rerank-popover-submit"
              disabled={!vlmQuery.trim() || reranking}
            >
              <Zap size={13} /> Bắt đầu Rerank
            </button>
          </form>
        </div>
      )}
    </div>
  );
}
