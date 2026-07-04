import { memo, useMemo } from "react";
import { buildHighlightedSnippet, renderSnippetPlainText } from "../utils/textSnippet";

/** Renders a bounded (non-runaway) text snippet with query-keyword matches
 * wrapped in <mark>. Falls back to a plain head-truncation when there is no
 * keyword match, so we still never dump raw, unbounded text into the UI. */
const HighlightedSnippet = memo(function HighlightedSnippet({
  text,
  query,
  maxLength = 160,
  className = "",
  quoted = false,
}) {
  const snippet = useMemo(
    () => buildHighlightedSnippet(text, query, { maxLength }),
    [text, query, maxLength]
  );

  if (!snippet.segments.length) return null;

  const title = renderSnippetPlainText(snippet);

  return (
    <span className={className} title={text || title}>
      {snippet.truncatedStart && <span className="snippet-ellipsis" aria-hidden="true">…</span>}
      {quoted && "\u201c"}
      {snippet.segments.map((seg, i) =>
        seg.match ? (
          <mark key={i} className="match-highlight">{seg.text}</mark>
        ) : (
          <span key={i}>{seg.text}</span>
        )
      )}
      {quoted && "\u201d"}
      {snippet.truncatedEnd && <span className="snippet-ellipsis" aria-hidden="true">…</span>}
    </span>
  );
});

export default HighlightedSnippet;
