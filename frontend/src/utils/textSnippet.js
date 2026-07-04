// Builds a length-limited snippet from a long OCR/ASR text, keeping the
// window centered on the query keyword matches (instead of always cutting
// from the start) and returning highlight-ready segments so matched keywords
// can be rendered with <mark>.

const STOPWORDS = new Set([
  "va", "la", "cua", "va", "the", "and", "or", "a", "an", "of", "in", "on",
  "to", "is", "are", "at", "for", "with",
]);

function stripDiacritics(str) {
  return str
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/đ/g, "d")
    .replace(/Đ/g, "D");
}

function normalize(str) {
  return stripDiacritics(str).toLowerCase();
}

/** Extract meaningful search terms from a query string (longest first, so
 * multi-word phrases are matched before their individual words). */
export function getQueryTerms(query) {
  if (!query || typeof query !== "string") return [];
  const raw = query
    .split(/[\s,.;:!?"'()[\]{}]+/)
    .map((t) => t.trim())
    .filter((t) => t.length >= 2 && !STOPWORDS.has(normalize(t)));

  const terms = new Set(raw);
  const full = query.trim();
  if (full.length >= 2) terms.add(full);

  return Array.from(terms).sort((a, b) => b.length - a.length);
}

/** Find non-overlapping match ranges for the given terms inside `text`.
 * Matching is diacritic- and case-insensitive but ranges refer to positions
 * in the original `text` string. */
function findMatchRanges(text, terms) {
  if (!text || !terms.length) return [];
  const normText = normalize(text);
  const ranges = [];

  for (const term of terms) {
    const normTerm = normalize(term);
    if (!normTerm) continue;
    let from = 0;
    while (from <= normText.length) {
      const idx = normText.indexOf(normTerm, from);
      if (idx === -1) break;
      ranges.push([idx, idx + normTerm.length]);
      from = idx + normTerm.length;
    }
  }

  if (!ranges.length) return ranges;

  ranges.sort((a, b) => a[0] - b[0]);
  const merged = [ranges[0]];
  for (const [start, end] of ranges.slice(1)) {
    const last = merged[merged.length - 1];
    if (start <= last[1]) {
      last[1] = Math.max(last[1], end);
    } else {
      merged.push([start, end]);
    }
  }
  return merged;
}

/**
 * Build a bounded snippet of `text` around the query matches.
 *
 * Returns:
 *   {
 *     segments: [{ text, match }],  // ready to render, already cropped
 *     truncatedStart: boolean,      // content was cut before the window
 *     truncatedEnd: boolean,        // content was cut after the window
 *     hasMatch: boolean,
 *   }
 */
export function buildHighlightedSnippet(text, query, { maxLength = 160 } = {}) {
  const safeText = typeof text === "string" ? text : "";
  if (!safeText) {
    return { segments: [], truncatedStart: false, truncatedEnd: false, hasMatch: false };
  }

  const terms = getQueryTerms(query);
  const ranges = findMatchRanges(safeText, terms);

  let windowStart = 0;
  let windowEnd = Math.min(safeText.length, maxLength);

  if (ranges.length) {
    // Prefer a window that captures as many matches as possible; anchor on
    // the first match and grow while it still fits the budget.
    const firstStart = ranges[0][0];
    let lastEndWithinBudget = ranges[0][1];
    for (const [, end] of ranges) {
      if (end - firstStart <= maxLength) lastEndWithinBudget = end;
      else break;
    }

    const matchSpan = lastEndWithinBudget - firstStart;
    const slack = Math.max(0, maxLength - matchSpan);
    const before = Math.floor(slack / 2);

    windowStart = Math.max(0, firstStart - before);
    windowEnd = Math.min(safeText.length, windowStart + maxLength);
    // If we hit the end early, pull the start back to still use the full budget.
    windowStart = Math.max(0, windowEnd - maxLength);
  }

  const truncatedStart = windowStart > 0;
  const truncatedEnd = windowEnd < safeText.length;
  const windowText = safeText.slice(windowStart, windowEnd);

  // Recompute ranges relative to the cropped window for rendering.
  const localRanges = ranges
    .map(([s, e]) => [Math.max(s, windowStart) - windowStart, Math.min(e, windowEnd) - windowStart])
    .filter(([s, e]) => e > s);

  const segments = [];
  let cursor = 0;
  for (const [s, e] of localRanges) {
    if (s > cursor) segments.push({ text: windowText.slice(cursor, s), match: false });
    segments.push({ text: windowText.slice(s, e), match: true });
    cursor = e;
  }
  if (cursor < windowText.length) segments.push({ text: windowText.slice(cursor), match: false });
  if (!segments.length) segments.push({ text: windowText, match: false });

  return { segments, truncatedStart, truncatedEnd, hasMatch: localRanges.length > 0 };
}

export function renderSnippetPlainText({ segments, truncatedStart, truncatedEnd }) {
  const body = segments.map((s) => s.text).join("");
  return `${truncatedStart ? "…" : ""}${body}${truncatedEnd ? "…" : ""}`;
}
