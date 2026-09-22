export const SEARCH_MODES = Object.freeze([
  { key: "text", searchMode: "semantic", label: "Semantic", description: "Visual meaning", shortcut: "Ctrl+1", code: "Digit1" },
  { key: "temporal", searchMode: "temporal", label: "Temporal", description: "Ordered events", shortcut: "Ctrl+2", code: "Digit2" },
  { key: "auto", searchMode: "auto", label: "Auto", description: "Detect query intent", shortcut: "Ctrl+3", code: "Digit3" },
  { key: "ocr", searchMode: "ocr", label: "OCR", description: "On-screen text", shortcut: "Ctrl+4", code: "Digit4" },
  { key: "asr", searchMode: "asr", label: "ASR", description: "Spoken content", shortcut: "Ctrl+5", code: "Digit5" },
  { key: "fusion", searchMode: "fusion", label: "Fusion", description: "Combine sources", shortcut: "Ctrl+6", code: "Digit6" },
  { key: "color", searchMode: "color", label: "Color", description: "Dominant color grid", shortcut: "Ctrl+7", code: "Digit7" },
]);

export const SEARCH_MODE_BY_CODE = Object.freeze(
  Object.fromEntries(SEARCH_MODES.map((mode) => [mode.code, mode]))
);

export const SEARCH_MODE_BY_KEY = Object.freeze(
  Object.fromEntries(SEARCH_MODES.map((mode) => [mode.key, mode]))
);

export function resolveSearchMode(modeKey) {
  return SEARCH_MODE_BY_KEY[modeKey]?.searchMode ?? "semantic";
}
