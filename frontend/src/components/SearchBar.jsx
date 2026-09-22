import { Plus, Search, Mic, Zap, Settings2, X, ImagePlus, Palette } from "lucide-react";
import { useEffect, useRef, useState, useCallback } from "react";
import FusionSettingsModal from "./FusionSettingsModal";
import { useVideoFilter } from "./VideoFilter";
import { SEARCH_MODES, resolveSearchMode } from "../config/searchModes";

const FALLBACK_MODELS = ["siglip2-so400m", "vitH-378-quickgelu"];

export default function SearchBar({
  query = "",
  theme = "dark",
  model,
  mode,
  loading,
  disabled = false,
  durationLimit = -1,
  reasoningEnabled = false,
  availableModels = [],
  semanticModelRoles,
  fusionConfig,
  queryClauses = [""],
  queryConnectors = [],
  clauseImages = [[]],
  onModelChange,
  onModeChange,
  onDurationLimitChange,
  onReasoningToggle,
  onFusionConfigChange,
  onQueryChange,
  onInlineClauseChange,
  onInsertClauseImages,
  onOpenColorSearch,
  onRemoveClauseImage,
  expandedByDefault = false,
  onSearch,
}) {
  const [recording, setRecording] = useState(false);
  const videoFilter = useVideoFilter();
  const [fusionModalOpen, setFusionModalOpen] = useState(false);
  const [addMenuOpen, setAddMenuOpen] = useState(false);
  // The home composer starts expanded; the bottom composer starts compact
  // and expands only when the query wraps past the normal height.
  const [isExpanded, setIsExpanded] = useState(expandedByDefault);
  const expandedRef = useRef(expandedByDefault);

  const modelOptions = availableModels.length > 0 ? availableModels : FALLBACK_MODELS;

  const textareaRef = useRef(null);
  const recognitionRef = useRef(null);
  const fileInputRef = useRef(null);
  const committedTranscriptRef = useRef("");
  const activeInlineClauseRef = useRef(0);
  const inlineInputRefs = useRef([]);

  const isTemporal = mode === "temporal";
  const isOcr = mode === "ocr";
  const isAsr = mode === "asr";
  const isFusion = mode === "fusion";
  const isColor = mode === "color";
  const hasQueryImages = clauseImages.some((items) => items.length > 0);
  const resizeTextarea = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    requestAnimationFrame(() => {
      // Measure against the collapsed 40px textarea rather than using a
      // character-count heuristic. A short query can wrap on a narrow
      // viewport, while a long query may still fit on a wide one.
      const probe = el.cloneNode(true);
      probe.style.position = "absolute";
      probe.style.visibility = "hidden";
      probe.style.pointerEvents = "none";
      probe.style.left = "-100000px";
      probe.style.top = "0";
      probe.style.width = `${Math.max(1, el.clientWidth)}px`;
      probe.style.height = "40px";
      probe.style.minHeight = "40px";
      probe.style.maxHeight = "40px";
      probe.style.overflow = "hidden";
      probe.style.whiteSpace = "pre-wrap";
      probe.value = el.value;
      document.body.appendChild(probe);
      const shouldExpand = probe.scrollHeight > 40 || query.includes("\n");
      probe.remove();

      // Do not measure the expanded layout and immediately collapse it again:
      // the expanded textarea is wider, so the same text can fit on one line
      // after expansion and create an expand/collapse feedback loop.
      if (!expandedRef.current && shouldExpand) {
        expandedRef.current = true;
        setIsExpanded(true);
      } else if (expandedRef.current && !query.trim()) {
        expandedRef.current = false;
        setIsExpanded(false);
      }

      el.style.height = "auto";
      el.style.height = `${Math.min(el.scrollHeight, 220)}px`;
    });
  }, [query]);

  useEffect(() => {
    resizeTextarea();
  }, [query, resizeTextarea]);

  useEffect(() => {
    return () => stopBrowserSpeech();
  }, []);

  function runSearch(nextQuery) {
    if (isColor) { onOpenColorSearch?.(); return; }
    const cleanQuery = String(nextQuery || "").trim();
    if ((!cleanQuery && !hasQueryImages) || loading || disabled) return;
    const searchMode = resolveSearchMode(mode);
    onSearch({
      query: cleanQuery,
      searchMode,
      durationLimit: searchMode === "temporal" ? Number(durationLimit) : -1,
      fusionConfig: searchMode === "fusion" ? fusionConfig : null,
      reasoning: false,
    });
  }

  function activeInsertionPoint() {
    if (hasQueryImages) {
      const clauseIndex = Math.max(0, Math.min(activeInlineClauseRef.current, queryClauses.length - 1));
      const input = inlineInputRefs.current[clauseIndex];
      return { clauseIndex, offset: input?.selectionStart ?? queryClauses[clauseIndex]?.length ?? 0 };
    }
    const caret = textareaRef.current?.selectionStart ?? query.length;
    const beforeCaret = query.slice(0, caret);
    const separatorPattern = /\b(?:AND|THEN)\b/g;
    const separators = beforeCaret.match(separatorPattern);
    const clauseIndex = Math.min(separators?.length ?? 0, Math.max(0, queryClauses.length - 1));
    let prefixLength = 0;
    for (let index = 0; index < clauseIndex; index += 1) {
      prefixLength += String(queryClauses[index] || "").length;
      prefixLength += ` ${queryConnectors[index] || "AND"} `.length;
    }
    return {
      clauseIndex,
      offset: Math.max(0, Math.min(caret - prefixLength, String(queryClauses[clauseIndex] || "").length)),
    };
  }

  function insertFiles(files, point = activeInsertionPoint()) {
    if (!files?.length) return;
    onInsertClauseImages?.(point.clauseIndex, point.offset, files);
    activeInlineClauseRef.current = point.clauseIndex + 1;
    requestAnimationFrame(() => requestAnimationFrame(() => {
      const input = inlineInputRefs.current[point.clauseIndex + 1];
      input?.focus();
      input?.setSelectionRange(0, 0);
    }));
  }

  function imageQueryPosition(clauseIndex, imageIndex) {
    let position = 1;
    for (let index = 0; index < clauseIndex; index += 1) {
      position += (queryClauses[index]?.trim() ? 1 : 0) + (clauseImages[index]?.length ?? 0);
    }
    return position + (queryClauses[clauseIndex]?.trim() ? 1 : 0) + imageIndex;
  }

  function updateInlineClause(clauseIndex, value) {
    const nextClauses = queryClauses.map((clause, index) => index === clauseIndex ? value : clause);
    let nextQuery = String(nextClauses[0] || "");
    for (let index = 1; index < nextClauses.length; index += 1) {
      nextQuery += ` ${queryConnectors[index - 1] || "AND"} ${nextClauses[index] || ""}`;
    }
    committedTranscriptRef.current = nextQuery;
    onInlineClauseChange?.(clauseIndex, value);
    const connectorCount = (value.match(/\b(?:AND|THEN)\b/g) || []).length;
    if (connectorCount) {
      const nextIndex = clauseIndex + connectorCount;
      activeInlineClauseRef.current = nextIndex;
      requestAnimationFrame(() => requestAnimationFrame(() => {
        const input = inlineInputRefs.current[nextIndex];
        if (!input) return;
        input.focus();
        input.setSelectionRange(input.value.length, input.value.length);
      }));
    }
  }

  function handlePaste(event) {
    const images = Array.from(event.clipboardData?.files ?? []).filter((file) => file.type.startsWith("image/"));
    if (!images.length) return;
    event.preventDefault();
    insertFiles(images, activeInsertionPoint());
  }

  function handleSubmit(e) {
    e?.preventDefault();
    runSearch(query);
  }

  function handleKeyDown(e) {
    const noCommandModifier = !e.ctrlKey && !e.metaKey && !e.altKey;
    const addSemantic = e.shiftKey && noCommandModifier && (
      e.key === "+" || e.key === ":" || e.code === "Equal"
    );
    const addTemporal = e.shiftKey && noCommandModifier && (
      e.key === "|" || e.code === "Backslash"
    );
    const undoClause = e.shiftKey && noCommandModifier && (
      e.key === "_" || e.key === "-" || e.code === "Minus"
    );

    if (addSemantic || addTemporal) {
      e.preventDefault();
      const connector = addSemantic ? "AND" : "THEN";
      const base = query.trimEnd();
      activeInlineClauseRef.current = queryClauses.length;
      onQueryChange?.(`${base}${base ? " " : ""}${connector} `);
      requestAnimationFrame(() => {
        const inlineInput = inlineInputRefs.current[queryClauses.length];
        if (inlineInput) {
          inlineInput.focus();
          return;
        }
        const end = textareaRef.current?.value.length ?? 0;
        textareaRef.current?.setSelectionRange(end, end);
        textareaRef.current?.focus();
      });
      return;
    }
    if (undoClause) {
      const matches = [...query.matchAll(/\s+\b(?:AND|THEN)\b\s*/g)];
      if (matches.length) {
        e.preventDefault();
        onQueryChange?.(query.slice(0, matches.at(-1).index).trimEnd());
      }
      return;
    }
    if (e.key === "Enter" && e.shiftKey) return;
    if (e.key === "Enter") {
      e.preventDefault();
      handleSubmit();
    }
  }

  function startBrowserSpeech() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      alert("This browser does not support speech recognition. Use Chrome or Edge.");
      return;
    }
    committedTranscriptRef.current = query.trim();
    const recognition = new SpeechRecognition();
    recognition.lang = "vi-VN";
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;
    recognition.onstart = () => setRecording(true);
    recognition.onresult = (event) => {
      let finalText = "";
      let interimText = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript.trim();
        if (!transcript) continue;
        if (event.results[i].isFinal) finalText += ` ${transcript}`;
        else interimText += ` ${transcript}`;
      }
      if (finalText.trim()) {
        committedTranscriptRef.current = `${committedTranscriptRef.current} ${finalText}`
          .replace(/\s+/g, " ").trim();
      }
      const nextText = `${committedTranscriptRef.current} ${interimText}`.replace(/\s+/g, " ").trim();
      onQueryChange?.(nextText);
    };
    recognition.onerror = (event) => {
      if (event.error === "network")
        alert("Speech recognition encountered a network error. Check your internet connection, VPN, and firewall.");
      if (event.error === "not-allowed")
        alert("Microphone permission has not been granted.");
    };
    recognition.onend = () => { setRecording(false); recognitionRef.current = null; };
    recognitionRef.current = recognition;
    recognition.start();
  }

  function stopBrowserSpeech() {
    recognitionRef.current?.stop();
    recognitionRef.current = null;
    setRecording(false);
  }

  function handleMicClick() {
    if (recording) { stopBrowserSpeech(); return; }
    startBrowserSpeech();
  }

  return (
    <form
      className={[
        "search-wrapper",
        loading ? "searching-active" : "",
        isExpanded || hasQueryImages ? "expanded" : "",
        recording ? "voice-recording-active" : "",
      ].filter(Boolean).join(" ")}
      onSubmit={handleSubmit}
    >
      <div className="search-inner">
        {!hasQueryImages ? <textarea
          ref={textareaRef}
          className="search-chat-textarea"
          value={query}
          rows={1}
          placeholder={
            recording
              ? "Listening... Speak to enter a query."
              : isTemporal
                ? "Example: person opens box THEN reads label..."
                : isOcr
                  ? "Enter on-screen text, such as signs or subtitles..."
                  : isAsr
                    ? "Enter the spoken content to find..."
                    : isFusion
                      ? "Enter a query. Results will use the saved Fusion configuration..."
                      : "Enter a retrieval query..."
          }
          onChange={(e) => {
            committedTranscriptRef.current = e.target.value;
            onQueryChange?.(e.target.value);
          }}
          onKeyDown={handleKeyDown}
          onPaste={handlePaste}
        /> : (
          <div className="inline-query-composer" aria-label="Ordered text and image query">
            {queryClauses.map((clause, clauseIndex) => (
              <div className="inline-query-clause" key={`clause-${clauseIndex}`}>
                {clauseIndex > 0 && !(
                  (queryConnectors[clauseIndex - 1] || "AND") === "AND" &&
                  (clauseImages[clauseIndex - 1] || []).length > 0
                ) && (
                  <span className={`inline-query-connector is-${(queryConnectors[clauseIndex - 1] || "AND").toLowerCase()}`}>
                    {queryConnectors[clauseIndex - 1] || "AND"}
                  </span>
                )}
                <input
                  ref={(element) => { inlineInputRefs.current[clauseIndex] = element; }}
                  className="inline-query-text-input"
                  style={{ width: `${Math.max(1, Math.min(42, clause.length + 1))}ch` }}
                  value={clause}
                  placeholder={clauseIndex === 0 && !(clauseImages[0] || []).length ? "Enter a query..." : ""}
                  onFocus={() => { activeInlineClauseRef.current = clauseIndex; }}
                  onChange={(event) => updateInlineClause(clauseIndex, event.target.value)}
                  onKeyDown={handleKeyDown}
                  onPaste={(event) => {
                    const images = Array.from(event.clipboardData?.files ?? []).filter((file) => file.type.startsWith("image/"));
                    if (!images.length) return;
                    event.preventDefault();
                    insertFiles(images, {
                      clauseIndex,
                      offset: event.currentTarget.selectionStart ?? clause.length,
                    });
                  }}
                />
                {(clauseImages[clauseIndex] ?? []).map((item, imageIndex) => (
                  <div className="query-image-chip" key={item.id} title={`Query ${imageQueryPosition(clauseIndex, imageIndex)}: ${item.name}`}>
                    <img src={item.url} alt={item.name} />
                    <span>Q{String(imageQueryPosition(clauseIndex, imageIndex)).padStart(2, "0")}</span>
                    <button type="button" onClick={() => onRemoveClauseImage?.(clauseIndex, item.id)} aria-label={`Remove ${item.name}`}>
                      <X size={11} />
                    </button>
                  </div>
                ))}
              </div>
            ))}
          </div>
        )}

        <div className="search-chat-footer">
          <input ref={fileInputRef} className="query-image-input" type="file" accept="image/*" multiple

            onChange={(event) => { insertFiles(event.target.files); event.target.value = ""; }} />
          <div className="search-add-menu-host">
          <button type="button" className="search-icon-button" aria-label="Add image or color canvas"
            aria-expanded={addMenuOpen} title="Add query input" onClick={() => setAddMenuOpen((value) => !value)}>
            <Plus size={20} />
          </button>
          {addMenuOpen && <div className="search-add-menu" role="menu">
            <button type="button" role="menuitem" onClick={() => { setAddMenuOpen(false); fileInputRef.current?.click(); }}>
              <ImagePlus size={16} /><span>Image query</span>
            </button>
            <button type="button" role="menuitem" onClick={() => { setAddMenuOpen(false); onOpenColorSearch?.(); }}>
              <Palette size={16} /><span>Colour canvas</span>
            </button>
          </div>}
          </div>

          <div className="search-chat-controls">
            {!isFusion && (
              <select className="search-select" value={model} onChange={(e) => onModelChange(e.target.value)}>
                {modelOptions.map((modelKey) => (
                  <option key={modelKey} value={modelKey}>{modelKey}</option>
                ))}
              </select>
            )}

            <select className="search-select" value={mode} onChange={(e) => onModeChange(e.target.value)}>
              {SEARCH_MODES.map(({ key, label, shortcut }) => <option key={key} value={key}>{label} ({shortcut})</option>)}
            </select>

            {isTemporal && (
              <input
                className="search-duration-input"
                type="number"
                value={durationLimit}
                min={-1}
                step={1}
                title="-1 searches the entire video; values above 0 limit the duration in seconds"
                onChange={(e) => onDurationLimitChange?.(Number(e.target.value))}
              />
            )}

            {/* ── LLM reasoning toggle ─────────────────────────────────── */}
            {videoFilter && <button type="button"
              className={`rerank-tag ${videoFilter.enabled ? "rerank-tag--active" : ""}`}
              aria-pressed={videoFilter.enabled}
              title="Limit the next search to cached videos. An empty cache searches all videos."
              onClick={() => videoFilter.setEnabled(!videoFilter.enabled)}>
              Filter {videoFilter.ids.length > 0 && `(${videoFilter.ids.length})`}
            </button>}
            {(
              <button
                type="button"
                className={["rerank-tag", reasoningEnabled ? "rerank-tag--active" : ""].filter(Boolean).join(" ")}
                aria-label={reasoningEnabled ? "Disable Reasoning" : "Enable Reasoning"}
                title={
                  reasoningEnabled
                    ? "Reasoning preview is enabled but is not connected to the search pipeline"
                    : "Reasoning preview is currently display-only"
                }
                onClick={() => onReasoningToggle?.(!reasoningEnabled)}
                disabled={disabled}
              >
                <Zap size={13} />
                <span>Reasoning</span>
              </button>
            )}

            {isFusion && (
              <button
                type="button"
                className={["search-icon-button", "fusion-settings-btn", fusionConfig?.hasConfig ? "has-config" : ""].filter(Boolean).join(" ")}
                aria-label="Configure Fusion search"
                title="Configure models, methods, and weights for Fusion search"
                onClick={() => setFusionModalOpen(true)}
                disabled={disabled}
              >
                <Settings2 size={18} />
              </button>
            )}

            <button
              type="button"
              className={["search-icon-button", "mic-button", recording ? "recording" : ""].filter(Boolean).join(" ")}
              aria-label={recording ? "Stop voice search" : "Voice search"}
              onClick={handleMicClick}
              disabled={disabled}
            >
              <Mic size={18} />
              {recording && <span className="mic-live-dot" />}
            </button>

            <button
              type="submit"
              className="search-submit"
              disabled={loading || disabled || (!isColor && !query.trim() && !hasQueryImages)}
              aria-label="Search"
            >
              <Search size={18} />
            </button>
          </div>
        </div>
      </div>

      <FusionSettingsModal
        key={fusionModalOpen ? "fusion-open" : "fusion-closed"}
        open={fusionModalOpen}
        theme={theme}
        models={modelOptions}
        modelRoles={semanticModelRoles}
        value={fusionConfig}
        onClose={() => setFusionModalOpen(false)}
        onSave={(next) => onFusionConfigChange?.({ ...next, hasConfig: true })}
      />
    </form>
  );
}
