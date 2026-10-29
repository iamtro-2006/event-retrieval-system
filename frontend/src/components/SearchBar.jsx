import { Plus, Search, Mic, Zap, X, ImagePlus, Palette, ChevronDown, Check, Square, ArrowUp } from "lucide-react";
import { useEffect, useRef, useState, useCallback } from "react";
import AgentReasoningPicker from "./AgentReasoningPicker";
import "./SearchBar.css";
import { SEARCH_MODES, resolveSearchMode } from "../config/searchModes";

function SearchPicker({ kind, label, value, options, onChange, disabled = false }) {
  const [open, setOpen] = useState(false);
  const hostRef = useRef(null);

  useEffect(() => {
    if (!open) return undefined;
    const closeOutside = (event) => {
      if (!hostRef.current?.contains(event.target)) setOpen(false);
    };
    const closeOnEscape = (event) => {
      if (event.key === "Escape") {
        setOpen(false);
        hostRef.current?.querySelector(".search-picker-trigger")?.focus();
      }
    };
    document.addEventListener("pointerdown", closeOutside);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("pointerdown", closeOutside);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [open]);

  const selected = options.find((option) => option.value === value);
  return (
    <div className={`search-picker search-picker--${kind}`} ref={hostRef}>
      <button
        type="button"
        className="search-select search-picker-trigger"
        aria-label={`${label}: ${selected?.label ?? value}`}
        aria-haspopup="listbox"
        aria-expanded={open}
        disabled={disabled}
        onClick={() => setOpen((current) => !current)}
      >
        <span>{selected?.label ?? value}</span>
        <ChevronDown size={12} aria-hidden="true" />
      </button>
      {open && (
        <div className="search-picker-menu" role="listbox" aria-label={label}>
          <div className="search-picker-heading">{label}</div>
          {options.map((option) => (
            <button
              key={option.value}
              type="button"
              className="search-picker-option"
              role="option"
              aria-selected={option.value === value}
              title={option.label}
              onClick={() => { onChange(option.value); setOpen(false); }}
            >
              <span className="search-picker-option-label">{option.label}</span>
              {option.shortcut && <span className="search-picker-shortcut">{option.shortcut}</span>}
              {option.value === value && <Check size={13} aria-hidden="true" />}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

export default function SearchBar({
  query = "",
  model,
  mode,
  loading,
  disabled = false,
  durationLimit = -1,
  reasoningEnabled = false,
  agentAvailable = false,
  agentConfig,
  contextFrameCount = 0,
  colorSelectionCount = 0,
  onClearContext,
  onStop,
  availableModels = [],
  fusionConfig,
  queryClauses = [""],
  queryConnectors = [],
  clauseImages = [[]],
  onModelChange,
  onModeChange,
  onDurationLimitChange,
  onReasoningToggle,
  onQueryChange,
  onInlineClauseChange,
  onInsertClauseImages,
  onOpenColorSearch,
  onRemoveClauseImage,
  onSearch,
}) {
  const [recording, setRecording] = useState(false);
  const [addMenuOpen, setAddMenuOpen] = useState(false);
  // The backend is the only source of truth for loaded indexes. Never show
  // guessed/fallback model keys because selecting one produces an invalid
  // model_key for the active dataset.
  const modelOptions = availableModels;

  const textareaRef = useRef(null);
  const recognitionRef = useRef(null);
  const fileInputRef = useRef(null);
  const committedTranscriptRef = useRef("");
  const activeInlineClauseRef = useRef(0);
  const inlineInputRefs = useRef([]);

  const isTemporal = mode === "temporal";
  const isColor = mode === "color";
  const activeModeLabel = SEARCH_MODES.find((item) => item.key === mode)?.label ?? "Semantic";
  const searchPlaceholder = reasoningEnabled ? "Ask the assistant or describe what to find…" : `${activeModeLabel} Search`;
  const hasQueryImages = clauseImages.some((items) => items.length > 0);
  const resizeTextarea = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    requestAnimationFrame(() => {
      el.style.height = "auto";
      el.style.height = `${Math.min(el.scrollHeight, 92)}px`;
    });
  }, []);

  useEffect(() => {
    resizeTextarea();
  }, [query, resizeTextarea]);

  useEffect(() => {
    return () => stopBrowserSpeech();
  }, []);

  function runSearch(nextQuery) {
    if (isColor && !reasoningEnabled && !colorSelectionCount) { onOpenColorSearch?.(); return; }
    const cleanQuery = String(nextQuery || "").trim() || (isColor && !reasoningEnabled ? "Search selected colors" : "");
    if ((!cleanQuery && !hasQueryImages) || loading || disabled) return;
    const searchMode = reasoningEnabled ? "auto" : resolveSearchMode(mode);
    onSearch({
      query: cleanQuery,
      searchMode,
      durationLimit: searchMode === "temporal" ? Number(durationLimit) : -1,
      fusionConfig: searchMode === "fusion" ? fusionConfig : null,
      reasoning: reasoningEnabled,
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
    if (e.nativeEvent?.isComposing) return;
    if (reasoningEnabled) {
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSubmit(); }
      return;
    }
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
        "expanded",
        recording ? "voice-recording-active" : "",
      ].filter(Boolean).join(" ")}
      onSubmit={handleSubmit}
    >
      <div className="search-inner">
        {isColor && colorSelectionCount > 0 && <div className="composer-context-chip">{colorSelectionCount}/25 color cells
          <button type="button" onClick={onOpenColorSearch}>Edit canvas</button></div>}
        {contextFrameCount > 0 && <div className="composer-context-chip">{contextFrameCount} frames attached
          <button type="button" onClick={onClearContext} aria-label="Remove frame context"><X size={12} /></button></div>}
        {!hasQueryImages ? <textarea
          ref={textareaRef}
          className="search-chat-textarea"
          aria-label="Main search"
          value={query}
          rows={1}
          placeholder={
            recording
              ? "Listening..."
              : searchPlaceholder
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
                  placeholder={clauseIndex === 0 && !(clauseImages[0] || []).length ? searchPlaceholder : ""}
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
            <div className="composer-options">
              <button
                type="button"
                className={["rerank-tag", reasoningEnabled ? "rerank-tag--active" : ""].filter(Boolean).join(" ")}
                aria-label={reasoningEnabled ? "Disable Reasoning" : "Enable Reasoning"}
                title={
                  reasoningEnabled
                    ? "Assistant uses tools to search and inspect frames"
                    : "Enable assistant chat and tool use"
                }
                onClick={() => onReasoningToggle?.(!reasoningEnabled)}
                aria-pressed={reasoningEnabled}
                disabled={disabled || !agentAvailable || loading}
              >
                <Zap size={13} />
                <span>Reasoning</span>
              </button>

              {!reasoningEnabled && modelOptions.length > 0 && (
                <SearchPicker kind="model" label="Model" value={model}
                  options={modelOptions.map((modelKey) => ({ value: modelKey, label: modelKey }))}
                  onChange={onModelChange} />
              )}

              {!reasoningEnabled && <SearchPicker kind="mode" label="Search mode" value={mode}
                options={SEARCH_MODES.map(({ key, label, shortcut }) => ({ value: key, label, shortcut }))}
                onChange={onModeChange} />}

              {!reasoningEnabled && isTemporal && (
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

              {reasoningEnabled && agentConfig && <AgentReasoningPicker config={agentConfig} disabled={loading} />}


            </div>
            <div className="composer-actions">
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
                type={loading ? "button" : "submit"}
                className={`search-submit${loading ? " is-stop" : ""}`}
                onClick={loading ? onStop : undefined}
                disabled={!loading && (disabled || ((reasoningEnabled || !isColor) && !query.trim() && !hasQueryImages))}
                aria-label={loading ? "Stop current run" : reasoningEnabled ? "Send to assistant" : "Search"}
                title={loading ? "Stop current run" : reasoningEnabled ? "Send message" : "Search"}
              >
                {loading ? <Square size={15} fill="currentColor" /> : reasoningEnabled ? <ArrowUp size={19} /> : <Search size={18} />}
              </button>
            </div>
          </div>
        </div>
      </div>

    </form>
  );
}
