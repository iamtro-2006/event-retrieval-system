import { Plus, Search, Mic, Zap, Settings2, X } from "lucide-react";
import { useEffect, useRef, useState, useCallback } from "react";
import FusionSettingsModal from "./FusionSettingsModal";

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
  fusionConfig,
  useSplit = true,
  queryClauses = [""],
  clauseImages = [[]],
  onModelChange,
  onModeChange,
  onDurationLimitChange,
  onReasoningToggle,
  onFusionConfigChange,
  onQueryChange,
  onAddClauseImages,
  onRemoveClauseImage,
  expandedByDefault = false,
  onSearch,
}) {
  const [recording, setRecording] = useState(false);
  const [fusionModalOpen, setFusionModalOpen] = useState(false);
  // The home composer starts expanded; the bottom composer starts compact
  // and expands only when the query wraps past the normal height.
  const [isExpanded, setIsExpanded] = useState(expandedByDefault);
  const expandedRef = useRef(expandedByDefault);

  const modelOptions = availableModels.length > 0 ? availableModels : FALLBACK_MODELS;

  const textareaRef = useRef(null);
  const recognitionRef = useRef(null);
  const fileInputRef = useRef(null);
  const committedTranscriptRef = useRef("");

  const isTemporal = mode === "temporal";
  const isOcr = mode === "ocr";
  const isAsr = mode === "asr";
  const isFusion = mode === "fusion";
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

  function resolveSearchMode() {
    if (mode === "temporal" || mode === "auto" || mode === "ocr" || mode === "asr" || mode === "fusion") return mode;
    return "semantic";
  }

  function runSearch(nextQuery) {
    const cleanQuery = String(nextQuery || "").trim();
    if ((!cleanQuery && !hasQueryImages) || loading || disabled) return;
    const searchMode = resolveSearchMode();
    onSearch({
      query: cleanQuery,
      searchMode,
      durationLimit: searchMode === "temporal" ? Number(durationLimit) : -1,
      fusionConfig: searchMode === "fusion" ? fusionConfig : null,
      reasoning: searchMode === "temporal" || searchMode === "fusion" ? reasoningEnabled : false,
    });
  }

  function activeClauseIndex() {
    const caret = textareaRef.current?.selectionStart ?? query.length;
    const beforeCaret = query.slice(0, caret);
    const separatorPattern = mode === "temporal" || mode === "auto" ? /[;\n]/g : /,/g;
    const separators = beforeCaret.match(separatorPattern);
    const clauseIndex = Math.min(separators?.length ?? 0, Math.max(0, queryClauses.length - 1));
    const segmentStart = (mode === "temporal" || mode === "auto"
      ? Math.max(beforeCaret.lastIndexOf(";"), beforeCaret.lastIndexOf("\n"))
      : beforeCaret.lastIndexOf(",")) + 1;
    // Pasting immediately after a delimiter means “insert before the next
    // text clause”. Store it after the previous clause so the flattened
    // query order is text → image → next text.
    if (!beforeCaret.slice(segmentStart).trim() && clauseIndex > 0) return clauseIndex - 1;
    return clauseIndex;
  }

  function attachFiles(files) {
    onAddClauseImages?.(activeClauseIndex(), files);
  }

  function imageQueryPosition(clauseIndex, imageIndex) {
    if (!useSplit) return 1;
    let position = 1;
    for (let index = 0; index < clauseIndex; index += 1) {
      position += (queryClauses[index]?.trim() ? 1 : 0) + (clauseImages[index]?.length ?? 0);
    }
    return position + (queryClauses[clauseIndex]?.trim() ? 1 : 0) + imageIndex;
  }

  const sequenceTokens = [];
  queryClauses.forEach((clause, clauseIndex) => {
    const text = String(clause || "").trim();
    if (text) sequenceTokens.push({ type: "text", text, key: `text-${clauseIndex}` });
    (clauseImages[clauseIndex] ?? []).forEach((item, imageIndex) => {
      sequenceTokens.push({ type: "image", item, clauseIndex, imageIndex, key: item.id });
    });
  });
  const sequenceConnector = useSplit
    ? (isTemporal || mode === "auto" ? "→" : ",")
    : "+";

  function handlePaste(event) {
    const images = Array.from(event.clipboardData?.files ?? []).filter((file) => file.type.startsWith("image/"));
    if (!images.length) return;
    event.preventDefault();
    attachFiles(images);
  }

  function handleSubmit(e) {
    e?.preventDefault();
    runSearch(query);
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && e.shiftKey) return;
    if (e.key === "Enter") {
      e.preventDefault();
      handleSubmit();
    }
  }

  function startBrowserSpeech() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      alert("Trình duyệt chưa hỗ trợ Speech Recognition. Hãy dùng Chrome hoặc Edge.");
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
        alert("Speech Recognition bị lỗi network. Hãy kiểm tra internet/VPN/firewall.");
      if (event.error === "not-allowed")
        alert("Trình duyệt chưa được cấp quyền microphone.");
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
        <textarea
          ref={textareaRef}
          className="search-chat-textarea"
          value={query}
          rows={1}
          placeholder={
            recording
              ? "Đang nghe, nói để nhập truy vấn..."
              : isTemporal
                ? "Ví dụ: person opens box; reads label..."
                : isOcr
                  ? "Nhập chữ xuất hiện trên màn hình (biển hiệu, phụ đề...)..."
                  : isAsr
                    ? "Nhập nội dung lời thoại/giọng nói cần tìm..."
                    : isFusion
                      ? "Nhập truy vấn — kết quả sẽ được fusion theo cấu hình đã lưu..."
                      : "Nhập truy vấn retrieval..."
          }
          onChange={(e) => {
            committedTranscriptRef.current = e.target.value;
            onQueryChange?.(e.target.value);
          }}
          onKeyDown={handleKeyDown}
          onPaste={handlePaste}
        />

        {hasQueryImages && (
          <div className="query-sequence-strip" aria-label="Ordered multimodal query">
            {sequenceTokens.map((token, tokenIndex) => (
              <div className="query-sequence-item" key={token.key}>
                {tokenIndex > 0 && <span className="query-sequence-connector">{sequenceConnector}</span>}
                {token.type === "text" ? (
                  <span className="query-text-token" title={token.text}>{token.text}</span>
                ) : (
                  <div className="query-image-chip" title={`Query ${imageQueryPosition(token.clauseIndex, token.imageIndex)}: ${token.item.name}`}>
                    <img src={token.item.url} alt={token.item.name} />
                    <span>Q{String(imageQueryPosition(token.clauseIndex, token.imageIndex)).padStart(2, "0")}</span>
                    <button type="button" onClick={() => onRemoveClauseImage?.(token.clauseIndex, token.item.id)} aria-label={`Remove ${token.item.name}`}>
                      <X size={11} />
                    </button>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        <div className="search-chat-footer">
          <input ref={fileInputRef} className="query-image-input" type="file" accept="image/*" multiple
            onChange={(event) => { attachFiles(event.target.files); event.target.value = ""; }} />
          <button type="button" className="search-icon-button" aria-label="Attach query image"
            title="Upload image query" onClick={() => fileInputRef.current?.click()}>
            <Plus size={20} />
          </button>

          <div className="search-chat-controls">
            {!isFusion && (
              <select className="search-select" value={model} onChange={(e) => onModelChange(e.target.value)}>
                {modelOptions.map((modelKey) => (
                  <option key={modelKey} value={modelKey}>{modelKey}</option>
                ))}
              </select>
            )}

            <select className="search-select" value={mode} onChange={(e) => onModeChange(e.target.value)}>
              <option value="text">Semantic</option>
              <option value="temporal">Temporal</option>
              <option value="auto">Auto</option>
              <option value="ocr">OCR (on-screen text)</option>
              <option value="asr">ASR (speech)</option>
              <option value="fusion">Fusion</option>
            </select>

            {isTemporal && (
              <input
                className="search-duration-input"
                type="number"
                value={durationLimit}
                min={-1}
                step={1}
                title="-1 = quét toàn video; >0 = giới hạn số giây"
                onChange={(e) => onDurationLimitChange?.(Number(e.target.value))}
              />
            )}

            {/* ── LLM reasoning toggle ─────────────────────────────────── */}
            {(isTemporal || isFusion) && (
              <button
                type="button"
                className={["rerank-tag", reasoningEnabled ? "rerank-tag--active" : ""].filter(Boolean).join(" ")}
                aria-label={reasoningEnabled ? "Tắt Reasoning" : "Bật Reasoning"}
                title={
                  reasoningEnabled
                    ? "LLM reasoning đang bật: temporal order, scene split và focused fusion queries"
                    : "Bật LLM reasoning cho temporal/fusion"
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
                aria-label="Cấu hình Fusion search"
                title="Cấu hình model, method và weight cho fusion search"
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
              disabled={loading || disabled || (!query.trim() && !hasQueryImages)}
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
        value={fusionConfig}
        onClose={() => setFusionModalOpen(false)}
        onSave={(next) => onFusionConfigChange?.({ ...next, hasConfig: true })}
      />
    </form>
  );
}
