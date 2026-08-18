import { Plus, Search, Mic, Zap, Settings2 } from "lucide-react";
import { useEffect, useRef, useState, useCallback } from "react";
import FusionSettingsModal from "./FusionSettingsModal";

const FALLBACK_MODELS = ["siglip2-so400m", "vitH-378-quickgelu"];

export default function SearchBar({
  model,
  mode,
  loading,
  disabled = false,
  durationLimit = -1,
  rerankEnabled = false,
  availableModels = [],
  fusionConfig,
  translateProvider = "google",
  onModelChange,
  onModeChange,
  onDurationLimitChange,
  onRerankToggle,
  onFusionConfigChange,
  onTranslateProviderChange,
  expandedByDefault = false,
  onSearch,
}) {
  const [query, setQuery] = useState("");
  const [recording, setRecording] = useState(false);
  const [fusionModalOpen, setFusionModalOpen] = useState(false);
  // The home composer starts expanded; the bottom composer starts compact
  // and expands only when the query wraps past the normal height.
  const [isExpanded, setIsExpanded] = useState(expandedByDefault);
  const expandedRef = useRef(expandedByDefault);

  const modelOptions = availableModels.length > 0 ? availableModels : FALLBACK_MODELS;

  const textareaRef = useRef(null);
  const recognitionRef = useRef(null);
  const committedTranscriptRef = useRef("");

  const isTemporal = mode === "temporal";
  const isOcr = mode === "ocr";
  const isAsr = mode === "asr";
  const isFusion = mode === "fusion";
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function resolveSearchMode() {
    if (mode === "temporal" || mode === "auto" || mode === "ocr" || mode === "asr" || mode === "fusion") return mode;
    return "semantic";
  }

  function runSearch(nextQuery) {
    const cleanQuery = String(nextQuery || "").trim();
    if (!cleanQuery || loading || disabled) return;
    const searchMode = resolveSearchMode();
    onSearch({
      query: cleanQuery,
      searchMode,
      durationLimit: searchMode === "temporal" ? Number(durationLimit) : -1,
      fusionConfig: searchMode === "fusion" ? fusionConfig : null,
    });
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
      setQuery(nextText);
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
        isExpanded ? "expanded" : "",
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
            setQuery(e.target.value);
          }}
          onKeyDown={handleKeyDown}
        />

        <div className="search-chat-footer">
          <button type="button" className="search-icon-button" aria-label="Attach">
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

            <select
              className="search-select search-provider-select"
              value={translateProvider}
              onChange={(e) => onTranslateProviderChange?.(e.target.value)}
              title="Translation provider"
              aria-label="Translation provider"
            >
              <option value="google">Google</option>
              <option value="llm">LLM</option>
              <option value="envit5">EnViT5</option>
            </select>

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

            {/* ── Rerank toggle tag ───────────────────────────────────── */}
            {!isFusion && (
              <button
                type="button"
                className={["rerank-tag", rerankEnabled ? "rerank-tag--active" : ""].filter(Boolean).join(" ")}
                aria-label={rerankEnabled ? "Tắt Auto-Rerank" : "Bật Auto-Rerank (VLM)"}
                title={
                  rerankEnabled
                    ? "Auto-Rerank đang BẬT — sau khi search, VLM sẽ tự động chấm lại kết quả"
                    : "Bật Auto-Rerank — VLM sẽ load ngầm và cập nhật kết quả"
                }
                onClick={() => onRerankToggle?.(!rerankEnabled)}
                disabled={disabled}
              >
                <Zap size={13} />
                <span>Rerank</span>
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
              disabled={loading || disabled}
              aria-label="Search"
            >
              <Search size={18} />
            </button>
          </div>
        </div>
      </div>

      <FusionSettingsModal
        open={fusionModalOpen}
        models={modelOptions}
        value={fusionConfig}
        onClose={() => setFusionModalOpen(false)}
        onSave={(next) => onFusionConfigChange?.({ ...next, hasConfig: true })}
      />
    </form>
  );
}
