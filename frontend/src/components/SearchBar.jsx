import { Plus, Search, Mic } from "lucide-react";
import { useEffect, useRef, useState, useCallback } from "react";

export default function SearchBar({
  model,
  mode,
  loading,
  disabled = false,
  durationLimit = -1,
  onModelChange,
  onModeChange,
  onDurationLimitChange,
  onSearch,
}) {
  const [query, setQuery] = useState("");
  const [recording, setRecording] = useState(false);

  const textareaRef = useRef(null);
  const recognitionRef = useRef(null);
  const committedTranscriptRef = useRef("");

  const isTemporal = mode === "temporal";
  const isExpanded = query.includes("\n") || query.length > 80;

  const resizeTextarea = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;

    requestAnimationFrame(() => {
      el.style.height = "auto";
      el.style.height = `${Math.min(el.scrollHeight, 220)}px`;
    });
  }, []);

  useEffect(() => {
    resizeTextarea();
  }, [query, resizeTextarea]);

  useEffect(() => {
    return () => {
      stopBrowserSpeech();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function resolveSearchMode() {
    if (mode === "temporal") return "temporal";
    if (mode === "auto") return "auto";
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
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;

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

    recognition.onstart = () => {
      setRecording(true);
    };

    recognition.onresult = (event) => {
      let finalText = "";
      let interimText = "";

      for (let i = event.resultIndex; i < event.results.length; i += 1) {
        const transcript = event.results[i][0].transcript.trim();

        if (!transcript) continue;

        if (event.results[i].isFinal) {
          finalText += ` ${transcript}`;
        } else {
          interimText += ` ${transcript}`;
        }
      }

      if (finalText.trim()) {
        committedTranscriptRef.current = `${committedTranscriptRef.current} ${finalText}`
          .replace(/\s+/g, " ")
          .trim();
      }

      const nextText = `${committedTranscriptRef.current} ${interimText}`
        .replace(/\s+/g, " ")
        .trim();

      setQuery(nextText);
    };

    recognition.onerror = (event) => {
      console.error("Speech recognition error:", event.error);

      if (event.error === "network") {
        alert(
          "Speech Recognition bị lỗi network. Hãy kiểm tra internet/VPN/firewall, hoặc thử Chrome/Edge khác."
        );
      }

      if (event.error === "not-allowed") {
        alert("Trình duyệt chưa được cấp quyền microphone.");
      }
    };

    recognition.onend = () => {
      setRecording(false);
      recognitionRef.current = null;
    };

    recognitionRef.current = recognition;
    recognition.start();
  }

  function stopBrowserSpeech() {
    recognitionRef.current?.stop();
    recognitionRef.current = null;
    setRecording(false);
  }

  function handleMicClick() {
    if (recording) {
      stopBrowserSpeech();
      return;
    }

    startBrowserSpeech();
  }

  return (
    <form
      className={[
        "search-wrapper",
        loading ? "searching-active" : "",
        isExpanded ? "expanded" : "",
        recording ? "voice-recording-active" : "",
      ]
        .filter(Boolean)
        .join(" ")}
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
            <select
              className="search-select"
              value={model}
              onChange={(e) => onModelChange(e.target.value)}
            >
              <option value="OpenCLIP">OpenCLIP</option>
              <option value="SigLIP">SigLIP</option>
              <option value="DINOv3">DINOv3</option>
              <option value="Hybrid">Hybrid</option>
            </select>

            <select
              className="search-select"
              value={mode}
              onChange={(e) => onModeChange(e.target.value)}
            >
              <option value="text">Semantic</option>
              <option value="temporal">Temporal</option>
              <option value="auto">Auto</option>
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

            <button
              type="button"
              className={[
                "search-icon-button",
                "mic-button",
                recording ? "recording" : "",
              ]
                .filter(Boolean)
                .join(" ")}
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
    </form>
  );
}