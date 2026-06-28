import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Sidebar from "./components/Sidebar";
import SearchBar from "./components/SearchBar";
import ResultToolbar from "./components/ResultToolbar";
import ResultGrid from "./components/ResultGrid";
import GroupedResults from "./components/GroupedResults";
import DetailPanel from "./components/DetailPanel";
import SettingsPanel from "./components/SettingsPanel";
import ToastHost from "./components/ToastHost";
import SurroundingFramesModal from "./components/SurroundingFramesModal";
import SimilarityFramesModal from "./components/SimilarityFramesModal";
import VideoModal from "./components/VideoModal";
import { useRetrievalSearch } from "./hooks/useRetrievalSearch";
import {
  checkBackendHealth,
  getBackendConfig,
  getSurroundingFrames,
  similaritySearch,
} from "./api/retrievalAPI";
import {
  getDefaultSubmissionSettings,
  loginDresViaBackend,
  submitDresViaBackend,
} from "./api/submissionAPI";
import { playNotifySound } from "./utils/notifySound";

const DEFAULT_SURROUND_MODAL = Object.freeze({
  open: false,
  center: null,
  frames: [],
  loading: false,
});

const DEFAULT_SIMILAR_MODAL = Object.freeze({
  open: false,
  source: null,
  frames: [],
  loading: false,
});

function getResultLabel(result) {
  if (!result) return "";
  return `${result.video_id}/${String(result.frame_id ?? 0).padStart(6, "0")}`;
}

function getErrorMessage(error, fallback = "Unexpected error") {
  return error?.message || String(error || fallback);
}

export default function App() {
  const defaultSubmission = useMemo(() => getDefaultSubmissionSettings(), []);
  const searchIdRef = useRef(0);
  const toastTimersRef = useRef(new Map());

  // ── UI state ──────────────────────────────────────────
  const [theme, setTheme] = useState("dark");
  const [model, setModel] = useState("ViT-B-16-quickgelu");
  const [mode, setMode] = useState("text");
  const [durationLimit, setDurationLimit] = useState(-1);
  const [columns, setColumns] = useState(4);
  const [grouped, setGrouped] = useState(false);
  const [selected, setSelected] = useState(null);

  // ── Modal state ───────────────────────────────────────
  const [surroundModal, setSurroundModal] = useState(DEFAULT_SURROUND_MODAL);
  const [surroundColumns, setSurroundColumns] = useState(5);
  const [similarModal, setSimilarModal] = useState(DEFAULT_SIMILAR_MODAL);
  const [similarColumns, setSimilarColumns] = useState(5);
  const [modalOrder, setModalOrder] = useState([]);
  const [videoResult, setVideoResult] = useState(null);

  // ── Settings ─────────────────────────────────────────
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settings, setSettings] = useState({
    useSplit: true,
    useTranslate: true,
    topK: 20,
    candidateMultiplier: 5,
    submitUrl: defaultSubmission.dresUrl,
    evaluationId: defaultSubmission.evaluationId,
    username: defaultSubmission.teamId,
    password: defaultSubmission.teamPassword,
  });

  // ── Backend ──────────────────────────────────────────
  const [backendReady, setBackendReady] = useState(false);
  const [backendStatus, setBackendStatus] = useState("Checking backend...");

  // ── DRES session ─────────────────────────────────────
  const [dres, setDres] = useState({ loading: false, sessionId: "", user: null });

  // ── Toasts ───────────────────────────────────────────
  const [toasts, setToasts] = useState([]);

  // ── Search hook ──────────────────────────────────────
  const { results, latency, count, lastQuery, loading, error, search, reset } =
    useRetrievalSearch();

  const hasResults = results.length > 0;
  const resolvedMode = mode === "temporal" ? "temporal" : mode === "auto" ? "auto" : "semantic";

  const searchBarProps = {
    model,
    mode,
    loading,
    disabled: !backendReady,
    durationLimit,
    onModelChange: setModel,
    onModeChange: setMode,
    onDurationLimitChange: setDurationLimit,
  };

  // ── Bootstrap ────────────────────────────────────────
  useEffect(() => {
    let alive = true;

    async function bootstrap() {
      try {
        await checkBackendHealth();
        const config = await getBackendConfig();

        if (!alive) return;

        setSettings((prev) => ({
          ...prev,
          topK: config.search?.default_top_k ?? prev.topK,
          candidateMultiplier:
            config.search?.candidate_multiplier ?? prev.candidateMultiplier,
          useTranslate: config.translate?.enabled_default ?? prev.useTranslate,
        }));

        if (config.model?.name) setModel(config.model.name);

        setBackendReady(true);
        setBackendStatus("Backend connected");
      } catch (err) {
        if (!alive) return;
        setBackendReady(false);
        setBackendStatus(getErrorMessage(err, "Backend disconnected"));
      }
    }

    bootstrap();

    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    const timers = toastTimersRef.current;
    return () => {
      for (const timer of timers.values()) {
        clearTimeout(timer);
      }
      timers.clear();
    };
  }, []);

  // ── Toast helpers ────────────────────────────────────
  const dismissToast = useCallback((id) => {
    const timer = toastTimersRef.current.get(id);
    if (timer) clearTimeout(timer);
    toastTimersRef.current.delete(id);
    setToasts((prev) => prev.filter((toast) => toast.id !== id));
  }, []);

  const pushToast = useCallback(
    (type, title, message = "") => {
      const id = crypto.randomUUID();
      setToasts((prev) => [...prev, { id, type, title, message }]);
      playNotifySound(type);

      const timer = setTimeout(() => dismissToast(id), 3200);
      toastTimersRef.current.set(id, timer);
    },
    [dismissToast]
  );

  // ── Handlers ─────────────────────────────────────────
  const handleReset = useCallback(() => {
    searchIdRef.current += 1;
    reset();
    setSelected(null);
    setGrouped(false);
    setSurroundModal(DEFAULT_SURROUND_MODAL);
    setSimilarModal(DEFAULT_SIMILAR_MODAL);
    setModalOrder([]);
    setVideoResult(null);
  }, [reset]);

  const handleSearch = useCallback(
    async (payload) => {
      const query = typeof payload === "string" ? payload : payload?.query;
      const cleanQuery = String(query || "").trim();
      if (!cleanQuery || loading || !backendReady) return;

      const searchId = ++searchIdRef.current;

      const searchMode =
        typeof payload === "object" && payload?.searchMode
          ? payload.searchMode
          : resolvedMode;

      const nextDurationLimit =
        typeof payload === "object" && payload?.durationLimit !== undefined
          ? Number(payload.durationLimit)
          : searchMode === "temporal"
            ? Number(durationLimit)
            : -1;

      setSelected(null);

      try {
        await search({
          query: cleanQuery,
          topK: settings.topK,
          candidateMultiplier: settings.candidateMultiplier,
          useSplit: settings.useSplit,
          useTranslate: settings.useTranslate,
          searchMode,
          durationLimit: nextDurationLimit,
        });
      } catch (err) {
        if (err?.name === "AbortError" || err?.name === "StaleSearchError") return;
        if (searchId !== searchIdRef.current) return;
        console.error(err);
        pushToast("warning", "Search failed", getErrorMessage(err));
      }
    },
    [backendReady, durationLimit, loading, pushToast, resolvedMode, search, settings]
  );

  const handleOpenSurroundingImages = useCallback(
    async (result) => {
      if (!result) return;

      setModalOrder((prev) => [...prev.filter((x) => x !== "surround"), "surround"]);
      setSurroundModal({ open: true, center: result, frames: [result], loading: true });

      try {
        const frames = await getSurroundingFrames(result.video_id, result.frame_id, 12);
        setSurroundModal({ open: true, center: result, frames, loading: false });
      } catch (err) {
        console.error(err);
        setSurroundModal({ open: true, center: result, frames: [result], loading: false });
        pushToast("warning", "Surrounding frames failed", getErrorMessage(err));
      }
    },
    [pushToast]
  );

  const handleSimilaritySearch = useCallback(
    async (result) => {
      if (!result) return;

      setModalOrder((prev) => [...prev.filter((x) => x !== "similar"), "similar"]);
      setSimilarModal({ open: true, source: result, frames: [], loading: true });

      try {
        const data = await similaritySearch({
          videoId: result.video_id,
          frameId: result.frame_id,
          topK: settings.topK,
        });

        setSimilarModal({
          open: true,
          source: result,
          frames: data.results ?? [],
          loading: false,
        });
      } catch (err) {
        console.error(err);
        setSimilarModal({ open: true, source: result, frames: [], loading: false });
        pushToast("warning", "Similarity search failed", getErrorMessage(err));
      }
    },
    [pushToast, settings.topK]
  );

  const handleDresLogin = useCallback(async () => {
    setDres((prev) => ({ ...prev, loading: true }));

    try {
      const data = await loginDresViaBackend({
        dresUrl: settings.submitUrl,
        username: settings.username,
        password: settings.password,
      });

      setDres({ loading: false, sessionId: data.session_id, user: data.user ?? null });

      if (data.evaluation_id) {
        setSettings((prev) => ({ ...prev, evaluationId: data.evaluation_id }));
      }

      pushToast(
        "correct",
        "DRES logged in",
        data.evaluation_id
          ? `Session + evaluation: ${data.evaluation_id}`
          : "Session ID received."
      );
    } catch (err) {
      setDres((prev) => ({ ...prev, loading: false }));
      pushToast("warning", "DRES login failed", getErrorMessage(err));
    }
  }, [pushToast, settings.password, settings.submitUrl, settings.username]);

  const handleDresLogout = useCallback(() => {
    setDres({ loading: false, sessionId: "", user: null });
    pushToast("pending", "DRES session cleared");
  }, [pushToast]);

  const handleSubmitResult = useCallback(
    async (result) => {
      if (!result) {
        pushToast("warning", "Cannot submit", "No selected frame.");
        return;
      }

      try {
        let sessionId = dres.sessionId;
        let evaluationId = settings.evaluationId;

        if (!sessionId) {
          const loginData = await loginDresViaBackend({
            dresUrl: settings.submitUrl,
            username: settings.username,
            password: settings.password,
          });

          sessionId = loginData.session_id;
          evaluationId = loginData.evaluation_id || evaluationId;

          setDres({ loading: false, sessionId, user: loginData.user ?? null });

          if (loginData.evaluation_id) {
            setSettings((prev) => ({ ...prev, evaluationId: loginData.evaluation_id }));
          }
        }

        const response = await submitDresViaBackend({
          dresUrl: settings.submitUrl,
          sessionId,
          evaluationId,
          result,
        });

        const label = getResultLabel(result);

        if (response.status === "correct") {
          pushToast("correct", "Correct", label);
        } else if (response.status === "wrong") {
          pushToast("wrong", "Wrong", response.message || label);
        } else if (response.status === "pending") {
          pushToast("pending", "Submitted", response.message || label);
        } else {
          pushToast("warning", "Cannot submit", response.message || label);
        }
      } catch (err) {
        pushToast("warning", "Cannot submit", getErrorMessage(err));
      }
    },
    [dres.sessionId, pushToast, settings]
  );

  const closeAllModals = useCallback(() => {
    setSurroundModal(DEFAULT_SURROUND_MODAL);
    setSimilarModal(DEFAULT_SIMILAR_MODAL);
    setVideoResult(null);
    setModalOrder([]);
  }, []);

  const handlePlayResult = useCallback((result) => {
    if (!result) return;
    setVideoResult(result);
    setModalOrder((prev) => [...prev.filter((x) => x !== "video"), "video"]);
  }, []);

  function modalLayer(name) {
    const index = modalOrder.indexOf(name);
    return index < 0 ? 0 : index + 1;
  }

  // ── Render ───────────────────────────────────────────
  return (
    <div className={theme === "dark" ? "theme-dark" : "theme-light"}>
      <div className={`ambient-bg ${loading ? "ambient-searching" : ""}`} />

      <div className="app-root">
        <Sidebar
          theme={theme}
          onToggleTheme={() => setTheme((prev) => (prev === "dark" ? "light" : "dark"))}
          onReset={handleReset}
          onOpenSettings={() => setSettingsOpen(true)}
        />

        <main className={hasResults ? "main-layout has-results" : "main-layout is-home"}>
          {!hasResults && (
            <section className="home-panel">
              <div className="backend-status">
                <span className={backendReady ? "backend-dot connected" : "backend-dot"} />
                {backendStatus}
              </div>

              <h1 className="main-title">Bạn muốn retrieval sự kiện nào?</h1>

              <SearchBar {...searchBarProps} onSearch={handleSearch} />
            </section>
          )}

          {hasResults && (
            <>
              <section className="display-area">
                <ResultToolbar
                  model={model}
                  latency={latency}
                  columns={columns}
                  grouped={grouped}
                  onColumnsChange={setColumns}
                  onGroupedChange={setGrouped}
                />

                <div className="query-summary">
                  <span>
                    Query: <strong>{lastQuery}</strong>
                  </span>
                  <span>{count} results</span>
                </div>

                <div className="result-body">
                  <div className="result-list">
                    {grouped ? (
                      <GroupedResults
                        results={results}
                        columns={columns}
                        selectedId={selected?.id}
                        onSelect={setSelected}
                        onSubmit={handleSubmitResult}
                        onPlay={handlePlayResult}
                        onSimilaritySearch={handleSimilaritySearch}
                        onSurroundingImages={handleOpenSurroundingImages}
                      />
                    ) : (
                      <ResultGrid
                        results={results}
                        columns={columns}
                        selectedId={selected?.id}
                        onSelect={setSelected}
                        onSubmit={handleSubmitResult}
                        onPlay={handlePlayResult}
                        onSimilaritySearch={handleSimilaritySearch}
                        onSurroundingImages={handleOpenSurroundingImages}
                      />
                    )}
                  </div>

                  {selected && (
                    <DetailPanel
                      key={selected.id}
                      result={selected}
                      onClose={() => setSelected(null)}
                      onSubmit={handleSubmitResult}
                    />
                  )}
                </div>
              </section>

              <div className={`bottom-search-zone ${loading ? "is-searching" : ""}`}>
                <SearchBar {...searchBarProps} onSearch={handleSearch} />

                <p className="footer-note">
                  Retrieval result có thể thiếu chính xác, cần kiểm tra lại bằng video gốc.
                </p>
              </div>
            </>
          )}

          {error && <p className="error-text">{error}</p>}
        </main>

        <VideoModal
          open={Boolean(videoResult)}
          result={videoResult}
          layer={modalLayer("video")}
          onClose={closeAllModals}
          onSubmit={handleSubmitResult}
        />

        <SettingsPanel
          open={settingsOpen}
          settings={settings}
          dres={dres}
          onChange={setSettings}
          onClose={() => setSettingsOpen(false)}
          onDresLogin={handleDresLogin}
          onDresLogout={handleDresLogout}
        />

        <SurroundingFramesModal
          open={surroundModal.open}
          centerResult={surroundModal.center}
          frames={surroundModal.frames}
          loading={surroundModal.loading}
          columns={surroundColumns}
          onColumnsChange={setSurroundColumns}
          layer={modalLayer("surround")}
          onClose={closeAllModals}
          onSelect={setSelected}
          onSubmit={handleSubmitResult}
          onSimilaritySearch={handleSimilaritySearch}
          onSurroundingImages={handleOpenSurroundingImages}
        />

        <SimilarityFramesModal
          open={similarModal.open}
          sourceResult={similarModal.source}
          frames={similarModal.frames}
          loading={similarModal.loading}
          columns={similarColumns}
          onColumnsChange={setSimilarColumns}
          layer={modalLayer("similar")}
          onClose={closeAllModals}
          onSelect={setSelected}
          onSubmit={handleSubmitResult}
          onSimilaritySearch={handleSimilaritySearch}
          onSurroundingImages={handleOpenSurroundingImages}
        />

        <ToastHost toasts={toasts} />
      </div>
    </div>
  );
}
