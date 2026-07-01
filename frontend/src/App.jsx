import {
  startTransition,
  useCallback,
  useDeferredValue,
  useEffect,
  useMemo,
  useRef,
  useState,
  useTransition,
} from "react";
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
  rerankResults,
} from "./api/retrievalAPI";
import RerankPopover from "./components/RerankPopover";
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
  const surroundReqRef = useRef(0);
  const similarReqRef = useRef(0);
  const rerankRunRef = useRef(0);

  const [, startNonUrgentUpdate] = useTransition();

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
  const [dres, setDres] = useState({
    loading: false,
    sessionId: "",
    user: null,
  });

  // ── Toasts ───────────────────────────────────────────
  const [toasts, setToasts] = useState([]);

  // ── Rerank state ─────────────────────────────────────
  const [rerankEnabled, setRerankEnabled] = useState(false);
  const [reranking, setReranking] = useState(false);
  const [rerankResultsData, setRerankResultsData] = useState(null);

  // ── Search hook ──────────────────────────────────────
  const {
    results: rawResults,
    latency,
    count,
    lastQuery,
    loading,
    error,
    search,
    reset,
  } = useRetrievalSearch();

  const results = rerankResultsData ?? rawResults;
  const deferredResults = useDeferredValue(results);
  const hasResults = deferredResults.length > 0;
  const selectedId = selected?.id ?? null;

  const resolvedMode = useMemo(() => {
    return mode === "temporal" ||
      mode === "auto" ||
      mode === "ocr" ||
      mode === "asr"
      ? mode
      : "semantic";
  }, [mode]);

  const isHeavyDataset = deferredResults.length >= 120;

  const rootClassName = useMemo(() => {
    const themeClass = theme === "dark" ? "theme-dark" : "theme-light";
    return `${themeClass}${isHeavyDataset ? " performance-mode" : ""}`;
  }, [theme, isHeavyDataset]);

  const searchBarProps = useMemo(
    () => ({
      model,
      mode,
      loading,
      disabled: !backendReady,
      durationLimit,
      rerankEnabled,
      onModelChange: setModel,
      onModeChange: setMode,
      onDurationLimitChange: setDurationLimit,
      onRerankToggle: setRerankEnabled,
    }),
    [model, mode, loading, backendReady, durationLimit, rerankEnabled]
  );

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

        if (config.model?.name) {
          setModel(config.model.name);
        }

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
    return () => {
      for (const timer of toastTimersRef.current.values()) {
        clearTimeout(timer);
      }
      toastTimersRef.current.clear();
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

  // ── Stable UI handlers ───────────────────────────────
  const handleToggleTheme = useCallback(() => {
    setTheme((prev) => (prev === "dark" ? "light" : "dark"));
  }, []);

  const handleOpenSettings = useCallback(() => {
    setSettingsOpen(true);
  }, []);

  const handleCloseSettings = useCallback(() => {
    setSettingsOpen(false);
  }, []);

  const handleColumnsChange = useCallback((value) => {
    startNonUrgentUpdate(() => {
      setColumns(value);
    });
  }, []);

  const handleGroupedChange = useCallback((value) => {
    startNonUrgentUpdate(() => {
      setGrouped(value);
    });
  }, []);

  const handleSelectResult = useCallback((result) => {
    startNonUrgentUpdate(() => {
      setSelected(result);
    });
  }, []);

  const handleCloseDetail = useCallback(() => {
    startNonUrgentUpdate(() => {
      setSelected(null);
    });
  }, []);

  // ── Main actions ─────────────────────────────────────
  const handleReset = useCallback(() => {
    searchIdRef.current += 1;
    rerankRunRef.current += 1;
    surroundReqRef.current += 1;
    similarReqRef.current += 1;

    reset();
    setSelected(null);
    setGrouped(false);
    setSurroundModal(DEFAULT_SURROUND_MODAL);
    setSimilarModal(DEFAULT_SIMILAR_MODAL);
    setModalOrder([]);
    setVideoResult(null);
    setRerankResultsData(null);
    setReranking(false);
  }, [reset]);

  const handleSearch = useCallback(
    async (payload) => {
      const query = typeof payload === "string" ? payload : payload?.query;
      const cleanQuery = String(query || "").trim();

      if (!cleanQuery || loading || !backendReady) return;

      const searchId = ++searchIdRef.current;
      rerankRunRef.current += 1;

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

      startNonUrgentUpdate(() => {
        setSelected(null);
        setRerankResultsData(null);
      });

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
        if (err?.name === "AbortError" || err?.name === "StaleSearchError") {
          return;
        }
        if (searchId !== searchIdRef.current) return;
        console.error(err);
        pushToast("warning", "Search failed", getErrorMessage(err));
      }
    },
    [backendReady, durationLimit, loading, pushToast, resolvedMode, search, settings]
  );

  // Auto-rerank tối ưu hơn: debounce + chống stale update
  useEffect(() => {
    if (!rerankEnabled || rawResults.length === 0 || loading || !lastQuery) return;

    const runId = ++rerankRunRef.current;
    let cancelled = false;

    const timer = setTimeout(async () => {
      if (cancelled) return;

      setReranking(true);

      try {
        const data = await rerankResults({
          results: rawResults,
          query: lastQuery,
          searchMode: resolvedMode,
          topCandidate: 1.0,
          topK: settings.topK,
        });

        if (cancelled || runId !== rerankRunRef.current) return;

        startTransition(() => {
          setRerankResultsData(data.results ?? []);
        });
      } catch (err) {
        if (cancelled || runId !== rerankRunRef.current) return;
        console.warn("[AUTO-RERANK] failed:", err?.message || err);
        pushToast("warning", "Auto-rerank thất bại", getErrorMessage(err));
      } finally {
        if (!cancelled && runId === rerankRunRef.current) {
          setReranking(false);
        }
      }
    }, isHeavyDataset ? 260 : 140);

    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [
    rawResults,
    rerankEnabled,
    loading,
    lastQuery,
    resolvedMode,
    settings.topK,
    pushToast,
    isHeavyDataset,
  ]);

  const handleManualRerank = useCallback(
    async ({ query, topCandidate, topK }) => {
      if (rawResults.length === 0) return;

      const runId = ++rerankRunRef.current;
      setReranking(true);

      startNonUrgentUpdate(() => {
        setRerankResultsData(null);
      });

      try {
        const data = await rerankResults({
          results: rawResults,
          query,
          searchMode: resolvedMode,
          topCandidate,
          topK,
        });

        if (runId !== rerankRunRef.current) return;

        startTransition(() => {
          setRerankResultsData(data.results ?? []);
        });

        pushToast("correct", "Rerank xong", `${data.count} kết quả`);
      } catch (err) {
        if (runId !== rerankRunRef.current) return;
        pushToast("warning", "Rerank thất bại", getErrorMessage(err));
      } finally {
        if (runId === rerankRunRef.current) {
          setReranking(false);
        }
      }
    },
    [rawResults, resolvedMode, pushToast]
  );

  const handleOpenSurroundingImages = useCallback(
    async (result) => {
      if (!result) return;

      const reqId = ++surroundReqRef.current;

      setModalOrder((prev) => [...prev.filter((x) => x !== "surround"), "surround"]);
      setSurroundModal({
        open: true,
        center: result,
        frames: [result],
        loading: true,
      });

      try {
        const frames = await getSurroundingFrames(result.video_id, result.frame_id, 12);

        if (reqId !== surroundReqRef.current) return;

        setSurroundModal({
          open: true,
          center: result,
          frames,
          loading: false,
        });
      } catch (err) {
        if (reqId !== surroundReqRef.current) return;

        console.error(err);
        setSurroundModal({
          open: true,
          center: result,
          frames: [result],
          loading: false,
        });
        pushToast("warning", "Surrounding frames failed", getErrorMessage(err));
      }
    },
    [pushToast]
  );

  const handleSimilaritySearch = useCallback(
    async (result) => {
      if (!result) return;

      const reqId = ++similarReqRef.current;

      setModalOrder((prev) => [...prev.filter((x) => x !== "similar"), "similar"]);
      setSimilarModal({
        open: true,
        source: result,
        frames: [],
        loading: true,
      });

      try {
        const data = await similaritySearch({
          videoId: result.video_id,
          frameId: result.frame_id,
          topK: settings.topK,
        });

        if (reqId !== similarReqRef.current) return;

        setSimilarModal({
          open: true,
          source: result,
          frames: data.results ?? [],
          loading: false,
        });
      } catch (err) {
        if (reqId !== similarReqRef.current) return;

        console.error(err);
        setSimilarModal({
          open: true,
          source: result,
          frames: [],
          loading: false,
        });
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

      setDres({
        loading: false,
        sessionId: data.session_id,
        user: data.user ?? null,
      });

      if (data.evaluation_id) {
        setSettings((prev) => ({
          ...prev,
          evaluationId: data.evaluation_id,
        }));
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

          setDres({
            loading: false,
            sessionId,
            user: loginData.user ?? null,
          });

          if (loginData.evaluation_id) {
            setSettings((prev) => ({
              ...prev,
              evaluationId: loginData.evaluation_id,
            }));
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
    surroundReqRef.current += 1;
    similarReqRef.current += 1;

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

  return (
    <div className={rootClassName}>
      <div className={`ambient-bg ${loading && !isHeavyDataset ? "ambient-searching" : ""}`} />

      <div className="app-root">
        <Sidebar
          theme={theme}
          onToggleTheme={handleToggleTheme}
          onReset={handleReset}
          onOpenSettings={handleOpenSettings}
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
                  onColumnsChange={handleColumnsChange}
                  onGroupedChange={handleGroupedChange}
                />

                <div className="query-summary">
                  <span>
                    Query: <strong>{lastQuery}</strong>
                  </span>

                  <span className="query-summary-right">
                    {reranking && (
                      <span className="rerank-status-badge">
                        <span className="rerank-spinner" /> VLM đang rerank...
                      </span>
                    )}

                    {rerankResultsData && !reranking && (
                      <span className="rerank-status-badge rerank-status-badge--done">
                        ✓ Đã rerank
                      </span>
                    )}

                    <span>{count} results</span>

                    {!rerankEnabled && (
                      <RerankPopover
                        currentQuery={lastQuery}
                        searchMode={resolvedMode}
                        loading={loading}
                        reranking={reranking}
                        onRerank={handleManualRerank}
                      />
                    )}
                  </span>
                </div>

                <div className="result-body">
                  <div className="result-list">
                    {grouped ? (
                      <GroupedResults
                        results={deferredResults}
                        columns={columns}
                        selectedId={selectedId}
                        onSelect={handleSelectResult}
                        onSubmit={handleSubmitResult}
                        onPlay={handlePlayResult}
                        onSimilaritySearch={handleSimilaritySearch}
                        onSurroundingImages={handleOpenSurroundingImages}
                      />
                    ) : (
                      <ResultGrid
                        results={deferredResults}
                        columns={columns}
                        selectedId={selectedId}
                        onSelect={handleSelectResult}
                        onSubmit={handleSubmitResult}
                        onPlay={handlePlayResult}
                        onSimilaritySearch={handleSimilaritySearch}
                        onSurroundingImages={handleOpenSurroundingImages}
                      />
                    )}
                  </div>

                  {selected && (
                    <DetailPanel
                      result={selected}
                      onClose={handleCloseDetail}
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
          onClose={handleCloseSettings}
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
          onSelect={handleSelectResult}
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
          onSelect={handleSelectResult}
          onSubmit={handleSubmitResult}
          onSimilaritySearch={handleSimilaritySearch}
          onSurroundingImages={handleOpenSurroundingImages}
        />

        <ToastHost toasts={toasts} />
      </div>
    </div>
  );
}
