import { useEffect, useRef, useState } from "react";
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
import { useRetrievalSearch } from "./hooks/useRetrievalSearch";
import {
  getBackendConfig,
  checkBackendHealth,
  getSurroundingFrames,
  similaritySearch,
} from "./api/retrievalAPI";
import {
  getDefaultSubmissionSettings,
  loginDresViaBackend,
  submitDresViaBackend,
} from "./api/submissionAPI";
import { playNotifySound } from "./utils/notifySound";

export default function App() {
  const defaultSubmission = getDefaultSubmissionSettings();
  const searchIdRef = useRef(0);

  const [theme, setTheme] = useState("dark");
  const [model, setModel] = useState("ViT-B-16-quickgelu");
  const [mode, setMode] = useState("text");
  const [durationLimit, setDurationLimit] = useState(-1);
  const [columns, setColumns] = useState(4);
  const [grouped, setGrouped] = useState(false);
  const [selected, setSelected] = useState(null);

  const [surroundModal, setSurroundModal] = useState({
    open: false,
    center: null,
    frames: [],
    loading: false,
  });
  const [surroundColumns, setSurroundColumns] = useState(5);

  const [similarModal, setSimilarModal] = useState({
    open: false,
    source: null,
    frames: [],
    loading: false,
  });
  const [similarColumns, setSimilarColumns] = useState(5);

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

  const [backendReady, setBackendReady] = useState(false);
  const [backendStatus, setBackendStatus] = useState("Checking backend...");

  const [dres, setDres] = useState({
    loading: false,
    sessionId: "",
    user: null,
  });

  const [toasts, setToasts] = useState([]);

  const {
    results,
    latency,
    count,
    lastQuery,
    loading,
    error,
    search,
    reset,
  } = useRetrievalSearch();

  const hasResults = results.length > 0;

  useEffect(() => {
    async function bootstrap() {
      try {
        await checkBackendHealth();
        const config = await getBackendConfig();

        setSettings((prev) => ({
          ...prev,
          topK: config.search?.default_top_k ?? 20,
          candidateMultiplier: config.search?.candidate_multiplier ?? 5,
          useTranslate: config.translate?.enabled_default ?? true,
        }));

        if (config.model?.name) {
          setModel(config.model.name);
        }

        setBackendReady(true);
        setBackendStatus("Backend connected");
      } catch (err) {
        setBackendReady(false);
        setBackendStatus(err.message || "Backend disconnected");
      }
    }

    bootstrap();
  }, []);

  function handleReset() {
    searchIdRef.current += 1;
    reset();
    setSelected(null);
    setGrouped(false);

    setSurroundModal({
      open: false,
      center: null,
      frames: [],
      loading: false,
    });

    setSimilarModal({
      open: false,
      source: null,
      frames: [],
      loading: false,
    });
  }

  async function handleSearch(payload) {
    const query = typeof payload === "string" ? payload : payload?.query;

    if (!query || !String(query).trim()) {
      return;
    }

    const searchId = ++searchIdRef.current;

    const searchMode =
      typeof payload === "object" && payload?.searchMode
        ? payload.searchMode
        : mode === "temporal"
          ? "temporal"
          : mode === "auto"
            ? "auto"
            : "semantic";

    const nextDurationLimit =
      typeof payload === "object" && payload?.durationLimit !== undefined
        ? Number(payload.durationLimit)
        : searchMode === "temporal"
          ? Number(durationLimit)
          : -1;

    setSelected(null);

    try {
      await search({
        query: String(query).trim(),
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

      if (searchId !== searchIdRef.current) {
        return;
      }

      console.error(err);
    }
  }

  async function handleOpenSurroundingImages(result) {
    console.log("[SURROUND OPEN]", result);

    if (!result) return;

    setSurroundModal({
      open: true,
      center: result,
      frames: [result],
      loading: true,
    });

    try {
      const frames = await getSurroundingFrames(
        result.video_id,
        result.frame_id,
        12
      );

      setSurroundModal({
        open: true,
        center: result,
        frames,
        loading: false,
      });
    } catch (err) {
      console.error(err);

      setSurroundModal({
        open: true,
        center: result,
        frames: [result],
        loading: false,
      });

      pushToast("warning", "Surrounding frames failed", err.message);
    }
  }

  async function handleSimilaritySearch(result) {
    console.log("[SIMILAR OPEN]", result);

    if (!result) return;

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

      setSimilarModal({
        open: true,
        source: result,
        frames: data.results ?? [],
        loading: false,
      });
    } catch (err) {
      console.error(err);

      setSimilarModal({
        open: true,
        source: result,
        frames: [],
        loading: false,
      });

      pushToast("warning", "Similarity search failed", err.message);
    }
  }

  function pushToast(type, title, message = "") {
    const id = crypto.randomUUID();

    setToasts((prev) => [
      ...prev,
      {
        id,
        type,
        title,
        message,
      },
    ]);

    playNotifySound(type);

    setTimeout(() => {
      setToasts((prev) => prev.filter((toast) => toast.id !== id));
    }, 3200);
  }

  async function handleDresLogin() {
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
      pushToast("warning", "DRES login failed", err.message);
    }
  }

  function handleDresLogout() {
    setDres({
      loading: false,
      sessionId: "",
      user: null,
    });

    pushToast("pending", "DRES session cleared");
  }

  async function handleSubmitResult(result) {
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

      const label = `${result.video_id}/${String(result.frame_id).padStart(6, "0")}`;

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
      pushToast("warning", "Cannot submit", err.message);
    }
  }

  return (
    <div className={theme === "dark" ? "theme-dark" : "theme-light"}>
      <div className={`ambient-bg ${loading ? "ambient-searching" : ""}`} />

      <div className="app-root">
        <Sidebar
          theme={theme}
          onToggleTheme={() => setTheme(theme === "dark" ? "light" : "dark")}
          onReset={handleReset}
          onOpenSettings={() => setSettingsOpen(true)}
        />

        <main
          className={hasResults ? "main-layout has-results" : "main-layout is-home"}
        >
          {!hasResults && (
            <section className="home-panel">
              <div className="backend-status">
                <span
                  className={
                    backendReady ? "backend-dot connected" : "backend-dot"
                  }
                />
                {backendStatus}
              </div>

              <h1 className="main-title">Bạn muốn retrieval sự kiện nào?</h1>

              <SearchBar
                model={model}
                mode={mode}
                loading={loading}
                disabled={!backendReady}
                durationLimit={durationLimit}
                onModelChange={setModel}
                onModeChange={setMode}
                onDurationLimitChange={setDurationLimit}
                onSearch={handleSearch}
              />
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
                        onSimilaritySearch={handleSimilaritySearch}
                        onSurroundingImages={handleOpenSurroundingImages}
                      />
                    )}
                  </div>

                  {selected && (
                    <DetailPanel
                      result={selected}
                      onClose={() => setSelected(null)}
                      onSubmit={handleSubmitResult}
                    />
                  )}
                </div>
              </section>

              <div className={`bottom-search-zone ${loading ? "is-searching" : ""}`}>
                <SearchBar
                  model={model}
                  mode={mode}
                  loading={loading}
                  disabled={!backendReady}
                  durationLimit={durationLimit}
                  onModelChange={setModel}
                  onModeChange={setMode}
                  onDurationLimitChange={setDurationLimit}
                  onSearch={handleSearch}
                />

                <p className="footer-note">
                  Retrieval result có thể thiếu chính xác, cần kiểm tra lại bằng video gốc.
                </p>
              </div>
            </>
          )}

          {error && <p className="error-text">{error}</p>}
        </main>

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
          onClose={() =>
            setSurroundModal({
              open: false,
              center: null,
              frames: [],
              loading: false,
            })
          }
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
          onClose={() =>
            setSimilarModal({
              open: false,
              source: null,
              frames: [],
              loading: false,
            })
          }
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