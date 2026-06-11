import { useEffect, useState } from "react";
import Sidebar from "./components/Sidebar";
import SearchBar from "./components/SearchBar";
import ResultToolbar from "./components/ResultToolbar";
import ResultGrid from "./components/ResultGrid";
import GroupedResults from "./components/GroupedResults";
import DetailPanel from "./components/DetailPanel";
import SettingsPanel from "./components/SettingsPanel";
import { useRetrievalSearch } from "./hooks/useRetrievalSearch";
import { getBackendConfig, checkBackendHealth } from "./api/retrievalApi";

export default function App() {
  const [theme, setTheme] = useState("dark");
  const [model, setModel] = useState("ViT-B-16-quickgelu");
  const [mode, setMode] = useState("text");
  const [columns, setColumns] = useState(4);
  const [grouped, setGrouped] = useState(false);
  const [selected, setSelected] = useState(null);

  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settings, setSettings] = useState({
    useSplit: true,
    useTranslate: true,
    topK: 20,
    candidateMultiplier: 5,
    submitUrl: "",
    username: "",
    password: "",
  });

  const [backendReady, setBackendReady] = useState(false);
  const [backendStatus, setBackendStatus] = useState("Checking backend...");

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
    reset();
    setSelected(null);
    setGrouped(false);
  }

  function handleSearch(query) {
    setSelected(null);

    search({
      query,
      topK: settings.topK,
      candidateMultiplier: settings.candidateMultiplier,
      useSplit: settings.useSplit,
      useTranslate: settings.useTranslate,
    });
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

        <main className={hasResults ? "main-layout has-results" : "main-layout is-home"}
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

              <h1 className="main-title">
                Bạn muốn retrieval sự kiện nào?
              </h1>

              <SearchBar
                model={model}
                mode={mode}
                loading={loading}
                disabled={!backendReady}
                onModelChange={setModel}
                onModeChange={setMode}
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
                      />
                    ) : (
                      <ResultGrid
                        results={results}
                        columns={columns}
                        selectedId={selected?.id}
                        onSelect={setSelected}
                      />
                    )}
                  </div>

                  {selected && (
                    <DetailPanel
                      result={selected}
                      onClose={() => setSelected(null)}
                    />
                  )}
                </div>
              </section>

              <div className="bottom-search-zone">
                <SearchBar
                  model={model}
                  mode={mode}
                  loading={loading}
                  disabled={!backendReady}
                  onModelChange={setModel}
                  onModeChange={setMode}
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
          onChange={setSettings}
          onClose={() => setSettingsOpen(false)}
        />
      </div>
    </div>
  );
}