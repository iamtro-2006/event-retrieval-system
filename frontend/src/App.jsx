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
import { VideoFilterProvider, VideoFilterBar, useVideoFilter } from "./components/VideoFilter";
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
import PreviewVideoModal from "./components/PreviewVideoModal";
import SubmissionQueueModal from "./components/SubmissionQueueModal";
import ColorSearchModal from "./components/ColorSearch";
import { EMPTY_COLOR_GRID, selectedColorCells } from "./utils/colorGrid";
import { useRetrievalSearch } from "./hooks/useRetrievalSearch";
import {
  checkBackendHealth,
  getBackendConfig,
  getAvailableModels,
  getSurroundingFrames,
  similaritySearch,
  rerankResults,
  normalizeResults,
} from "./api/retrievalAPI";
import {
  getDefaultSubmissionSettings,
  loginDresViaBackend,
  submitDresViaBackend,
} from "./api/submissionAPI";
import { getGodModeEndpoint, godModeSocketUrl, submitDresViaGodMode } from "./api/godmodeAPI";
import { playNotifySound } from "./utils/notifySound";
import { ChevronDown, ChevronUp } from "lucide-react";
import { SEARCH_MODE_BY_CODE } from "./config/searchModes";

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

function getErrorMessage(error, fallback = "Unexpected error") {
  return error?.message || String(error || fallback);
}

function toSubmissionItem(result) {
  const frameId = Number(result?.raw?.frame_idx ?? result?.frame_idx ?? result?.frame_id ?? 0);
  const timestamp = Number(result?.timestamp ?? result?.timestamp_sec ?? 0);
  return {
    ...result,
    queue_id: `${result.video_id}:${frameId}:${Math.round(timestamp * 1000)}`,
    frame_id: frameId,
    timestamp,
    image_url: result.image_url || result.raw?.image_url || "",
  };
}

function parseExplicitQuery(query) {
  const text = String(query || "");
  if (!text.trim()) return { clauses: [""], connectors: [] };
  const pieces = text.split(/\b(AND|THEN)\b/);
  const clauses = [pieces[0].trim()];
  const connectors = [];
  for (let index = 1; index < pieces.length; index += 2) {
    const clause = String(pieces[index + 1] || "").trim();
    connectors.push(pieces[index]);
    clauses.push(clause);
  }
  return { clauses: clauses.length ? clauses : [""], connectors };
}

function composeQuery(clauses, connectors = []) {
  const normalized = clauses.map((clause) => String(clause || "").trim());
  let value = normalized[0] || "";
  for (let index = 1; index < normalized.length; index += 1) {
    value += ` ${connectors[index - 1] || "AND"} ${normalized[index]}`;
  }
  return value.trim();
}

function createImageAttachment(file) {
  return {
    id: crypto.randomUUID(),
    file,
    name: file.name || "Pasted image",
    url: URL.createObjectURL(file),
  };
}

export default function App() {
  return <VideoFilterProvider><RetrievalApp /></VideoFilterProvider>;
}

function RetrievalApp() {
  const { activeIds } = useVideoFilter();
  const defaultSubmission = useMemo(() => getDefaultSubmissionSettings(), []);
  const defaultGodModeEndpoint = useMemo(() => getGodModeEndpoint(), []);
  const searchIdRef = useRef(0);
  const toastTimersRef = useRef(new Map());
  const surroundReqRef = useRef(0);
  const similarReqRef = useRef(0);
  const rerankRunRef = useRef(0);
  const resultListRef = useRef(null);

  const [, startNonUrgentUpdate] = useTransition();

  // ── UI state ──────────────────────────────────────────
  const [theme, setTheme] = useState("dark");
  const [previewOpen, setPreviewOpen] = useState(false);
  const [colorModalOpen, setColorModalOpen] = useState(false);
  const [colorGrid, setColorGrid] = useState(() => [...EMPTY_COLOR_GRID]);
  const [model, setModel] = useState("siglip2-so400m");
  const [availableModels, setAvailableModels] = useState([]);
  const [semanticModelRoles, setSemanticModelRoles] = useState({ local: null, global: null });
  const [mode, setMode] = useState("text");
  const [query, setQuery] = useState("");
  const [queryClauses, setQueryClauses] = useState([""]);
  const [queryConnectors, setQueryConnectors] = useState([]);
  const [clauseImages, setClauseImages] = useState([[]]);
  const [sidebarExpanded, setSidebarExpanded] = useState(false);
  const [resultsHeaderCollapsed, setResultsHeaderCollapsed] = useState(false);
  const [durationLimit, setDurationLimit] = useState(-1);
  const [fusionConfig, setFusionConfig] = useState({
    semanticModels: [],
    temporal: false,
    durationLimit: -1,
    useOcr: false,
    useAsr: false,
    weights: { semantic: 0.8, ocr: 0.1, asr: 0.1 },
    semanticLambda: 0.5,
    hasConfig: false,
  });
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
  const [submissionQueue, setSubmissionQueue] = useState([]);
  const [submissionQueueOpen, setSubmissionQueueOpen] = useState(false);
  const [submissionPending, setSubmissionPending] = useState(false);

  // ── Settings ─────────────────────────────────────────
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settings, setSettings] = useState({
    useTranslate: true,
    topK: 20,
    candidateMultiplier: 5,
    submitUrl: defaultSubmission.dresUrl,
    evaluationId: defaultSubmission.evaluationId,
    username: defaultSubmission.teamId,
    password: defaultSubmission.teamPassword,
    godMode: false,
    godModeEndpoint: defaultGodModeEndpoint,
  });
  const [godModeResults, setGodModeResults] = useState([]);
  const [godModeConnected, setGodModeConnected] = useState(false);

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
  const [reasoningEnabled, setReasoningEnabled] = useState(false);
  const rerankEnabled = false;
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
    searchMultimodal,
    searchWithFusion,
    searchByColor,
    reset,
  } = useRetrievalSearch();

  const baseResults = rerankResultsData ?? rawResults;
  const results = useMemo(() => {
    if (!settings.godMode) return baseResults;
    const verifiedKeys = new Set(godModeResults.map((item) => `${item.video_id}:${item.frame_id}`));
    return [...godModeResults, ...baseResults.filter((item) => !verifiedKeys.has(`${item.video_id}:${item.frame_id}`))];
  }, [baseResults, godModeResults, settings.godMode]);
  const deferredResults = useDeferredValue(results);
  const hasResults = deferredResults.length > 0;
  const selectedId = selected?.id ?? null;

  useEffect(() => {
    if (!settings.godMode || !settings.godModeEndpoint || !settings.evaluationId) {
      return undefined;
    }
    let disposed = false;
    let retryTimer;
    let socket;
    const merge = (incoming) => setGodModeResults((previous) => {
      const key = (item) => `${item.evaluation_id}:${item.video_id}:${item.frame_id}`;
      const byId = new Map(previous.map((item) => [key(item), item]));
      normalizeResults(incoming).forEach((item) => byId.set(key(item), item));
      return [...byId.values()].sort((a, b) => Number(b.verified_at) - Number(a.verified_at));
    });
    const connect = () => {
      if (disposed) return;
      socket = new WebSocket(godModeSocketUrl(settings.godModeEndpoint, settings.evaluationId));
      socket.onopen = () => setGodModeConnected(true);
      socket.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          if (message.evaluation_id !== settings.evaluationId) return;
          // God Mode is scoped to the current search. Historical relay snapshots
          // belong to older queries and must not be restored after a refresh.
          if (message.type === "verified_result" && message.result) merge([message.result]);
        } catch { /* ignore malformed relay messages */ }
      };
      socket.onclose = () => {
        setGodModeConnected(false);
        if (!disposed) retryTimer = window.setTimeout(connect, 2000);
      };
    };
    connect();
    return () => {
      disposed = true;
      window.clearTimeout(retryTimer);
      socket?.close();
    };
  }, [settings.evaluationId, settings.godMode, settings.godModeEndpoint]);

  const resolvedMode = useMemo(() => {
    return mode === "temporal" ||
      mode === "auto" ||
      mode === "ocr" ||
      mode === "asr" ||
      mode === "fusion"
      || mode === "color"
      ? mode
      : "semantic";
  }, [mode]);

  const isHeavyDataset = deferredResults.length >= 120;

  const rootClassName = useMemo(() => {
    const themeClass = theme === "dark" ? "theme-dark" : "theme-light";
    return `${themeClass}${isHeavyDataset ? " performance-mode" : ""}`;
  }, [theme, isHeavyDataset]);

  const handleQueryChange = useCallback((nextQuery) => {
    const value = String(nextQuery ?? "");
    const { clauses: nextClauses, connectors: nextConnectors } = parseExplicitQuery(value);
    setQuery(value);
    setQueryClauses(nextClauses);
    setQueryConnectors(nextConnectors);
    setClauseImages((previous) => {
      previous.slice(nextClauses.length).flat().forEach((item) => URL.revokeObjectURL(item.url));
      return nextClauses.map((_, index) => previous[index] ?? []);
    });
  }, []);

  const handleModeChange = useCallback((nextMode) => {
    setMode(nextMode);
  }, []);

  useEffect(() => {
    const handleSearchModeShortcut = (event) => {
      if (!event.ctrlKey || event.altKey || event.metaKey || event.shiftKey || event.repeat) return;

      const nextMode = SEARCH_MODE_BY_CODE[event.code]?.key;
      if (!nextMode) return;

      event.preventDefault();
      handleModeChange(nextMode);
    };

    window.addEventListener("keydown", handleSearchModeShortcut);
    return () => window.removeEventListener("keydown", handleSearchModeShortcut);
  }, [handleModeChange]);

  const scrollResultsToTop = useCallback(() => {
    resultListRef.current?.scrollTo({ top: 0, behavior: "smooth" });
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, []);

  const handleInlineClauseChange = useCallback((clauseIndex, value) => {
    const pieces = String(value ?? "").split(/\b(AND|THEN)\b/);
    const typedConnectors = [];
    const typedClauses = [pieces[0]];
    for (let index = 1; index < pieces.length; index += 2) {
      typedConnectors.push(pieces[index]);
      typedClauses.push(pieces[index + 1] || "");
    }

    if (!typedConnectors.length) {
      const previousConnector = queryConnectors[clauseIndex - 1];
      const followsVisibleConnector = clauseIndex > 0 && !(
        previousConnector === "AND" && (clauseImages[clauseIndex - 1] || []).length > 0
      );
      const normalizedValue = followsVisibleConnector && !String(queryClauses[clauseIndex] || "").trim()
        ? String(value ?? "").trimStart()
        : value;
      const next = queryClauses.map((clause, index) => index === clauseIndex ? normalizedValue : clause);
      setQueryClauses(next);
      setQuery(composeQuery(next, queryConnectors));
      return;
    }

    // Spaces around a connector belong to layout, not either text fragment.
    const normalizedTypedClauses = typedClauses.map((clause, index) => {
      if (index === 0) return clause.trimEnd();
      if (index === typedClauses.length - 1) return clause.trimStart();
      return clause.trim();
    });
    const nextClauses = [
      ...queryClauses.slice(0, clauseIndex),
      ...normalizedTypedClauses,
      ...queryClauses.slice(clauseIndex + 1),
    ];
    const nextConnectors = [
      ...queryConnectors.slice(0, clauseIndex),
      ...typedConnectors,
      ...queryConnectors.slice(clauseIndex),
    ];
    const currentImages = clauseImages[clauseIndex] || [];
    const splitImages = normalizedTypedClauses.map((_, index) =>
      index === normalizedTypedClauses.length - 1 ? currentImages : []
    );
    const nextImages = [
      ...clauseImages.slice(0, clauseIndex),
      ...splitImages,
      ...clauseImages.slice(clauseIndex + 1),
    ];

    setQueryClauses(nextClauses);
    setQueryConnectors(nextConnectors);
    setClauseImages(nextImages);
    setQuery(composeQuery(nextClauses, nextConnectors));
  }, [clauseImages, queryClauses, queryConnectors]);

  const handleClausesChange = useCallback((nextClauses, options = {}) => {
    const normalized = nextClauses?.length ? nextClauses : [""];
    let nextConnectors = [...queryConnectors];
    if (Number.isInteger(options.removedIndex)) {
      const connectorIndex = options.removedIndex === 0 ? 0 : options.removedIndex - 1;
      nextConnectors.splice(connectorIndex, 1);
    } else if (normalized.length > queryClauses.length) {
      nextConnectors.push(options.connector === "THEN" ? "THEN" : "AND");
    }
    nextConnectors = nextConnectors.slice(0, Math.max(0, normalized.length - 1));
    setQueryClauses(normalized);
    setQueryConnectors(nextConnectors);
    setClauseImages((previous) => {
      if (Number.isInteger(options.removedIndex)) {
        previous[options.removedIndex]?.forEach((item) => URL.revokeObjectURL(item.url));
        const remaining = previous.filter((_, index) => index !== options.removedIndex);
        return normalized.map((_, index) => remaining[index] ?? []);
      }
      return normalized.map((_, index) => previous[index] ?? []);
    });
    setQuery(composeQuery(normalized, nextConnectors));
  }, [queryClauses.length, queryConnectors]);

  const handleAddClauseImages = useCallback((clauseIndex, files) => {
    const accepted = Array.from(files || []).filter((file) => file.type.startsWith("image/"));
    if (!accepted.length) return;
    setClauseImages((previous) => {
      const next = queryClauses.map((_, index) => [...(previous[index] ?? [])]);
      next[Math.max(0, Math.min(clauseIndex, next.length - 1))].push(...accepted.map(createImageAttachment));
      return next;
    });
  }, [queryClauses]);

  const handleInsertClauseImages = useCallback((clauseIndex, offset, files) => {
    const accepted = Array.from(files || []).filter((file) => file.type.startsWith("image/"));
    if (!accepted.length) return;

    const index = Math.max(0, Math.min(Number(clauseIndex) || 0, queryClauses.length - 1));
    const clause = String(queryClauses[index] || "");
    const splitAt = Math.max(0, Math.min(Number(offset) || 0, clause.length));
    const nextClauses = [
      ...queryClauses.slice(0, index),
      clause.slice(0, splitAt),
      clause.slice(splitAt),
      ...queryClauses.slice(index + 1),
    ];
    const nextConnectors = [
      ...queryConnectors.slice(0, index),
      "AND",
      ...queryConnectors.slice(index),
    ];
    const attachments = accepted.map(createImageAttachment);
    const nextImages = [
      ...clauseImages.slice(0, index),
      attachments,
      [...(clauseImages[index] || [])],
      ...clauseImages.slice(index + 1),
    ];

    setQueryClauses(nextClauses);
    setQueryConnectors(nextConnectors);
    setClauseImages(nextImages);
    setQuery(composeQuery(nextClauses, nextConnectors));
  }, [clauseImages, queryClauses, queryConnectors]);

  const handleRemoveClauseImage = useCallback((clauseIndex, imageId) => {
    setClauseImages((previous) => previous.map((items, index) => {
      if (index !== clauseIndex) return items;
      const removed = items.find((item) => item.id === imageId);
      if (removed) URL.revokeObjectURL(removed.url);
      return items.filter((item) => item.id !== imageId);
    }));
  }, []);

  const searchBarProps = useMemo(
    () => ({
      query,
      theme,
      model,
      mode,
      loading,
      disabled: !backendReady,
      durationLimit,
      reasoningEnabled,
      availableModels,
      semanticModelRoles,
      fusionConfig,
      queryClauses,
      queryConnectors,
      clauseImages,
      onModelChange: setModel,
      onModeChange: handleModeChange,
      onQueryChange: handleQueryChange,
      onInlineClauseChange: handleInlineClauseChange,
      onAddClauseImages: handleAddClauseImages,
      onInsertClauseImages: handleInsertClauseImages,
      onOpenColorSearch: () => setColorModalOpen(true),
      onRemoveClauseImage: handleRemoveClauseImage,
      onDurationLimitChange: setDurationLimit,
      onReasoningToggle: setReasoningEnabled,
      onFusionConfigChange: setFusionConfig,
    }),
    [query, theme, model, mode, loading, backendReady, durationLimit, reasoningEnabled, availableModels, semanticModelRoles, fusionConfig, queryClauses, queryConnectors, clauseImages, handleModeChange, handleQueryChange, handleInlineClauseChange, handleAddClauseImages, handleInsertClauseImages, handleRemoveClauseImage]
  );

  // ── Bootstrap ────────────────────────────────────────
  useEffect(() => {
    let alive = true;

    async function bootstrap() {
      try {
        await checkBackendHealth();
        const config = await getBackendConfig();

        if (!alive) return;
        setSemanticModelRoles(config.semantic_model_roles || { local: null, global: null });

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

        try {
          const models = await getAvailableModels();
          if (alive && Array.isArray(models) && models.length > 0) {
            setAvailableModels(models);
            setModel((prev) => (models.includes(prev) ? prev : models[0]));
          }
        } catch (modelsErr) {
          console.warn("[bootstrap] getAvailableModels failed:", modelsErr);
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
    const toastTimers = toastTimersRef.current;
    return () => {
      for (const timer of toastTimers.values()) {
        clearTimeout(timer);
      }
      toastTimers.clear();
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

  const handleColorSearch = useCallback(async () => {
    const cells = selectedColorCells(colorGrid);
    if (!cells.length || loading || !backendReady) return;
    scrollResultsToTop();
    setGodModeResults([]);
    setMode("color");
    setSelected(null);
    setRerankResultsData(null);
    try {
      await searchByColor({ cells, topK: settings.topK });
      setColorModalOpen(false);
    } catch (err) {
      pushToast("warning", "Color search failed", getErrorMessage(err));
    }
  }, [backendReady, colorGrid, loading, pushToast, scrollResultsToTop, searchByColor, settings.topK]);

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
    setGodModeResults([]);
    setSelected(null);
    setGrouped(false);
    setSurroundModal(DEFAULT_SURROUND_MODAL);
    setSimilarModal(DEFAULT_SIMILAR_MODAL);
    setModalOrder([]);
    setVideoResult(null);
    setRerankResultsData(null);
    setReranking(false);
    setQuery("");
    setQueryClauses([""]);
    setQueryConnectors([]);
    setClauseImages((previous) => {
      previous.flat().forEach((item) => URL.revokeObjectURL(item.url));
      return [[]];
    });
    setResultsHeaderCollapsed(false);
  }, [reset]);

  const handleSearch = useCallback(
    async (payload) => {
      const query = typeof payload === "string" ? payload : payload?.query;
      const cleanQuery = String(query || "").trim();

      const hasImages = clauseImages.some((items) => items.length > 0);
      if ((!cleanQuery && !hasImages) || loading || !backendReady) return;

      scrollResultsToTop();
      setGodModeResults([]);
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
        if (hasImages) {
          if (!["semantic", "temporal", "auto"].includes(searchMode)) {
            pushToast("warning", "Images are not supported in this mode", "Use Semantic, Temporal, or Auto for image queries. OCR, ASR, and Fusion support text queries only.");
            return;
          }
          await searchMultimodal({
            query: cleanQuery,
            clauses: queryClauses,
            queryConnectors,
            clauseImages,
            topK: settings.topK,
            candidateMultiplier: settings.candidateMultiplier,
            useSplit: true,
            useTranslate: settings.useTranslate,
            searchMode,
            modelKey: model,
            durationLimit: nextDurationLimit,
          });
        } else if (searchMode === "fusion") {
          const cfg = (typeof payload === "object" && payload?.fusionConfig) || fusionConfig;
          if (!cfg?.hasConfig) {
            pushToast("warning", "Fusion is not configured", "Open Fusion settings and select at least one model or method before searching.");
            return;
          }
          await searchWithFusion({
            query: cleanQuery,
            topK: settings.topK,
            candidateMultiplier: settings.candidateMultiplier,
            useSplit: true,
            useTranslate: settings.useTranslate,
             fusionConfig: cfg,
             reasoning: false,
          });
        } else {
          await search({
            query: cleanQuery,
            topK: settings.topK,
            candidateMultiplier: settings.candidateMultiplier,
            useSplit: true,
            useTranslate: settings.useTranslate,
            searchMode,
            modelKey: model,
            durationLimit: nextDurationLimit,
             reasoning: false,
          });
        }
      } catch (err) {
        if (err?.name === "AbortError" || err?.name === "StaleSearchError") {
          return;
        }
        if (searchId !== searchIdRef.current) return;
        console.error(err);
        pushToast("warning", "Search failed", getErrorMessage(err));
      }
    },
    [backendReady, clauseImages, durationLimit, fusionConfig, loading, model, pushToast, queryClauses, queryConnectors, resolvedMode, scrollResultsToTop, search, searchMultimodal, searchWithFusion, settings]
  );

  const handleSidebarSearch = useCallback(() => {
    handleSearch({
      query,
      searchMode: resolvedMode,
      durationLimit: resolvedMode === "temporal" ? Number(durationLimit) : -1,
      fusionConfig: resolvedMode === "fusion" ? fusionConfig : null,
      reasoning: false,
    });
  }, [durationLimit, fusionConfig, handleSearch, query, resolvedMode]);

  // Debounce automatic reranking and ignore stale updates.
  useEffect(() => {
    if (!rerankEnabled || mode === "fusion" || rawResults.length === 0 || loading || !lastQuery) return;

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
        pushToast("warning", "Automatic reranking failed", getErrorMessage(err));
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
    mode,
    settings.topK,
    pushToast,
    isHeavyDataset,
  ]);

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
          videoIds: activeIds,
          videoId: result.video_id,
          frameId: result.frame_id,
          topK: settings.topK,
          modelKey: result.model_key || result.raw?.model_key || model,
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
    [activeIds, model, pushToast, settings.topK]
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

  const handleQueueResult = useCallback((result) => {
    if (!result?.video_id) return;
    const item = toSubmissionItem(result);
    setSubmissionQueue((previous) => previous.some((queued) => queued.queue_id === item.queue_id) ? previous : [...previous, item]);
  }, []);

  const handleSubmitResult = useCallback(
    async ({ task, items, answer }) => {
      if (!items?.length) {
        pushToast("warning", "Cannot submit", "Submission queue is empty.");
        return;
      }

      try {
        setSubmissionPending(true);
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

        if (settings.godMode && !settings.godModeEndpoint) {
          throw new Error("The God Mode endpoint is not configured.");
        }
        if (settings.godMode && !evaluationId) {
          throw new Error("God Mode requires an active evaluation ID.");
        }
        const response = settings.godMode
          ? await submitDresViaGodMode({ endpoint: settings.godModeEndpoint, dresUrl: settings.submitUrl, sessionId, evaluationId, task, items, answer })
          : await submitDresViaBackend({ dresUrl: settings.submitUrl, sessionId, evaluationId, task, items, answer });

        const label = `${task.toUpperCase()} · ${items.length} frame(s)`;

        if (response.status === "correct") {
          if (response.verified_result) {
            const verified = normalizeResults([response.verified_result])[0];
            setGodModeResults((previous) => [verified, ...previous.filter((item) =>
              item.evaluation_id !== verified.evaluation_id || item.video_id !== verified.video_id || item.frame_id !== verified.frame_id
            )]);
          }
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
      } finally {
        setSubmissionPending(false);
      }
    },
    [dres.sessionId, pushToast, settings]
  );

  const handleResultAction = useCallback((result, action = "submit") => {
    if (!result?.video_id) return;
    const item = toSubmissionItem(result);
    if (action === "queue") {
      handleQueueResult(item);
      return;
    }
    void handleSubmitResult({ task: "kis", items: [item], answer: "" });
  }, [handleQueueResult, handleSubmitResult]);

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

      <div className={`app-root ${sidebarExpanded ? "sidebar-is-expanded" : ""}`}>
        <Sidebar
          theme={theme}
          mode={mode}
          queryClauses={queryClauses}
          queryConnectors={queryConnectors}
          clauseImages={clauseImages}
          expanded={sidebarExpanded}
          loading={loading}
          disabled={!backendReady}
          onToggleExpanded={() => setSidebarExpanded((value) => !value)}
          onModeChange={handleModeChange}
          onClausesChange={handleClausesChange}
          onAddClauseImages={handleAddClauseImages}
          onRemoveClauseImage={handleRemoveClauseImage}
          onSearch={handleSidebarSearch}
          onToggleTheme={handleToggleTheme}
          onReset={handleReset}
          onOpenSettings={handleOpenSettings}
          onOpenPreview={() => setPreviewOpen(true)}
          colorGrid={colorGrid}
          onColorGridChange={setColorGrid}
          onColorSearch={handleColorSearch}
        />

        <main className={hasResults ? "main-layout has-results" : "main-layout is-home"}>
          <VideoFilterBar />
          {!hasResults && (
            <section className="home-panel">
              <div className="backend-status">
                <span className={backendReady ? "backend-dot connected" : "backend-dot"} />
                {backendStatus}
              </div>

              <h1 className="main-title">Multimodal Retrieval System</h1>

              <SearchBar {...searchBarProps} expandedByDefault onSearch={handleSearch} />
            </section>
          )}

          {hasResults && (
            <>
              <section className="display-area">
                <div className={`results-header ${resultsHeaderCollapsed ? "is-collapsed" : ""}`}>
                  {!resultsHeaderCollapsed && <>
                    <ResultToolbar
                      model={model}
                      latency={latency}
                      columns={columns}
                      grouped={grouped}
                      onColumnsChange={handleColumnsChange}
                      onGroupedChange={handleGroupedChange}
                    />

                    <div className="query-summary">
                      <span>Query: <strong>{lastQuery}</strong></span>
                      <span className="query-summary-right">
                        {reranking && <span className="rerank-status-badge"><span className="rerank-spinner" /> VLM reranking...</span>}
                        {rerankResultsData && !reranking && <span className="rerank-status-badge rerank-status-badge--done">✓ Reranked</span>}
                        <span>{count} results</span>
                      </span>
                    </div>
                  </>}
                  <button type="button" className="results-header-toggle"
                    onClick={() => setResultsHeaderCollapsed((value) => !value)}
                    aria-label={resultsHeaderCollapsed ? "Show result toolbar" : "Hide result toolbar"}
                    title={resultsHeaderCollapsed ? "Show toolbar" : "Hide toolbar"}>
                    {resultsHeaderCollapsed ? <ChevronDown size={15} /> : <ChevronUp size={15} />}
                  </button>
                </div>

                <div className="result-body">
                  <div className="result-list" ref={resultListRef}>
                    {grouped ? (
                      <GroupedResults
                        results={deferredResults}
                        columns={columns}
                        selectedId={selectedId}
                        onSelect={handleSelectResult}
                        onSubmit={handleResultAction}
                        onPlay={handlePlayResult}
                        onSimilaritySearch={handleSimilaritySearch}
                        onSurroundingImages={handleOpenSurroundingImages}
                        query={lastQuery}
                      />
                    ) : (
                      <ResultGrid
                        results={deferredResults}
                        columns={columns}
                        selectedId={selectedId}
                        onSelect={handleSelectResult}
                        onSubmit={handleResultAction}
                        onPlay={handlePlayResult}
                        onSimilaritySearch={handleSimilaritySearch}
                        onSurroundingImages={handleOpenSurroundingImages}
                        query={lastQuery}
                      />
                    )}
                  </div>

                  {selected && (
                    <DetailPanel
                      key={selected.id}
                      result={selected}
                      onClose={handleCloseDetail}
                      onSubmit={handleResultAction}
                    />
                  )}
                </div>
              </section>

              <div className={`bottom-search-zone ${loading ? "is-searching" : ""}`}>
                <SearchBar {...searchBarProps} expandedByDefault={false} onSearch={handleSearch} />

                <p className="footer-note">
                  Retrieval results may be inaccurate. Verify them against the source video.
                </p>
              </div>
            </>
          )}

          {error && <p className="error-text">{error}</p>}
        </main>

        <VideoModal
          key={videoResult ? `${videoResult.id}-open` : "video-closed"}
          open={Boolean(videoResult)}
          result={videoResult}
          layer={modalLayer("video")}
          onClose={closeAllModals}
          onSubmit={handleResultAction}
        />

        <PreviewVideoModal open={previewOpen} onClose={() => setPreviewOpen(false)} onSubmit={handleResultAction} />

        <ColorSearchModal open={colorModalOpen} value={colorGrid} onChange={setColorGrid}
          loading={loading} onClose={() => setColorModalOpen(false)} onSearch={handleColorSearch} />

        <SettingsPanel
          open={settingsOpen}
          settings={settings}
          dres={dres}
          onChange={setSettings}
          onClose={handleCloseSettings}
          onDresLogin={handleDresLogin}
          onDresLogout={handleDresLogout}
          godModeConnected={godModeConnected}
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
          onSubmit={handleResultAction}
          onPlay={handlePlayResult}
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
          onSubmit={handleResultAction}
          onPlay={handlePlayResult}
          onSimilaritySearch={handleSimilaritySearch}
          onSurroundingImages={handleOpenSurroundingImages}
        />

        <button type="button" className="submission-queue-launcher" onClick={() => setSubmissionQueueOpen(true)}>
          Queue <span>{submissionQueue.length}</span>
        </button>
        <SubmissionQueueModal
          open={submissionQueueOpen}
          items={submissionQueue}
          submitting={submissionPending}
          onClose={() => setSubmissionQueueOpen(false)}
          onChange={setSubmissionQueue}
          onSubmit={handleSubmitResult}
        />

        <ToastHost toasts={toasts} />
      </div>
    </div>
  );
}
