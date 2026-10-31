import { useEffect, useImperativeHandle, useRef, useState } from "react";
import { Bot, Check, ChevronRight, LoaderCircle, Plus, Users, Wrench, X } from "lucide-react";
import { getAgentStatus, streamAgentChat, getSearchSession, stopSearchSession } from "../api/agentAPI";
import AssistantMarkdown from "./AssistantMarkdown";
import "./AgentChat.css";

async function readImages(files) {
  if (files.length > 6) throw new Error("Attach at most 6 images per message.");
  return Promise.all(files.map((file) => {
    if (!["image/jpeg", "image/png", "image/webp", "image/gif"].includes(file.type)) throw new Error("Use JPEG, PNG, WebP or GIF for assistant image inputs.");
    if (file.size > 2 * 1024 * 1024) throw new Error("Each assistant image must be at most 2 MB.");
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject(new Error("Cannot read attached image."));
      reader.readAsDataURL(file);
    });
  }));
}

// Transport/history live here; SearchBar is the sole input/send/stop surface.
export default function AgentChat({ ref, dataset, topK, videoIds, enabled, provider, model, workers, reasoningEffort,
  open, onOpenChange, onConnection, onCompose, onAccepted, onResults, onBusyChange, onNewChat }) {
  const [messages, setMessages] = useState([]);
  const [busy, setBusy] = useState(false);
  const [phase, setPhase] = useState("");
  const [error, setError] = useState("");
  const session = useRef(null);
  const [plans, setPlans] = useState([]);
  const selectedPool = useRef(null);
  const [poolSelection, setPoolSelection] = useState(null);
  const [restoringActive, setRestoringActive] = useState(false);
  const filterKey = JSON.stringify(videoIds || []);
  const active = useRef(null);
  const viewport = useRef(null);
  const follow = useRef(true);
  const refreshController = useRef(null);
  const mounted = useRef(true);
  const scrollFrame = useRef(null);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      active.current?.abort();
      refreshController.current?.abort();
      cancelAnimationFrame(scrollFrame.current);
      onConnection({ checking: false });
      onBusyChange(false);
    };
  }, [onConnection, onBusyChange]);

  useEffect(() => {
    if (!follow.current) return;
    scrollFrame.current = requestAnimationFrame(() => {
      if (viewport.current) viewport.current.scrollTop = viewport.current.scrollHeight;
    });
    return () => cancelAnimationFrame(scrollFrame.current);
  }, [messages, phase, open]);

  async function refreshStatus() {
    refreshController.current?.abort();
    const controller = new AbortController();
    refreshController.current = controller;
    onConnection({ checking: true });
    try {
      const next = await getAgentStatus(controller.signal);
      if (!controller.signal.aborted) onConnection({ ...next.codex, checking: false });
    } catch (err) {
      if (err.name !== "AbortError") onConnection({ available: false, message: err.message, checking: false });
    }
  }

  useEffect(() => {
    if (provider !== "codex") return;
    let cancelled = false;
    session.current = null; selectedPool.current = null;
    const id = localStorage.getItem(`ers-search-${dataset}`);
    Promise.resolve(id ? getSearchSession(id) : null).catch(() => null).then((saved) => {
      if (cancelled) return;
      setPoolSelection(null);
      if (!saved || JSON.stringify(saved.video_ids || []) !== filterKey) {
        setPlans([]); setRestoringActive(false); return;
      }
      session.current = id; setPlans(saved.plans || []); setRestoringActive(Boolean(saved.running));
    });
    return () => { cancelled = true; };
  }, [dataset, provider, filterKey]);

  useEffect(() => {
    if (busy || !session.current || (!restoringActive && !plans.some((p) => ["queued", "running"].includes(p.status)))) return;
    let cancelled = false;
    const timer = setInterval(() => getSearchSession(session.current).then((saved) => {
      if (!cancelled) { setPlans(saved.plans || []); setRestoringActive(Boolean(saved.running)); }
    }).catch(() => {}), 1500);
    return () => { cancelled = true; clearInterval(timer); };
  }, [busy, plans, restoringActive]);

  function clearChat() {
    if (session.current) stopSearchSession(session.current).catch(() => {});
    localStorage.removeItem(`ers-search-${dataset}`);
    active.current?.abort(); active.current = null; session.current = null; selectedPool.current = null; setPoolSelection(null); setPlans([]);
    setMessages([]); setError(""); setPhase(""); setBusy(false);
    follow.current = true;
    onBusyChange(false); onNewChat();
  }

  async function send(text, { files = [], frameIds = [], searchMode = "semantic", colorCells = [] } = {}) {
    const message = text.trim() || (files.length ? "Phân tích ảnh đính kèm và tìm các frame liên quan." : "");
    if (!message || active.current || !enabled) return;
    const controller = new AbortController();
    active.current = controller;
    const id = crypto.randomUUID();
    follow.current = true;
    onOpenChange(true);
    setBusy(true); setError(""); setPhase("Connecting…"); onBusyChange(true);
    function updateAssistant(update) {
      setMessages((items) => items.map((item) => item.id === id ? update(item) : item));
    }
    try {
      const images = await readImages(files);
      if (controller.signal.aborted) return;
      setMessages((items) => [...items.slice(-30), { id: `${id}-user`, role: "user", text: message,
        imageCount: images.length, frameCount: frameIds.length },
        { id, role: "assistant", text: "", tools: [], workers: [] }]);
      onAccepted();
      await streamAgentChat({ message, dataset, provider, model: provider === "codex" ? model || null : null,
        session_id: session.current, parent_pool_id: selectedPool.current, top_k: topK, top_k_limit: topK, video_ids: videoIds,
        max_workers: workers, reasoning_effort: reasoningEffort || null,
        images, context_frame_ids: frameIds, search_mode: searchMode, color_cells: colorCells }, {
        signal: controller.signal,
        onEvent(event) {
          if (active.current !== controller || controller.signal.aborted || !mounted.current) return;
          if (event.type === "session") {
            session.current = event.session_id;
            localStorage.setItem(`ers-search-${dataset}`, event.session_id);
          }
          else if (event.type === "plans") setPlans((old) => [...old, ...event.plans]);
          else if (event.type === "pool") {
            setPlans((old) => old.map((p) => p.id === event.id ? event : p));
            if (!selectedPool.current && event.status === "completed") {
              selectedPool.current = event.id;
              setPoolSelection(event.id);
              onResults(event.results, event.plan?.query || message, event.plan?.mode);
            }
          }
          else if (event.type === "status") setPhase(event.message);
          else if (event.type === "delta") updateAssistant((item) => ({ ...item, text: item.text + event.text }));
          else if (event.type === "tool_start") {
            setPhase(`Calling ${event.name}…`);
            updateAssistant((item) => ({ ...item, tools: [...item.tools, { name: event.name, args: event.arguments, running: true }] }));
          } else if (event.type === "tool_end") {
            setPhase("Reading tool result…");
            updateAssistant((item) => {
              const index = item.tools.findIndex((tool) => tool.name === event.name && tool.running);
              return { ...item, tools: item.tools.map((tool, i) => i === index ? { ...tool, running: false,
                success: event.success, count: event.count, error: event.error } : tool) };
            });
          } else if (event.type === "worker_start") {
            setPhase("Agents working in parallel…");
            updateAssistant((item) => ({ ...item, workers: [...item.workers, { ...event, running: true, text: "" }] }));
          } else if (event.type === "worker_delta" || event.type === "worker_end") {
            updateAssistant((item) => ({ ...item, workers: item.workers.map((worker) => worker.worker_id === event.worker_id
              ? { ...worker, text: worker.text + (event.text || ""), ...(event.type === "worker_end"
                ? { running: false, success: event.success, error: event.error } : {}) } : worker) }));
          } else if (event.type === "results") {
            onResults(event.results, event.query, event.plan?.mode);
            updateAssistant((item) => ({ ...item, results: event.results }));
          } else if (event.type === "done") setPhase("Completed");
        },
      });
    } catch (err) {
      if (active.current !== controller || !mounted.current) return;
      const stopped = err.name === "AbortError";
      if (!stopped) setError(err.message);
      setPhase(stopped ? "Stopped" : "Failed");
      updateAssistant((item) => ({ ...item, text: item.text || (stopped ? "Đã dừng lượt chat." : "Lượt chat chưa hoàn tất."),
        tools: item.tools.map((tool) => tool.running ? { ...tool, running: false, stopped } : tool),
        workers: item.workers.map((worker) => worker.running ? { ...worker, running: false, stopped } : worker) }));
    } finally {
      if (active.current === controller && mounted.current) {
        active.current = null; setBusy(false); onBusyChange(false);
      }
    }
  }

  useImperativeHandle(ref, () => ({ send, stop: () => { if (session.current) stopSearchSession(session.current).catch(() => {}); active.current?.abort(); setPhase("Stopping after active inference…"); }, refresh: refreshStatus }));

  return <>
    {open && <section className="agent-panel" aria-label="AI search assistant">
      <header className="agent-header"><div><Bot size={20} /><div><strong>Assistant activity</strong><small>{dataset.toUpperCase()} · {provider === "demo" ? "scripted demo" : "Codex"} · sequential plans</small></div></div>
        <div><button type="button" title="New chat" aria-label="New chat" onClick={clearChat}><Plus size={17} /></button>
          <button type="button" aria-label="Close AI chat" onClick={() => onOpenChange(false)}><X size={17} /></button></div>
      </header>
      <div className="agent-notice">Dùng thanh tìm kiếm chính để nhập và gửi. Bật Reasoning để agent gọi tool. Ảnh tìm kiếm hiện là mock.</div>
      <div className="agent-messages" ref={viewport} onScroll={() => {
        const node = viewport.current; follow.current = node.scrollHeight - node.scrollTop - node.clientHeight < 80;
      }}>
        {!messages.length && <div className="agent-empty"><div className="agent-empty-icon"><Bot size={29} /></div>
          <h2>Bắt đầu từ thanh search</h2><p>Chọn gợi ý để điền vào thanh chính, sau đó nhấn gửi.</p>
          {["Tìm một cảnh ngoài trời và xem thử các ảnh", "Thử các chiến thuật caption và ASR rồi cho tôi chọn pool", "Tìm hình ảnh trước rồi xếp hạng theo bối cảnh"].map((text) =>
            <button key={text} type="button" onClick={() => onCompose(text)}>{text}<ChevronRight size={14} /></button>)}
        </div>}
        {plans.length > 0 && <div className="agent-tools">
          <strong>Search plans / pools</strong>
          {plans.map((plan) => <details key={plan.id} open={poolSelection === plan.id}>
            <summary>{plan.plan?.label || plan.id.slice(0, 8)} · {plan.status}</summary>
            <pre>{JSON.stringify(plan.plan, null, 2)}</pre>
            {plan.error && <p role="alert">{plan.error}</p>}
            {plan.status === "completed" && <>
              <button type="button" onClick={() => {
                selectedPool.current = plan.id; setPoolSelection(plan.id);
                onResults(plan.results || [], plan.plan?.query, plan.plan?.mode);
              }}>Xem pool ({plan.results?.length || 0})</button>
              <button type="button" onClick={() => {
                selectedPool.current = plan.id; setPoolSelection(plan.id);
                onCompose(`Tinh chỉnh pool ${plan.plan?.label || plan.id}: `);
              }}>Feedback / tạo nhánh</button>
            </>}
          </details>)}
        </div>}
        {messages.map((item) => <article key={item.id} className={`agent-message is-${item.role}`}>
          <span className="agent-message-label">{item.role === "user" ? "Bạn" : provider === "demo" ? "Demo assistant" : "Codex"}</span>
          {(item.imageCount > 0 || item.frameCount > 0) && <small className="agent-message-label">{item.imageCount || 0} images · {item.frameCount || 0} frame references</small>}
          {item.tools?.length > 0 && <div className="agent-tools">{item.tools.map((tool, index) => <details key={`${index}-${tool.name}`}>
            <summary>{tool.running ? <LoaderCircle className="agent-spin" size={13} /> : tool.success ? <Check size={13} /> : <Wrench size={13} />}
              <code>{tool.name}</code><span>{tool.running ? "running" : tool.stopped ? "stopped" : tool.success ? tool.count == null ? "done" : `${tool.count} items` : "failed"}</span></summary>
            <pre>{JSON.stringify(tool.args, null, 2)}</pre>{tool.error && <p>{tool.error}</p>}
          </details>)}</div>}
          {item.workers?.length > 0 && <div className="agent-workers">{item.workers.map((worker) => <details key={worker.worker_id}>
            <summary>{worker.running ? <LoaderCircle size={13} className="agent-spin" /> : <Users size={13} />}
              <span>{worker.label}</span><small>{worker.running ? "running" : worker.stopped ? "stopped" : worker.success ? "done" : "failed"}</small></summary>
            <AssistantMarkdown>{worker.text || worker.error || "Waiting for report…"}</AssistantMarkdown>
          </details>)}</div>}
          {item.text && (item.role === "user"
            ? <p className="agent-message-text">{item.text}</p>
            : <AssistantMarkdown>{item.text}</AssistantMarkdown>)}
          {item.results && <button type="button" className="agent-show-results" disabled={busy}
            onClick={() => onCompose("Hiển thị lại nhóm ảnh này và giải thích lựa chọn.", { frameIds: item.results.map((row) => row.id) })}>
            Đưa {item.results.length} ảnh vào thanh chính ↗</button>}
        </article>)}
      </div>
      <footer className="agent-log-footer"><div className="agent-phase" role="status">{busy && <LoaderCircle size={12} className="agent-spin" />}{phase || "Ready"}</div>
        {error && <p className="agent-error" role="alert">{error}</p>}
      </footer>
    </section>}
  </>;
}
