const BASE = (import.meta.env?.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/+$/, "");
const NGROK_HEADER = { "ngrok-skip-browser-warning": "true" };

export async function getAgentStatus(signal) {
  const response = await fetch(`${BASE}/api/agent/status`, { signal, headers: NGROK_HEADER });
  if (!response.ok) throw new Error("Agent endpoint unavailable. Start the backend and check /api/agent/status.");
  return response.json();
}

// A POST stream permits structured inputs and cancellation without exposing auth.
export async function streamAgentChat(payload, { signal, onEvent }) {
  const response = await fetch(`${BASE}/api/agent/chat`, {
    method: "POST", headers: { "Content-Type": "application/json", ...NGROK_HEADER },
    body: JSON.stringify(payload), signal,
  });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    const detail = Array.isArray(data.detail)
      ? data.detail.map((item) => {
        const field = Array.isArray(item.loc) ? item.loc.filter((part) => part !== "body").join(".") : "request";
        return `${field}: ${item.msg || "Invalid value"}`;
      }).join("; ")
      : data.detail;
    throw new Error(typeof detail === "string" ? detail : `Chat request failed (${response.status})`);
  }
  if (!response.body) throw new Error("Streaming is not supported by this browser.");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let pending = "";
  let terminal = false;
  function consume(line) {
    if (!line.trim()) return;
    const event = JSON.parse(line);
    onEvent(event);
    if (event.type === "error") throw new Error(event.message);
    if (event.type === "done") terminal = true;
  }
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      pending += decoder.decode(value, { stream: true });
      const lines = pending.split("\n");
      pending = lines.pop();
      lines.forEach(consume);
    }
    pending += decoder.decode();
    consume(pending);
    if (!terminal) throw new Error("Chat connection ended early. Retry or start a new chat.");
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

export async function getSearchSession(id) {
  const response = await fetch(`${BASE}/api/agent/sessions/${encodeURIComponent(id)}`, { headers: NGROK_HEADER });
  if (!response.ok) throw new Error("Search session unavailable");
  return response.json();
}
export async function stopSearchSession(id) {
  return fetch(`${BASE}/api/agent/sessions/${encodeURIComponent(id)}/stop`, { method: "POST", headers: NGROK_HEADER });
}
