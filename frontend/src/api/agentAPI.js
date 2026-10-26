const BASE = (import.meta.env?.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/+$/, "");

export async function getAgentStatus(signal) {
  const response = await fetch(`${BASE}/api/agent/status`, { signal });
  if (!response.ok) throw new Error("Agent endpoint unavailable. Start the mock server on port 8001 and use npm run dev:mock.");
  return response.json();
}

// A POST stream permits structured inputs and cancellation without exposing auth.
export async function streamAgentChat(payload, { signal, onEvent }) {
  const response = await fetch(`${BASE}/api/agent/chat`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload), signal,
  });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(typeof data.detail === "string" ? data.detail : `Chat request failed (${response.status})`);
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
