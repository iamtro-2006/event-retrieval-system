const NGROK_HEADER = { "ngrok-skip-browser-warning": "true" };

export function getGodModeEndpoint() {
  return String(import.meta.env.VITE_GODMODE_ENDPOINT || import.meta.env.VITE_SOCKET_URL || "").replace(/\/+$/, "");
}

export function godModeSocketUrl(endpoint, evaluationId) {
  const base = endpoint.replace(/^http:/, "ws:").replace(/^https:/, "wss:");
  return `${base}/ws?evaluation_id=${encodeURIComponent(evaluationId)}`;
}

export async function submitDresViaGodMode({ endpoint, dresUrl, sessionId, evaluationId, result }) {
  const response = await fetch(`${endpoint}/api/godmode/submit`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...NGROK_HEADER },
    body: JSON.stringify({
      dres_url: dresUrl,
      session_id: sessionId,
      evaluation_id: evaluationId,
      video_id: result.video_id,
      frame_id: Number(result.raw?.frame_idx ?? result.frame_id ?? 0),
      timestamp: Number(result.timestamp ?? 0),
      result,
    }),
  });
  if (!response.ok) throw new Error((await response.text()) || "God Mode submission failed");
  return response.json();
}
