const RAW_API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

const API_BASE_URL = RAW_API_BASE_URL.replace(/\/+$/, "");
const NGROK_HEADER = { "ngrok-skip-browser-warning": "true" };

function apiUrl(path) {
  return `${API_BASE_URL}/${String(path).replace(/^\/+/, "")}`;
}

export function getDefaultSubmissionSettings() {
  return {
    dresUrl: import.meta.env.VITE_SUBMISSION || "",
    teamId: import.meta.env.VITE_TEAM_ID || "",
    teamPassword: import.meta.env.VITE_TEAM_PASSWORD || "",
    evaluationId: import.meta.env.VITE_DRES_EVALUATION_ID || "",
  };
}

export async function loginDresViaBackend({ dresUrl, username, password }) {
  const response = await fetch(apiUrl("/api/dres/login"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...NGROK_HEADER,
    },
    body: JSON.stringify({
      dres_url: dresUrl,
      username,
      password,
    }),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(extractErrorMessage(text, "DRES login failed"));
  }

  return response.json();
}

export async function submitDresViaBackend({
  dresUrl,
  sessionId,
  evaluationId,
  result,
}) {
  const response = await fetch(apiUrl("/api/dres/submit"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...NGROK_HEADER,
    },
    body: JSON.stringify({
      dres_url: dresUrl,
      session_id: sessionId,
      evaluation_id: evaluationId || null,
      video_id: result.video_id,
      frame_id: Number(result.raw?.frame_idx ?? result.frame_id ?? 0),
      timestamp: Number(result.timestamp ?? 0),
    }),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(extractErrorMessage(text, "DRES submit failed"));
  }

  return response.json();
}

function extractErrorMessage(text, fallback) {
  try {
    const data = JSON.parse(text);
    return data.detail || data.message || data.error || text || fallback;
  } catch {
    return text || fallback;
  }
}
