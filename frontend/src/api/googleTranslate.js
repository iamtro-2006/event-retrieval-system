const GOOGLE_TRANSLATE_KEY = import.meta.env.VITE_GOOGLE_TRANSLATE || "";
const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/+$/, "");

export async function translateViaBackend(text, { source = "vi", target = "en" } = {}) {
  if (!text || !text.trim() || source === target) return text;
  if (!GOOGLE_TRANSLATE_KEY) {
    throw new Error("VITE_GOOGLE_TRANSLATE is not configured");
  }

  const response = await fetch(`${API_BASE_URL}/api/translate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, source, target, api_key: GOOGLE_TRANSLATE_KEY }),
    });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data?.error?.message || "Google Cloud Translation failed");
  }
  return data.translated_text;
}

export const translateWithGoogle = (text, options = {}) =>
  translateViaBackend(text, options);

export async function translateBatchWithGoogle(texts, options = {}) {
  return Promise.all(texts.map((text) => translateWithGoogle(text, options)));
}
