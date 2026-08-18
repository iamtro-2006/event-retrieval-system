const GOOGLE_TRANSLATE_KEY = import.meta.env.VITE_GOOGLE_TRANSLATE || "";
const LLM_TRANSLATE_KEY = import.meta.env.VITE_LLM_TRANSLATE_API_KEY || "";
const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/+$/, "");

export async function translateViaBackend(text, { provider = "google", source = "vi", target = "en" } = {}) {
  if (!text || !text.trim() || source === target) return text;
  const apiKey = provider === "llm" ? LLM_TRANSLATE_KEY : provider === "google" ? GOOGLE_TRANSLATE_KEY : "";
  if ((provider === "google" || provider === "llm") && !apiKey) {
    throw new Error(`VITE_${provider === "google" ? "GOOGLE_TRANSLATE" : "LLM_TRANSLATE_API_KEY"} is not configured`);
  }

  const response = await fetch(`${API_BASE_URL}/api/translate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, source, target, provider, api_key: apiKey || null }),
    });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data?.error?.message || "Google Cloud Translation failed");
  }
  return data.translated_text;
}

export const translateWithGoogle = (text, options = {}) =>
  translateViaBackend(text, { ...options, provider: "google" });

export async function translateBatchWithGoogle(texts, options = {}) {
  return Promise.all(texts.map((text) => translateWithGoogle(text, options)));
}
