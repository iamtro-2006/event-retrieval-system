import test from "node:test";
import assert from "node:assert/strict";
import { streamAgentChat } from "../src/api/agentAPI.js";

function mockFetch(t, text, { status = 200, bytewise = false } = {}) {
  const previous = globalThis.fetch;
  t.after(() => { globalThis.fetch = previous; });
  const bytes = new TextEncoder().encode(text);
  globalThis.fetch = async () => new Response(new ReadableStream({
    start(controller) {
      if (bytewise) for (const value of bytes) controller.enqueue(new Uint8Array([value]));
      else controller.enqueue(bytes);
      controller.close();
    },
  }), { status });
}

test("chat protocol preserves Vietnamese UTF-8 across split network chunks", async (t) => {
  const expected = [{ type: "session", session_id: "one" }, { type: "delta", text: "Đã tìm ảnh" }, { type: "done" }];
  mockFetch(t, expected.map((event) => JSON.stringify(event)).join("\n"), { bytewise: true });
  const events = [];
  await streamAgentChat({ message: "test", dataset: "lhe" }, { onEvent: (event) => events.push(event) });
  assert.deepEqual(events, expected);
});

test("a disconnected stream is not reported as a completed chat", async (t) => {
  mockFetch(t, '{"type":"delta","text":"partial"}\n');
  await assert.rejects(streamAgentChat({}, { onEvent() {} }), /ended early/);
});

test("backend tool/auth failures are surfaced without a demo fallback", async (t) => {
  mockFetch(t, '{"type":"error","message":"Codex quota exceeded"}\n');
  await assert.rejects(streamAgentChat({}, { onEvent() {} }), /Codex quota exceeded/);
});

test("HTTP session conflicts preserve the actionable backend error", async (t) => {
  mockFetch(t, '{"detail":"Chat session expired. Start a new chat."}', { status: 409 });
  await assert.rejects(streamAgentChat({}, { onEvent() {} }), /session expired/);
});

test("Stop passes the AbortSignal through to fetch", async (t) => {
  const previous = globalThis.fetch;
  t.after(() => { globalThis.fetch = previous; });
  const controller = new AbortController();
  controller.abort();
  globalThis.fetch = async (_url, options) => {
    assert.equal(options.signal, controller.signal);
    options.signal.throwIfAborted();
  };
  await assert.rejects(streamAgentChat({}, { signal: controller.signal, onEvent() {} }), { name: "AbortError" });
});
