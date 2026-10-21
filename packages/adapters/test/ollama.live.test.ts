import { afterAll, describe, expect, it } from "vitest";
import type { ChatChunk, ChatRequest } from "@rlm/core";
import { OllamaChat, ollamaPlan, OpenAICompatChat, type OpenAICompatConfig } from "../src";

/** Runs against a real Ollama: RLM_LIVE_OLLAMA=http://127.0.0.1:11434/v1 [RLM_OLLAMA_SMALL=qwen2.5:0.5b] [RLM_OLLAMA_THINK=qwen3.5:9b] */
const base = process.env.RLM_LIVE_OLLAMA ?? "";
const small = process.env.RLM_OLLAMA_SMALL ?? "qwen2.5:0.5b";
const thinker = process.env.RLM_OLLAMA_THINK ?? "qwen3.5:9b";
const root = base.replace(/\/v1\/?$/, "");
const cfg = (model: string): OpenAICompatConfig => ({ baseUrl: base, apiKey: null, model, stallMs: 300_000 });

async function collect(port: { stream(r: ChatRequest): AsyncIterable<ChatChunk> }, req: ChatRequest) {
  const out = { text: "", thinking: "", input: 0 };
  for await (const c of port.stream(req)) {
    if (c.type === "text") out.text += c.text;
    if (c.type === "thinking") out.thinking += c.text;
    if (c.type === "usage") out.input = c.inputTokens;
  }
  return out;
}

const loadedCtx = async (model: string) => {
  const j = (await (await fetch(`${root}/api/ps`)).json()) as { models: Array<{ name: string; context_length?: number }> };
  return j.models.find((m) => m.name === model || m.name === `${model}:latest`)?.context_length ?? null;
};

describe.skipIf(!base)("Ollama, live", () => {
  afterAll(async () => {
    for (const model of [small, thinker]) await fetch(`${root}/api/generate`, { method: "POST", body: JSON.stringify({ model, keep_alive: 0 }) }).catch(() => undefined);
  });

  it("runs the model with the window the IDE chose", async () => {
    const plan = await ollamaPlan(cfg(small), { cap: 16384, kvRamShare: 0.35, configured: null });
    const r = await collect(new OllamaChat("s", cfg(small), async () => plan), { messages: [{ role: "user", content: "Reply with just the word: ok" }], maxTokens: 20, temperature: 0 });
    expect(r.text.toLowerCase()).toContain("ok");
    expect(r.input).toBeGreaterThan(0);
    expect(await loadedCtx(small)).toBe(plan.numCtx);
  }, 120_000);

  it("keeps the start of a long prompt that the OpenAI-compatible endpoint drops", async () => {
    const plan = await ollamaPlan(cfg(small), { cap: 16384, kvRamShare: 0.35, configured: null });
    const filler = Array.from({ length: 700 }, (_, i) => `Note ${i}: the build server rotated its logs and nothing else happened.`).join("\n");
    const messages: ChatRequest["messages"] = [
      { role: "system", content: "The secret code is PURPLE-HORSE-42. Answer questions briefly." },
      { role: "user", content: `${filler}\n\nWhat is the secret code? Reply with the code only.` },
    ];
    const native = await collect(new OllamaChat("s", cfg(small), async () => plan), { messages, maxTokens: 30, temperature: 0 });
    expect(native.input).toBeGreaterThan(8000);
    expect(native.text).toContain("PURPLE-HORSE-42");
    const compat = await collect(new OpenAICompatChat("s", cfg(small)), { messages, maxTokens: 30, temperature: 0 });
    console.log(`OpenAI-compatible endpoint saw ${compat.input} prompt tokens (native: ${native.input}) and answered: ${compat.text.trim().slice(0, 80)}`);
    expect(compat.input).toBeLessThan(native.input);
  }, 300_000);

  it("switches reasoning off and on for a reasoning model", async () => {
    await fetch(`${root}/api/generate`, { method: "POST", body: JSON.stringify({ model: small, keep_alive: 0 }) });
    const plan = await ollamaPlan(cfg(thinker), { cap: 8192, kvRamShare: 0.35, configured: null });
    expect(plan.thinking).toBe(true);
    const chat = new OllamaChat("t", cfg(thinker), async () => plan);
    const off = await collect(chat, { messages: [{ role: "user", content: "What is 17 + 25? Reply with the number." }], maxTokens: 400, temperature: 0, enableThinking: false });
    expect(off.thinking).toBe("");
    expect(off.text).toContain("42");
    const on = await collect(chat, { messages: [{ role: "user", content: "What is 17 + 25? Reply with the number." }], maxTokens: 2000, temperature: 0, enableThinking: true });
    expect(on.thinking.length).toBeGreaterThan(0);
  }, 600_000);
});
