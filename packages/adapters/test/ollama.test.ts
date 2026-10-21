import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { defaultSettings, type Settings } from "@rlm/contracts";
import type { ChatChunk } from "@rlm/core";
import { isOllama, OllamaChat, ollamaPlan, ProviderRouter } from "../src";

/** A fake Ollama: an 8B-like model (32 layers, 8 KV heads of 128) with a 128k native window and a reasoning mode. */
let server: Server;
let base = "";
const bodies: Array<Record<string, unknown>> = [];
let loaded = false;

beforeAll(async () => {
  server = createServer((req, res) => {
    let raw = "";
    req.on("data", (d) => (raw += d));
    req.on("end", () => {
      const json = (v: unknown) => {
        res.setHeader("Content-Type", "application/json");
        res.end(JSON.stringify(v));
      };
      if (req.url === "/api/version") return json({ version: "0.12.0" });
      if (req.url === "/api/show") {
        const model = JSON.parse(raw).model as string;
        return json({
          capabilities: model.startsWith("think") ? ["completion", "thinking"] : ["completion"],
          parameters: model.includes("ctx") ? "num_ctx 6000\nstop <|eot|>" : "stop <|eot|>",
          model_info: {
            "general.architecture": "llama",
            "llama.block_count": 32,
            "llama.context_length": 131072,
            "llama.attention.head_count": 32,
            "llama.attention.head_count_kv": 8,
            "llama.embedding_length": 4096,
          },
        });
      }
      if (req.url === "/api/tags") return json({ models: [{ name: "think-8b:latest", size: 5 * 2 ** 30 }, { name: "plain:latest", size: 5 * 2 ** 30 }] });
      if (req.url === "/api/ps") return json({ models: loaded ? [{ name: "plain:latest", size: 6 * 2 ** 30 }] : [] });
      if (req.url === "/api/chat") {
        bodies.push(JSON.parse(raw));
        res.setHeader("Content-Type", "application/x-ndjson");
        const lines = [
          { message: { role: "assistant", content: "", thinking: "Let me see." }, done: false },
          { message: { role: "assistant", content: "Hello" }, done: false },
          { message: { role: "assistant", content: " there." }, done: false },
          { message: { role: "assistant", content: "" }, done: true, prompt_eval_count: 42, eval_count: 3 },
        ];
        return res.end(lines.map((l) => JSON.stringify(l)).join("\n") + "\n");
      }
      res.statusCode = 404;
      res.end("not found");
    });
  });
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
  base = `http://127.0.0.1:${(server.address() as AddressInfo).port}/v1`;
});
afterAll(() => server.close());

const GB = 2 ** 30;
/** 32 layers x 8 heads x (128 + 128) x 2 bytes. */
const KV_PER_TOKEN = 32 * 8 * 256 * 2;

describe("Ollama", () => {
  it("is recognised by its version endpoint, other servers are not", async () => {
    expect(await isOllama(base, null)).toBe(true);
    expect(await isOllama("http://127.0.0.1:9/v1", null)).toBe(false);
  });

  it("sets the window from the cap when memory allows", async () => {
    const p = await ollamaPlan({ baseUrl: base, apiKey: null, model: "think-8b" }, { cap: 32768, kvRamShare: 0.35, configured: 131072, freeMem: () => 64 * GB });
    expect(p).toMatchObject({ numCtx: 32768, native: 131072, thinking: true });
  });

  it("shrinks the window to what the free memory can hold next to the weights", async () => {
    const p = await ollamaPlan({ baseUrl: base, apiKey: null, model: "plain" }, { cap: 32768, kvRamShare: 0.35, configured: null, freeMem: () => 8 * GB });
    const fit = Math.floor(((8 - 5) * GB * 0.35) / KV_PER_TOKEN);
    expect(p.numCtx).toBe(Math.floor(fit / 1024) * 1024);
    expect(p.numCtx).toBeLessThan(32768);
    expect(p.thinking).toBe(false);
    expect(p.why).toMatch(/MB free after the weights/);
  });

  it("does not count the weights twice when the model is already loaded", async () => {
    loaded = true;
    const p = await ollamaPlan({ baseUrl: base, apiKey: null, model: "plain" }, { cap: 32768, kvRamShare: 0.3, configured: null, freeMem: () => 4 * GB });
    loaded = false;
    expect(p.numCtx).toBe(Math.floor(Math.floor((4 * GB * 0.3) / KV_PER_TOKEN) / 1024) * 1024);
  });

  it("respects a num_ctx the user put in the Modelfile", async () => {
    const p = await ollamaPlan({ baseUrl: base, apiKey: null, model: "plain-ctx" }, { cap: 32768, kvRamShare: 0.35, configured: null, freeMem: () => 64 * GB });
    expect(p.numCtx).toBe(5120);
  });

  it("streams through the native API with num_ctx, and switches reasoning only on models that have it", async () => {
    const plan = { numCtx: 16384, native: 131072, thinking: true, why: "" };
    const chat = new OllamaChat("m", { baseUrl: base, apiKey: null, model: "think-8b" }, async () => plan);
    const out: ChatChunk[] = [];
    for await (const c of chat.stream({ messages: [{ role: "user", content: "hi" }], maxTokens: 300, enableThinking: false })) out.push(c);
    expect(out).toEqual([
      { type: "thinking", text: "Let me see." },
      { type: "text", text: "Hello" },
      { type: "text", text: " there." },
      { type: "usage", inputTokens: 42, outputTokens: 3 },
    ]);
    expect(bodies.at(-1)).toMatchObject({ model: "think-8b", stream: true, think: false, options: { num_ctx: 16384, num_predict: 300 } });

    const plain = new OllamaChat("p", { baseUrl: base, apiKey: null, model: "plain" }, async () => ({ ...plan, thinking: false }));
    for await (const _ of plain.stream({ messages: [{ role: "user", content: "hi" }], maxTokens: 10, enableThinking: false })) void _;
    expect(bodies.at(-1)).not.toHaveProperty("think");
  });

  it("makes the router size prompts for the window Ollama runs, not the native one", async () => {
    const settings: Settings = defaultSettings();
    settings.providers.push({ id: "ollama", kind: "openai_compat", label: "Ollama", baseUrl: base } as Settings["providers"][number]);
    settings.models.push({ id: "ollama:think", providerId: "ollama", name: "think-8b", label: "think", kind: "chat", contextLength: 131072, maxOutputTokens: null } as Settings["models"][number]);
    // The smallest window, so this machine's free memory cannot change the result.
    settings.local.ollamaContextCap = 2048;
    const router = new ProviderRouter({
      settings: () => settings,
      apiKey: () => null,
      localChat: () => {
        throw new Error("no local model");
      },
      localEmbed: async () => null,
      localNativeContext: () => null,
      continuePrompt: () => "continue",
    });
    expect(await router.limits("ollama:think")).toEqual({ contextLength: 2048, maxOutputTokens: 512 });
    expect(await router.chatFor("ollama:think")).toBeInstanceOf(OllamaChat);
  });
});
