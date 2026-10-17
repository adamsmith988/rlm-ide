import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { defaultSettings, type Settings } from "@rlm/contracts";
import { ProviderRouter, detectOpenAICompatLimits, redactKey } from "../src";

const KEY = "sk-test-0123456789abcdef";
let server: Server;
let base = "";
const seen: string[] = [];

function sse(res: import("node:http").ServerResponse, text: string): void {
  res.writeHead(200, { "Content-Type": "text/event-stream" });
  res.write(`data: ${JSON.stringify({ choices: [{ delta: { content: text } }] })}\n\n`);
  res.end("data: [DONE]\n\n");
}

beforeAll(async () => {
  server = createServer((req, res) => {
    const auth = req.headers.authorization ?? "";
    seen.push(`${req.method} ${req.url} ${auth}`);
    if (auth !== `Bearer ${KEY}`) {
      res.writeHead(401, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: { message: `invalid api key: ${auth}` } }));
      return;
    }
    if (req.url === "/v1/models") {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ data: [{ id: "tiny", max_model_len: 4096 }] }));
    } else if (req.url === "/v1/chat/completions") sse(res, "ready");
    else if (req.url === "/v1/embeddings") {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ data: [{ index: 0, embedding: [0.1, 0.2, 0.3] }] }));
    } else {
      res.writeHead(404);
      res.end();
    }
  });
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
  base = `http://127.0.0.1:${(server.address() as AddressInfo).port}/v1`;
});

afterAll(() => server.close());

function router(key: string | null): { r: ProviderRouter; s: Settings } {
  const s = defaultSettings();
  s.providers.push({ id: "p1", kind: "openai_compat", label: "Test", baseUrl: base, hasKey: !!key } as never);
  s.models.push({ id: "p1:tiny", providerId: "p1", name: "tiny", label: "tiny", kind: "chat", contextLength: null, maxOutputTokens: null, dim: null } as never);
  s.models.push({ id: "p1:emb", providerId: "p1", name: "emb", label: "emb", kind: "embed", contextLength: null, maxOutputTokens: null, dim: 3 } as never);
  const r = new ProviderRouter({
    settings: () => s,
    apiKey: (pid) => (pid === "p1" ? key : null),
    localChat: () => {
      throw new Error("no local");
    },
    localEmbed: async () => null,
    localNativeContext: () => null,
    continuePrompt: () => "continue",
  });
  return { r, s };
}

async function collect(r: ProviderRouter, id: string): Promise<string> {
  let out = "";
  for await (const c of (await r.chatFor(id)).stream({ messages: [{ role: "user", content: "hi" }], maxTokens: 8 })) if (c.type === "text") out += c.text;
  return out;
}

describe("provider API keys", () => {
  it("sends the provider key and works with the right one", async () => {
    const { r, s } = router(KEY);
    expect(await collect(r, "p1:tiny")).toBe("ready");
    const [v] = await (await r.embedderFor("p1:emb"))!.embed(["x"]);
    expect(v).toHaveLength(3);
    expect(await r.detect("p1:tiny")).toMatchObject({ contextLength: 4096 });
    expect(s.roles.agent).not.toBe("p1:tiny");
  });

  it("explains a rejected or missing key and never echoes it", async () => {
    const wrong = "sk-wrong-9999999999999999";
    const { r } = router(wrong);
    const err = await collect(r, "p1:tiny").catch((e: Error) => e.message);
    expect(err).toMatch(/401.*rejected the API key/);
    expect(err).not.toContain(wrong);
    await expect(r.detect("p1:tiny")).rejects.toThrow(/rejected the API key/);
    await expect((await r.embedderFor("p1:emb"))!.embed(["x"])).rejects.toThrow(/401/);

    const none = router(null).r;
    await expect(collect(none, "p1:tiny")).rejects.toThrow(/requires an API key and none is set/);
  });

  it("reports unreachable servers with the address", async () => {
    const { r, s } = router(KEY);
    s.providers.find((p) => p.id === "p1")!.baseUrl = "http://127.0.0.1:9/v1";
    r.invalidate();
    await expect(collect(r, "p1:tiny")).rejects.toThrow(/cannot reach http:\/\/127\.0\.0\.1:9\/v1\/chat\/completions/);
  });

  it("rejects using a model for the wrong kind", async () => {
    const { r } = router(KEY);
    await expect(r.chatFor("p1:emb")).rejects.toThrow(/embedding model/);
    await expect(r.embedderFor("p1:tiny")).rejects.toThrow(/chat model/);
  });

  it("fails a stalled stream with a clear error instead of hanging", async () => {
    const stuck = createServer((_req, res) => {
      res.writeHead(200, { "Content-Type": "text/event-stream" });
      res.write(`data: ${JSON.stringify({ choices: [{ delta: { content: "par" } }] })}\n\n`);
    });
    await new Promise<void>((r) => stuck.listen(0, "127.0.0.1", r));
    const { r, s } = router(KEY);
    s.providers.find((p) => p.id === "p1")!.baseUrl = `http://127.0.0.1:${(stuck.address() as AddressInfo).port}/v1`;
    s.rlm.modelStallSec = 1;
    r.invalidate();
    const t0 = Date.now();
    const parts: string[] = [];
    const err = await (async () => {
      for await (const c of (await r.chatFor("p1:tiny")).stream({ messages: [{ role: "user", content: "hi" }], maxTokens: 8 })) if (c.type === "text") parts.push(c.text);
    })().catch((e: Error) => e.message);
    stuck.closeAllConnections();
    stuck.close();
    expect(parts.join("")).toBe("par");
    expect(err).toMatch(/sent nothing for 1 s and looks stuck/);
    expect(Date.now() - t0).toBeLessThan(5000);
  });

  it("a user stop is not reported as a stall", async () => {
    const { r } = router(KEY);
    const ctl = new AbortController();
    ctl.abort();
    const err = await (async () => {
      for await (const _ of (await r.chatFor("p1:tiny")).stream({ messages: [{ role: "user", content: "hi" }], maxTokens: 8, signal: ctl.signal })) void _;
    })().catch((e: Error) => e);
    expect(String((err as Error)?.message ?? err)).not.toMatch(/stuck/);
  });

  it("redacts keys and bearer tokens", () => {
    expect(redactKey(`bad key ${KEY} Authorization: Bearer abcdefgh12345`, KEY)).toBe("bad key [redacted] Authorization: Bearer [redacted]");
    expect(detectOpenAICompatLimits).toBeTypeOf("function");
  });
});
