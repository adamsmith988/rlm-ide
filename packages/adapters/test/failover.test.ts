import { createServer, type Server, type ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";
import { afterEach, describe, expect, it } from "vitest";
import { ContextOverflowError, ModelWindowError, type ChatChunk, type ChatModelPort, type ChatRequest } from "@rlm/core";
import { defaultSettings, type FailoverSettings } from "@rlm/contracts";
import { ContinuationDeduper, FailoverChat, ModelHealth, ModelHttpError, ProviderRouter, classifyFailure, resumePoint, seamSpace, type FailoverCandidate } from "../src";

type Script = (req: ChatRequest) => AsyncIterable<ChatChunk>;

class FakeModel implements ChatModelPort {
  calls: ChatRequest[] = [];
  constructor(
    readonly modelId: string,
    private readonly script: Script,
  ) {}
  stream(req: ChatRequest): AsyncIterable<ChatChunk> {
    this.calls.push(req);
    return this.script(req);
  }
}

function says(...parts: string[]): Script {
  return async function* () {
    for (const p of parts) yield { type: "text", text: p };
  };
}

function failsWith(e: unknown, ...before: string[]): Script {
  return async function* () {
    for (const p of before) yield { type: "text", text: p };
    throw e;
  };
}

function cfg(over: Partial<FailoverSettings> = {}): FailoverSettings {
  return { ...defaultSettings().failover, ...over };
}

function setup(models: FakeModel[], over: Partial<FailoverSettings> = {}, clock = { t: 1_000_000 }, windows: Array<[number | null, number | null]> = []) {
  const health = new ModelHealth(() => clock.t);
  const slept: number[] = [];
  const chain: FailoverCandidate[] = models.map((m, i) => ({
    id: m.modelId,
    label: m.modelId.toUpperCase(),
    contextLength: windows[i]?.[0] ?? null,
    maxOutputTokens: windows[i]?.[1] ?? null,
    port: async () => m,
  }));
  const chat = new FailoverChat(chain, {
    settings: () => cfg(over),
    health,
    continuePrompt: () => "CONTINUE",
    now: () => clock.t,
    sleep: async (ms) => {
      slept.push(ms);
      clock.t += ms;
    },
  });
  return { chat, health, slept, clock };
}

async function run(chat: ChatModelPort, signal?: AbortSignal): Promise<{ text: string; chunks: ChatChunk[] }> {
  const chunks: ChatChunk[] = [];
  for await (const c of chat.stream({
    messages: [{ role: "user", content: "write a function" }],
    maxTokens: 100,
    signal,
  }))
    chunks.push(c);
  return {
    text: chunks.flatMap((c) => (c.type === "text" ? [c.text] : [])).join(""),
    chunks,
  };
}

describe("failover chain", () => {
  it("switches before the first token without the caller noticing, honouring Retry-After", async () => {
    const a = new FakeModel("a", failsWith(new ModelHttpError(429, "slow down", 45_000)));
    const b = new FakeModel("b", says("def add(x, y):", " return x + y"));
    const { chat, health } = setup([a, b]);
    const { text, chunks } = await run(chat);
    expect(text).toBe("def add(x, y): return x + y");
    expect(chunks.find((c) => c.type === "switch")).toMatchObject({
      from: "A",
      to: "B",
      reason: "rate limited (429)",
      resumed: false,
    });
    expect(health.available("a")).toBe(false);
    expect(health.snapshot().find((s) => s.id === "a")?.untilMs).toBe(45_000);
    expect(health.available("b")).toBe(true);
  });

  it("continues a reply that broke mid-stream on the next model and drops the repeated tail", async () => {
    const before = "Here is the plan. First we parse the input, then we validate each field and";
    const a = new FakeModel("a", failsWith(new Error("socket hang up"), before));
    const b = new FakeModel("b", says("then we validate each field and", " finally return the result."));
    const { chat } = setup([a, b]);
    const { text, chunks } = await run(chat);
    expect(text).toBe(`${before} finally return the result.`);
    expect(chunks.find((c) => c.type === "switch")).toMatchObject({
      resumed: true,
    });
    const sent = b.calls[0]!.messages;
    expect(sent.at(-2)).toEqual({ role: "assistant", content: before.slice(0, -"and".length) });
    expect(sent.at(-1)).toEqual({ role: "user", content: "CONTINUE" });
  });

  it("keeps the space when a reply is cut right after a word", async () => {
    const a = new FakeModel("a", failsWith(new Error("terminated"), "Hello. Your deploy"));
    const b = new FakeModel("b", says("deploy target is the staging cluster BLUEJAY."));
    const { chat } = setup([a, b]);
    expect((await run(chat)).text).toBe("Hello. Your deploy target is the staging cluster BLUEJAY.");
    expect(b.calls[0]!.messages.at(-2)).toEqual({ role: "assistant", content: "Hello. Your " });
  });

  it("moves past a continuation that comes back as template markup or nothing at all", async () => {
    const before = "2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31,";
    const a = new FakeModel("a", failsWith(new Error("terminated"), before));
    const b = new FakeModel("b", says("<|start|>", "assistant"));
    const c = new FakeModel("c", says(""));
    const d = new FakeModel("d", says("37, 41, 43, 47."));
    const { chat, health } = setup([a, b, c, d]);
    const { text, chunks } = await run(chat);
    expect(text).toBe(`${before} 37, 41, 43, 47.`);
    expect(chunks.filter((x) => x.type === "switch").map((x) => (x.type === "switch" ? `${x.to}:${x.reason}` : ""))).toEqual(["B:terminated", "C:garbled reply", "D:empty reply"]);
    expect(health.available("b")).toBe(false);
    expect(health.available("c")).toBe(false);
  });

  it("skips a continuation that starts the whole reply over", () => {
    const prior = "```python\ndef area(r):\n    return 3.14159 * r";
    const d = new ContinuationDeduper(prior, 16);
    const out = d.push(prior) + d.push(" * r\n```") + d.flush();
    expect(out).toBe(" * r\n```");
    expect(new ContinuationDeduper("abc", 16).push("brand new text that is long")).toBe("brand new text that is long");
  });

  it("restores the space a continuation drops at a prose seam, and nowhere else", () => {
    expect(seamSpace("2, 3, 5, 31,", "37, 41")).toBe(" ");
    expect(seamSpace("It works.", "Then we")).toBe(" ");
    expect(seamSpace("It works:", "**bold**")).toBe(" ");
    expect(seamSpace("2, 3, 31, ", "37")).toBe("");
    expect(seamSpace("hel", "lo")).toBe("");
    expect(seamSpace("pi is 3.", "14")).toBe("");
    expect(seamSpace("about 1,", "000 users")).toBe("");
    expect(seamSpace("at 10:", "30")).toBe("");
    expect(seamSpace("call os.", "path.join")).toBe("");
    expect(seamSpace("```python\nf(a,", "b)")).toBe("");
    expect(seamSpace("use `f(a,", "b)`")).toBe("");
    const d = new ContinuationDeduper("2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31,", 8);
    expect(d.push("37, 41, 43, 47.") + d.flush()).toBe(" 37, 41, 43, 47.");
  });

  it("resumes a backup before the last word, so a word cut or a word end joins cleanly", () => {
    expect(resumePoint("Hello. Your deploy")).toBe("Hello. Your ");
    expect(resumePoint("```repl\nresult = sum(i*i")).toBe("```repl\nresult = ");
    expect(resumePoint("2, 3, 31, ")).toBe("2, 3, 31, ");
    expect(resumePoint("Hello")).toBe("Hello");
    expect(resumePoint(`see ${"x".repeat(60)}`)).toBe(`see ${"x".repeat(60)}`);
  });

  it("joins a continuation that repeats the last word with that word's spacing", () => {
    const cont = (prior: string, next: string) => {
      const d = new ContinuationDeduper(prior, 16);
      return prior + d.push(next) + d.flush();
    };
    expect(cont("Hello. Your", "Your deploy target is BLUEJAY.")).toBe("Hello. Your deploy target is BLUEJAY.");
    expect(cont("Your dep", "deploy target is BLUEJAY.")).toBe("Your deploy target is BLUEJAY.");
    expect(cont("2, 3, 31, ", "31, 37, 41")).toBe("2, 3, 31, 37, 41");
    expect(cont("    return 3.14159 * r", "r\n```")).toBe("    return 3.14159 * r\n```");
    expect(cont("It works.", "Then we stop.")).toBe("It works. Then we stop.");
    expect(cont("```repl\nresult = sum(i*i", "i for i in range(1, 101))")).toBe("```repl\nresult = sum(i*i for i in range(1, 101))");
    expect(cont("```repl\nresult = sum(i*i", "i*i for i in range(1, 101))")).toBe("```repl\nresult = sum(i*i for i in range(1, 101))");
    expect(cont("Hello. Your", "r deploy")).toBe("Hello. Yourr deploy");
    expect(cont("x = total", "al + 1")).toBe("x = totalal + 1");
  });

  it("treats an empty reply as a failure and moves on", async () => {
    const a = new FakeModel("a", says());
    const b = new FakeModel("b", says("ok"));
    const { chat, health } = setup([a, b]);
    expect((await run(chat)).text).toBe("ok");
    expect(health.get("a")?.reason).toBe("empty reply");
  });

  it("treats a reply that stops after its reasoning as a failure, so the turn never ends without an answer", async () => {
    const a = new FakeModel("a", async function* () {
      yield {
        type: "thinking",
        text: "The user wants the number. Maybe they want us to state",
      };
    });
    const b = new FakeModel("b", says("42"));
    const { chat, health } = setup([a, b]);
    const { text, chunks } = await run(chat);
    expect(text).toBe("42");
    expect(chunks.find((c) => c.type === "switch")).toMatchObject({
      from: "A",
      to: "B",
      reason: "no answer after reasoning",
      resumed: false,
    });
    expect(health.get("a")?.reason).toBe("no answer after reasoning");
  });

  it("sends the system prompt inside the first user message for a model that drops the system role", async () => {
    const { foldSystemPrompt } = await import("@rlm/contracts");
    const msgs = [
      { role: "system" as const, content: "Memory: deploy target is BLUEJAY." },
      { role: "user" as const, content: "where do we deploy?" },
      { role: "assistant" as const, content: "let me check" },
    ];
    expect(foldSystemPrompt(msgs)).toEqual([
      {
        role: "user",
        content: "<instructions>\nMemory: deploy target is BLUEJAY.\n</instructions>\n\nwhere do we deploy?",
      },
      { role: "assistant", content: "let me check" },
    ]);
    expect(
      foldSystemPrompt([
        { role: "system", content: "S" },
        { role: "assistant", content: "a" },
      ]),
    ).toEqual([
      { role: "user", content: "<instructions>\nS\n</instructions>" },
      { role: "assistant", content: "a" },
    ]);
    expect(foldSystemPrompt([{ role: "user", content: "u" }])).toEqual([{ role: "user", content: "u" }]);
  });

  it("waits for the first model to recover when every model is cooling down, then finishes", async () => {
    let calls = 0;
    const a = new FakeModel("a", async function* () {
      if (calls++ === 0) throw new ModelHttpError(503, "overloaded", 20_000);
      yield { type: "text", text: "recovered" };
    });
    const { chat, slept } = setup([a], { waitForRecoverySec: 60 });
    const { text, chunks } = await run(chat);
    expect(text).toBe("recovered");
    expect(slept).toEqual([20_000]);
    expect(chunks.find((c) => c.type === "notice")).toMatchObject({
      text: expect.stringMatching(/cooling down; retrying A in 20 s/),
    });
  });

  it("gives up past the recovery window", async () => {
    const a = new FakeModel("a", failsWith(new ModelHttpError(429, "quota", 120_000)));
    const { chat } = setup([a], { waitForRecoverySec: 60 });
    await expect(run(chat)).rejects.toThrow(/quota/);
  });

  it("does not wait on failures a retry will not fix", async () => {
    const a = new FakeModel("a", failsWith(new ModelHttpError(401, "bad key")));
    const b = new FakeModel("b", failsWith(new ModelHttpError(404, "no such model")));
    const { chat, slept } = setup([a, b]);
    await expect(run(chat)).rejects.toThrow(/no such model/);
    expect(slept).toEqual([]);
  });

  it("moves a prompt a model refuses as too long to one that holds it, and lets the engine compact when none does", async () => {
    const a = new FakeModel("a", failsWith(new ContextOverflowError("prompt is too long")));
    const b = new FakeModel("b", says("fits"));
    const one = setup([a, b]);
    expect((await run(one.chat)).text).toBe("fits");
    expect(one.health.available("a")).toBe(true);
    expect(one.health.learnedWindow("a")).toBeGreaterThan(0);

    const c = new FakeModel("c", failsWith(new ContextOverflowError("prompt is too long")));
    const d = new FakeModel("d", failsWith(new ContextOverflowError("prompt is too long")));
    const two = setup([c, d]);
    await expect(run(two.chat)).rejects.toBeInstanceOf(ContextOverflowError);
    expect(c.calls).toHaveLength(1);
    expect(d.calls).toHaveLength(1);
    expect(two.health.available("c") && two.health.available("d")).toBe(true);
  });

  it("stops on a user abort without cooling the model down", async () => {
    const ctl = new AbortController();
    const a = new FakeModel("a", async function* () {
      yield { type: "text", text: "par" };
      ctl.abort();
      throw new DOMException("stopped", "AbortError");
    });
    const b = new FakeModel("b", says("never"));
    const { chat, health } = setup([a, b]);
    await expect(run(chat, ctl.signal)).rejects.toThrow(/stopped/);
    expect(b.calls).toHaveLength(0);
    expect(health.available("a")).toBe(true);
  });

  it("stays on the fallback while the primary cools down, then returns to it", async () => {
    let aCalls = 0;
    const a = new FakeModel("a", async function* () {
      if (aCalls++ === 0) throw new ModelHttpError(429, "limit", 30_000);
      yield { type: "text", text: "primary" };
    });
    const b = new FakeModel("b", says("fallback"));
    const { chat, clock } = setup([a, b]);
    expect((await run(chat)).text).toBe("fallback");
    const second = await run(chat);
    expect(second.text).toBe("fallback");
    expect(second.chunks.find((c) => c.type === "switch")).toMatchObject({
      reason: expect.stringMatching(/A is cooling down: rate limited/),
    });
    expect((await run(chat)).chunks.some((c) => c.type === "switch")).toBe(false);
    clock.t += 31_000;
    expect((await run(chat)).text).toBe("primary");
  });

  it("reports the window of the models that can serve now, so the caller sizes its prompt to it", async () => {
    const a = new FakeModel("a", says("x"));
    const b = new FakeModel("b", says("y"));
    const { chat, health } = setup([a, b], {}, undefined, [
      [100_000, 8000],
      [12_000, 1000],
    ]);
    expect(chat.activeLimits()).toEqual({
      contextLength: 100_000,
      maxOutputTokens: 8000,
    });
    health.fail(
      "a",
      {
        severity: "transient",
        reason: "rate limited (429)",
        retryAfterMs: 60_000,
      },
      cfg(),
    );
    expect(chat.activeLimits()).toEqual({
      contextLength: 12_000,
      maxOutputTokens: 1000,
    });
  });

  it("never sends a prompt to a model that cannot hold it, and does not mark that model broken", async () => {
    const a = new FakeModel("a", failsWith(new ModelHttpError(429, "slow down", 60_000)));
    const b = new FakeModel("b", says("fits"));
    const { chat, health } = setup([a, b], {}, undefined, [
      [100_000, 8000],
      [1_000, 200],
    ]);
    const big: ChatRequest = {
      messages: [{ role: "user", content: "x".repeat(20_000) }],
      maxTokens: 500,
    };
    const drain = async (req: ChatRequest) => {
      for await (const _ of chat.stream(req)) void _;
    };
    await expect(drain(big)).rejects.toBeInstanceOf(ModelWindowError);
    expect(b.calls).toHaveLength(0);
    expect(health.available("b")).toBe(true);
    // Once compacted to the backup's window it goes through, with the backup's own output limit.
    await drain({
      messages: [{ role: "user", content: "x".repeat(1_500) }],
      maxTokens: 500,
    });
    expect(b.calls).toHaveLength(1);
    expect(b.calls[0]!.maxTokens).toBe(200);
  });

  it("skips a model a continued reply outgrows without cooling it down, and hides the continuation's prompt size", async () => {
    const a = new FakeModel("a", failsWith(new Error("socket hang up"), "The plan has three parts. First, "));
    const b = new FakeModel("b", failsWith(new ContextOverflowError("too long")));
    const c = new FakeModel("c", async function* () {
      yield { type: "text", text: "we read the input." };
      yield { type: "usage", inputTokens: 999, outputTokens: 5 };
    });
    const { chat, health } = setup([a, b, c]);
    const { text, chunks } = await run(chat);
    expect(text).toBe("The plan has three parts. First, we read the input.");
    expect(health.available("b")).toBe(true);
    expect(chunks.find((x) => x.type === "usage")).toEqual({
      type: "usage",
      inputTokens: 0,
      outputTokens: 5,
    });
  });

  it("classifies failures", () => {
    expect(classifyFailure(new ModelHttpError(500, "x")).severity).toBe("transient");
    expect(classifyFailure(new ModelHttpError(402, "x")).severity).toBe("hard");
    expect(classifyFailure(new Error("model request: the server sent nothing for 5 s and looks stuck")).reason).toBe("stalled");
  });
});

describe("failover through the router over HTTP", () => {
  const servers: Server[] = [];
  afterEach(() => {
    for (const s of servers.splice(0)) {
      s.closeAllConnections();
      s.close();
    }
  });

  async function serve(handler: (res: ServerResponse) => void): Promise<string> {
    const s = createServer((req, res) => {
      req.resume();
      req.on("end", () => handler(res));
    });
    servers.push(s);
    await new Promise<void>((r) => s.listen(0, "127.0.0.1", r));
    return `http://127.0.0.1:${(s.address() as AddressInfo).port}/v1`;
  }

  const chunk = (o: object) => `data: ${JSON.stringify(o)}\n\n`;

  function router(urls: string[], stallSec = 300, s = defaultSettings()) {
    s.rlm.modelStallSec = stallSec;
    urls.forEach((u, i) => {
      s.providers.push({
        id: `p${i}`,
        kind: "openai_compat",
        label: `P${i}`,
        baseUrl: u,
        hasKey: false,
      } as never);
      s.models.push({
        id: `m${i}`,
        providerId: `p${i}`,
        name: `model-${i}`,
        label: `Model ${i}`,
        kind: "chat",
        contextLength: 8192,
        maxOutputTokens: 512,
        dim: null,
      } as never);
    });
    s.roles.agent = "m0";
    s.roles.agentFallbacks = urls.slice(1).map((_, i) => `m${i + 1}`);
    return new ProviderRouter({
      settings: () => s,
      apiKey: () => null,
      localChat: () => {
        throw new Error("no local");
      },
      localEmbed: async () => null,
      localNativeContext: () => null,
      continuePrompt: () => "CONTINUE",
    });
  }

  it("a stream that drops mid-reply is finished by the next provider", async () => {
    const dropper = await serve((res) => {
      res.writeHead(200, { "Content-Type": "text/event-stream" });
      res.write(
        chunk({
          choices: [{ delta: { content: "The answer is computed by summing" } }],
        }),
      );
      setTimeout(() => res.destroy(), 20);
    });
    const backup = await serve((res) => {
      res.writeHead(200, { "Content-Type": "text/event-stream" });
      res.write(chunk({ choices: [{ delta: { content: " both numbers." } }] }));
      res.end("data: [DONE]\n\n");
    });
    const r = router([dropper, backup]);
    const { text, chunks } = await run(await r.chat("agent"));
    expect(text).toBe("The answer is computed by summing both numbers.");
    expect(chunks.find((c) => c.type === "switch")).toMatchObject({
      from: "Model 0",
      to: "Model 1",
      resumed: true,
    });
    expect(r.health.available("m0")).toBe(false);
  });

  it("gateway keepalives do not hide a stuck upstream", async () => {
    const keepalive = await serve((res) => {
      res.writeHead(200, { "Content-Type": "text/event-stream" });
      const t = setInterval(
        () =>
          res.write(
            chunk({
              id: "chatcmpl-keepalive",
              model: "keepalive",
              choices: [{ delta: {} }],
            }),
          ),
        200,
      );
      res.on("close", () => clearInterval(t));
    });
    const backup = await serve((res) => {
      res.writeHead(200, { "Content-Type": "text/event-stream" });
      res.end(`${chunk({ choices: [{ delta: { content: "done" } }] })}data: [DONE]\n\n`);
    });
    const r = router([keepalive, backup], 1);
    const t0 = Date.now();
    const { text, chunks } = await run(await r.chat("agent"));
    expect(text).toBe("done");
    expect(chunks.find((c) => c.type === "switch")).toMatchObject({
      reason: "stalled",
    });
    expect(Date.now() - t0).toBeLessThan(5000);
  });

  it("after a switch, a backup that drops the system role still gets the instructions and memory", async () => {
    const limited = await serve((res) => {
      res.writeHead(429, { "Content-Type": "application/json" });
      res.end('{"error":"rate limited"}');
    });
    const bodies: Array<{
      messages: Array<{ role: string; content: string }>;
    }> = [];
    const s0 = createServer((req, res) => {
      let b = "";
      req.on("data", (d) => (b += d));
      req.on("end", () => {
        bodies.push(JSON.parse(b));
        res.writeHead(200, { "Content-Type": "text/event-stream" });
        res.end(`${chunk({ choices: [{ delta: { content: "BLUEJAY" } }] })}data: [DONE]\n\n`);
      });
    });
    servers.push(s0);
    await new Promise<void>((r) => s0.listen(0, "127.0.0.1", r));
    const s = defaultSettings();
    const r = router([limited, `http://127.0.0.1:${(s0.address() as AddressInfo).port}/v1`], 300, s);
    s.models.find((m) => m.id === "m1")!.systemAsUser = true;
    const chunks: ChatChunk[] = [];
    const chat = await r.chat("agent");
    for await (const c of chat.stream({
      messages: [
        { role: "system", content: "Memory: deploy target is BLUEJAY." },
        { role: "user", content: "where?" },
      ],
      maxTokens: 100,
    }))
      chunks.push(c);
    expect(chunks.find((c) => c.type === "switch")).toMatchObject({
      from: "Model 0",
      to: "Model 1",
    });
    expect(bodies[0]!.messages).toEqual([
      {
        role: "user",
        content: "<instructions>\nMemory: deploy target is BLUEJAY.\n</instructions>\n\nwhere?",
      },
    ]);
  });

  it("treats a gateway's input-limit refusal as an oversized prompt, not a rate limit, and learns the window", async () => {
    const ddg = await serve((res) => {
      res.writeHead(429, { "Content-Type": "application/json" });
      res.end('{"error":{"message":"[429]: DuckDuckGo AI Chat error: ERR_INPUT_LIMIT","type":"rate_limit_error"}}');
    });
    const backup = await serve((res) => {
      res.writeHead(200, { "Content-Type": "text/event-stream" });
      res.end(`${chunk({ choices: [{ delta: { content: "fits here" } }] })}data: [DONE]\n\n`);
    });
    const r = router([ddg, backup]);
    const { text, chunks } = await run(await r.chat("agent"));
    expect(text).toBe("fits here");
    expect(chunks.find((c) => c.type === "switch")).toMatchObject({ from: "Model 0", to: "Model 1", reason: "the prompt is larger than its real window" });
    expect(r.health.available("m0")).toBe(true);
    expect(r.health.learnedWindow("m0")).toBeGreaterThan(0);
  });

  it("keeps a real tokens-per-minute rate limit a rate limit", async () => {
    const groq = await serve((res) => {
      res.writeHead(429, { "Content-Type": "application/json" });
      res.end('{"error":{"message":"Rate limit reached on tokens per minute (TPM): too many tokens, try again in 20s"}}');
    });
    const backup = await serve((res) => {
      res.writeHead(200, { "Content-Type": "text/event-stream" });
      res.end(`${chunk({ choices: [{ delta: { content: "ok" } }] })}data: [DONE]\n\n`);
    });
    const r = router([groq, backup]);
    const { chunks } = await run(await r.chat("agent"));
    expect(chunks.find((c) => c.type === "switch")).toMatchObject({ reason: "rate limited (429)" });
    expect(r.health.available("m0")).toBe(false);
    expect(r.health.learnedWindow("m0")).toBeUndefined();
  });

  it("gives a separate summarizer the agent chain as backups, so compaction and memory keep working", async () => {
    const s = defaultSettings();
    const r = router(["http://127.0.0.1:9/v1", "http://127.0.0.1:9/v1", "http://127.0.0.1:9/v1"], 300, s);
    s.roles.summarizer = "m2";
    expect(r.chainFor("summarizer")).toEqual(["m2", "m0", "m1"]);
    expect(r.chainFor("agent")).toEqual(["m0", "m1", "m2"]);
  });

  it("uses the plain model when failover is off", async () => {
    const s = defaultSettings();
    const r = router(["http://127.0.0.1:9/v1", "http://127.0.0.1:9/v1"], 300, s);
    expect((await r.chat("agent")).constructor.name).toBe("FailoverChat");
    s.failover.enabled = false;
    r.invalidate();
    expect((await r.chat("agent")).constructor.name).toBe("OpenAICompatChat");
  });
});
