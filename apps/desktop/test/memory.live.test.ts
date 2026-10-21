import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import { FilePrompts, OpenAICompatChat, openDb, SqliteMemoryStore } from "@rlm/adapters";
import { defaultSettings } from "@rlm/contracts";
import { MemoryService, type MemoryScope } from "@rlm/core";
import { BgeM3Embedder, cpuThreads } from "@rlm/inference";

/**
 * Long-term memory with a real model and the real bge-m3: what typical exchanges store, and what a new chat recalls in
 * the same project and in another one.
 *   RLM_LIVE_MEMORY=http://localhost:8100/v1 [RLM_MEMORY_MODEL=Qwen3.5-4B] [RLM_MEMORY_KEY=...] pnpm vitest run apps/desktop/test/memory.live.test.ts
 */
const base = process.env.RLM_LIVE_MEMORY ?? "";
const ROOT = path.resolve(__dirname, "../../..");

describe.skipIf(!base)("long-term memory (live)", () => {
  const dir = mkdtempSync(path.join(tmpdir(), "rlm-mem-"));
  afterAll(() => rmSync(dir, { recursive: true, force: true }));
  const store = new SqliteMemoryStore(openDb(path.join(dir, "m.db"), null));
  const embedder = new BgeM3Embedder(path.join(ROOT, "resources/models") + path.sep, "bge-m3", 1024, () => 1024, () => cpuThreads(0));
  const chat = new OpenAICompatChat("live", { baseUrl: base, apiKey: process.env.RLM_MEMORY_KEY ?? null, model: process.env.RLM_MEMORY_MODEL ?? "Qwen3.5-4B" });
  let reply = "";
  const mem = new MemoryService({
    store,
    embedder: async () => embedder,
    chat: async () => ({
      ...chat,
      modelId: chat.modelId,
      async *stream(req) {
        reply = "";
        for await (const c of chat.stream(req)) {
          if (c.type === "text") reply += c.text;
          yield c;
        }
      },
    }),
    prompts: new FilePrompts(path.join(ROOT, "prompts")),
    clock: { now: () => Date.now() },
    log: { info() {}, warn: (m: string) => console.warn(m), error: (m: string) => console.warn(m) },
    status: { memory() {}, kernel() {} },
  });
  const s = defaultSettings().memory;
  const A: MemoryScope = { userId: "u", sessionId: "chat-a1", workspaceId: "/work/billing-api" };
  const B: MemoryScope = { userId: "u", sessionId: "chat-b1", workspaceId: "/work/landing-page" };
  const facts = async () =>
    (await store.list({ user_id: "u", memory_type: "long_term" })).map((p) => ({ text: p.payload.text, about: p.payload.workspace_id ? "project" : "user" }));
  const learned = async (user: string, assistant: string) => {
    const before = new Set((await facts()).map((f) => f.text));
    await mem.learnFacts(A, s, user, assistant);
    const added = (await facts()).filter((f) => !before.has(f.text));
    console.log(`${user.slice(0, 50).replace(/\n/g, " ")}…\n  model: ${reply.trim()}\n  stored: ${added.map((f) => `[${f.about}] ${f.text}`).join(" | ") || "nothing"}`);
    return added;
  };

  it("stores the user's identity and tools as user facts", async () => {
    const added = await learned("I'm Amanda, a backend engineer. Please always use pnpm, never npm, in my projects.", "Got it, I'll use pnpm.");
    expect(added.some((f) => f.about === "user" && /pnpm/i.test(f.text))).toBe(true);
    expect(added.some((f) => f.about === "user" && /amanda/i.test(f.text))).toBe(true);
    expect(added.some((f) => /priya|mobile/i.test(f.text))).toBe(false);
  }, 180_000);

  it("stores the project's stack and deploy command as project facts", async () => {
    const added = await learned(
      "Some background: this repo is the billing API. It runs on Node 22 with Postgres 16, and we deploy it with `docker compose up -d` on the staging box.",
      "Thanks, noted the stack and deploy command.",
    );
    expect(added.some((f) => f.about === "project" && /node 22|postgres/i.test(f.text))).toBe(true);
  }, 180_000);

  it.each([
    ["a one-off request", "make the submit button on the invoice form blue", "Changed the button color to blue in InvoiceForm.tsx."],
    [
      "a pasted error",
      "I get this error:\nTypeError: Cannot read properties of undefined (reading 'id')\n    at handler (/src/routes/invoice.ts:42:17)\n    at process.processTicksAndRejections (node:internal/process/task_queues:95:5)",
      "The invoice lookup returns undefined when the id is missing; I added a guard.",
    ],
    ["a question", "what does the computeTax function do?", "It applies the regional VAT rate to the subtotal and rounds to cents."],
    ["a secret", "here is the stripe key for testing: sk_test_FAKE_not_a_real_key, wire it up", "Added it to the .env file as STRIPE_KEY."],
    ["something only the assistant said", "ok thanks, looks good", "The project also uses Redis for caching and Kafka for events, so I updated both clients."],
  ])("stores nothing from %s", async (_, user, assistant) => {
    expect(await learned(user, assistant)).toEqual([]);
  }, 180_000);

  it("replaces a fact the user corrects instead of keeping both", async () => {
    await learned("Heads up: we upgraded the billing API from Node 22 to Node 24 last week.", "Noted, I'll target Node 24.");
    const all = (await facts()).map((f) => f.text).join("\n");
    expect(all).toMatch(/node 24/i);
    expect(all).not.toMatch(/node 22(?! to)/i);
  }, 180_000);

  it("recalls the project's facts in a new chat of that project only, and the user's facts everywhere", async () => {
    console.log(`stored:\n${(await facts()).map((f) => `  [${f.about}] ${f.text}`).join("\n")}`);
    const recall = async (scope: MemoryScope, q: string) => (await mem.recall(q, { ...scope, sessionId: `${scope.sessionId}-new` }, s)).items.map((i) => i.text);
    const a = await recall(A, "how do I deploy this to staging?");
    expect(a.join("\n")).toMatch(/docker compose/i);
    const b = await recall(B, "install the dependencies and start the dev server");
    expect(b.join("\n")).toMatch(/pnpm/i);
    expect(b.join("\n")).not.toMatch(/billing|postgres|node 22|docker compose/i);
  }, 180_000);
});
