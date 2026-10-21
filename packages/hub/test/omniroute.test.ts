import { mkdtempSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import net from "node:net";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import { agentPickerModels, BUILTIN, defaultSettings, OMNIROUTE_AUTO_MODEL_ID, type OmniRouteView, type Settings } from "@rlm/contracts";
import { OmniRouteService, capabilityTier, freeChatCandidates, interleaveByProvider, parseCliJson, satisfiesEngines, type GatewayModel } from "../src";

describe("omniroute helpers", () => {
  it("reads JSON after the CLI's banner lines", () => {
    expect(parseCliJson('  📋 Loaded env from /x/.env\n  ⚠ something\n{\n  "a": [1]\n}\n')).toEqual({ a: [1] });
    expect(parseCliJson("banner\n[1,2]")).toEqual([1, 2]);
    expect(() => parseCliJson("no json here")).toThrow(/no JSON/);
  });

  it("checks node versions against engine ranges", () => {
    const r = ">=22.22.2 <23 || >=24.0.0 <27";
    expect(satisfiesEngines("v22.23.2", r)).toBe(true);
    expect(satisfiesEngines("v22.12.0", r)).toBe(false);
    expect(satisfiesEngines("v23.1.0", r)).toBe(false);
    expect(satisfiesEngines("v26.0.0", r)).toBe(true);
    expect(satisfiesEngines("v27.0.0", r)).toBe(false);
    expect(satisfiesEngines("v20.0.0", ">=18")).toBe(true);
  });

  it("keeps text chat models of free providers, best first, capped per provider", () => {
    const models: GatewayModel[] = [
      {
        id: "cfp/a",
        owned_by: "cloudflare-playground",
        context_length: 128000,
      },
      {
        id: "cfp/b",
        owned_by: "cloudflare-playground",
        context_length: 256000,
        capabilities: { tool_calling: true },
      },
      { id: "cfp/c", owned_by: "cloudflare-playground", context_length: 64000 },
      { id: "horde/img", owned_by: "aihorde" },
      {
        id: "groq/x",
        owned_by: "groq",
        context_length: 131072,
        output_modalities: ["text"],
      },
      {
        id: "groq/tts",
        owned_by: "groq",
        context_length: 8000,
        output_modalities: ["audio"],
      },
      { id: "paid/x", owned_by: "openai", context_length: 400000 },
      { id: "rlm-free", owned_by: "combo", context_length: 400000 },
    ];
    const out = freeChatCandidates(models, new Set(["cloudflare-playground", "aihorde", "groq"]), 2).map((m) => m.id);
    expect(out).toEqual(["cfp/b", "cfp/a", "groq/x"]);
  });

  it("ranks unsized flagship models above models that say they are small", () => {
    expect(capabilityTier("cfp/zai-org/glm-5.2")).toBe(2);
    expect(capabilityTier("cfp/moonshotai/kimi-k2.7-code")).toBe(2);
    expect(capabilityTier("cfp/openai/gpt-oss-120b")).toBe(2);
    expect(capabilityTier("cfp/aisingapore/gemma-sea-lion-v4-27b-it")).toBe(0);
    expect(capabilityTier("cfp/openai/gpt-oss-20b")).toBe(0);
    expect(capabilityTier("unc/Qwen3.6-35B-A3B")).toBe(1);
    expect(capabilityTier("ddgw/gpt-5.4-mini")).toBe(1);
    expect(capabilityTier("x/llama-3.1-8b-instant")).toBe(0);
    const ranked = freeChatCandidates(
      [
        {
          id: "cfp/openai/gpt-oss-20b",
          owned_by: "cfp",
          context_length: 400000,
          capabilities: { tool_calling: true },
        },
        { id: "cfp/zai-org/glm-5.2", owned_by: "cfp", context_length: 128000 },
      ],
      new Set(["cfp"]),
      1,
    );
    expect(ranked.map((m) => m.id)).toEqual(["cfp/zai-org/glm-5.2"]);
  });

  it("interleaves providers so one outage costs one step", () => {
    const ranked = [
      { id: "a1", provider: "a" },
      { id: "a2", provider: "a" },
      { id: "a3", provider: "a" },
      { id: "b1", provider: "b" },
      { id: "c1", provider: "c" },
    ];
    expect(interleaveByProvider(ranked, 4).map((m) => m.id)).toEqual(["a1", "b1", "c1", "a2"]);
  });
});

/** A stand-in `omniroute` package: CLI + server speaking just enough of the real protocol. */
const FAKE_BIN = String.raw`
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
const data = process.env.DATA_DIR;
const st = path.join(data, "fake.json");
const load = () => { try { return JSON.parse(fs.readFileSync(st, "utf8")); } catch { return { keys: [], combos: {}, conns: [] }; } };
const save = (s) => fs.writeFileSync(st, JSON.stringify(s));
const out = (o) => console.log("  Loaded env from somewhere\n" + JSON.stringify(o, null, 2));
const [cmd, sub, ...rest] = process.argv.slice(2);
const MODELS = [
  { id: "kl/fast", owned_by: "keyless", context_length: 64000, max_output_tokens: 8000, capabilities: { tool_calling: true } },
  { id: "kl/broken", owned_by: "keyless", context_length: 200000, capabilities: { tool_calling: true } },
  { id: "kl/chatty", owned_by: "keyless", context_length: 32000 },
  { id: "kl/tiny", owned_by: "keyless", context_length: 400000 },
  { id: "kl/mute", owned_by: "keyless", context_length: 128000 },
  { id: "fr/big", owned_by: "freekey", context_length: 128000, capabilities: { tool_calling: true } },
  { id: "paid/x", owned_by: "paid", context_length: 400000 },
];
if (cmd === "serve") {
  fs.mkdirSync(data, { recursive: true });
  fs.writeFileSync(path.join(data, "env.json"), JSON.stringify({ host: process.env.OMNIROUTE_SERVER_HOST, requireKey: process.env.REQUIRE_API_KEY, password: !!process.env.INITIAL_PASSWORD, port: process.env.PORT }));
  const srv = http.createServer((req, res) => {
    const s = load();
    const send = (code, o) => { res.writeHead(code, { "Content-Type": "application/json" }); res.end(JSON.stringify(o)); };
    if (req.url === "/api/health") return send(200, { ok: true });
    const auth = (req.headers.authorization || "").replace("Bearer ", "");
    if (!s.keys.includes(auth)) return send(401, { error: { message: "key required" } });
    if (req.url === "/v1/models") return send(200, { data: MODELS.filter((m) => m.owned_by !== "freekey" || s.conns.some((c) => c.provider === "freekey")) });
    let body = "";
    req.on("data", (d) => (body += d));
    req.on("end", () => {
      const { model, messages } = JSON.parse(body);
      if (model === "kl/broken") return send(502, { error: { message: "upstream down" } });
      // Hidden input limits, worded like Cloudflare Playground's and DuckDuckGo's.
      const chars = messages.reduce((n, m) => n + m.content.length, 0);
      if (model === "fr/big" && chars > 20000) return send(502, { error: { message: "[502]: Prompt too long (max 20000 characters)." } });
      if (model === "kl/tiny" && chars > 6000) return send(429, { error: { message: "[429]: DuckDuckGo AI Chat error: ERR_INPUT_LIMIT" } });
      // The same limit relayed as a bare 502 that does not say why.
      if (model === "kl/mute" && chars > 6000) return send(502, { error: { message: "[502]: upstream error" } });
      // fr/big's provider drops the system role, like the web-playground providers do.
      const all = (model === "fr/big" ? messages.filter((m) => m.role !== "system") : messages).map((m) => m.content).join("\n");
      const words = [/secret word is (\S+)\./, /first test word is (\S+)\./, /last test word is (\S+)\./].map((re) => all.match(re)?.[1]).filter(Boolean);
      const fence = "\x60\x60\x60";
      const agentTask = /REPL cell that prints|files are in the project root/.test(all);
      const content =
        model === "kl/chatty" ? "Sure! Here is a word for you."
        : agentTask ? (model === "kl/fast" ? fence + "repl\nprint(17 * 23)\n" + fence : "I can't see your project files from here.")
        : words.join(" ") || "I don't know any secret word.";
      send(200, { choices: [{ message: { content } }] });
    });
  });
  srv.listen(Number(process.argv[process.argv.indexOf("--port") + 1]), process.env.OMNIROUTE_SERVER_HOST);
  fs.writeFileSync(path.join(data, "pid"), String(process.pid));
} else if (cmd === "stop") {
  try { process.kill(Number(fs.readFileSync(path.join(data, "pid"), "utf8"))); } catch {}
  console.log("stopped");
} else if (cmd === "api") {
  const s = load(); const key = "sk-test-" + s.keys.length; s.keys.push(key); save(s); out({ key, id: "k" + s.keys.length });
} else if (cmd === "providers" && sub === "available") {
  const cat = rest[rest.indexOf("--category") + 1];
  out({ providers: cat === "noauth" ? [{ id: "keyless", name: "Keyless", hasFree: true }] : [{ id: "freekey", name: "Free Key", hasFree: true, website: "https://example.com" }, { id: "paid", name: "Paid", hasFree: false }] });
} else if (cmd === "providers" && sub === "list") {
  out({ providers: load().conns });
} else if (cmd === "providers" && sub === "add") {
  const key = fs.readFileSync(0, "utf8").trim();
  if (!key) { console.error("key required"); process.exit(1); }
  const s = load(); s.conns.push({ id: "c1", provider: rest[0], name: rest[0], isActive: true, testStatus: "active", lastError: null, keyLen: key.length }); save(s); out({ ok: true });
} else if (cmd === "combo" && sub === "delete") {
  const s = load(); delete s.combos[rest[0]]; save(s); console.log("deleted");
} else if (cmd === "combo" && sub === "create") {
  const s = load(); s.combos[rest[0]] = { strategy: rest[rest.indexOf("--strategy") + 1], models: rest[rest.indexOf("--models") + 1].split(",") }; save(s); console.log("created");
} else { console.error("unknown " + process.argv.slice(2).join(" ")); process.exit(2); }
`;

function freePort(): Promise<number> {
  return new Promise((resolve) => {
    const s = net.createServer().listen(0, "127.0.0.1", () => {
      const p = (s.address() as net.AddressInfo).port;
      s.close(() => resolve(p));
    });
  });
}

describe("OmniRouteService against a fake gateway", () => {
  const dir = mkdtempSync(path.join(tmpdir(), "omni-"));
  const pkg = path.join(dir, "runtime", "node_modules", "omniroute");
  mkdirSync(path.join(pkg, "bin"), { recursive: true });
  writeFileSync(
    path.join(pkg, "package.json"),
    JSON.stringify({
      name: "omniroute",
      version: "9.9.9",
      engines: { node: ">=18" },
    }),
  );
  writeFileSync(path.join(pkg, "bin", "omniroute.mjs"), FAKE_BIN);
  let settings: Settings = defaultSettings();
  const secrets = new Map<string, string>();
  const views: OmniRouteView[] = [];
  let keyPushes = 0;
  let svc: OmniRouteService;

  afterAll(async () => {
    await svc?.stop();
  });

  it("starts locked down, makes a key, picks working free models and wires the failover chain", async () => {
    settings = {
      ...settings,
      roles: { ...settings.roles, agent: "builtin:agent" },
      omniroute: {
        ...settings.omniroute,
        port: await freePort(),
        maxChainModels: 3,
      },
    };
    svc = new OmniRouteService({
      dir,
      settings: () => settings,
      updateSettings: (p) => (settings = { ...settings, ...p }),
      secrets: {
        get: (k) => secrets.get(k) ?? null,
        set: (k, v) => void secrets.set(k, v),
      },
      emit: (v) => views.push(v),
      log: () => undefined,
      keysChanged: () => keyPushes++,
      nodeCandidates: () => [process.execPath],
      agentPrompt: () => "You are RLM. Run code in repl cells.\n{{tool_surface}}\nSub-agents: {{can_spawn}}, depth {{max_depth}}.",
    });
    await svc.boot();
    expect(svc.state().phase).toBe("stopped");
    expect(svc.state().version).toBe("9.9.9");

    await svc.start();
    expect(svc.state().phase).toBe("running");
    const env = JSON.parse(readFileSync(path.join(dir, "data", "env.json"), "utf8"));
    expect(env).toMatchObject({
      host: "127.0.0.1",
      requireKey: "true",
      password: true,
    });
    expect(secrets.get("provider:omniroute")).toBe("sk-test-0");
    expect(secrets.get("omniroute:dashboard")?.length).toBeGreaterThan(16);
    expect(keyPushes).toBe(1);
    expect(settings.providers.find((p) => p.id === "omniroute")).toMatchObject({
      managed: "omniroute",
      hasKey: true,
      baseUrl: `http://127.0.0.1:${settings.omniroute.port}/v1`,
    });
    // Once the gateway runs, the pickers list "OmniRoute"; the agent keeps its model until the user picks it.
    expect(settings.models.find((m) => m.id === OMNIROUTE_AUTO_MODEL_ID)).toMatchObject({ providerId: "omniroute", kind: "chat", label: expect.stringMatching(/^OmniRoute/) });
    expect(settings.roles.agent).toBe("builtin:agent");

    await svc.refreshCatalog();
    expect(svc.state().providers.map((p) => [p.id, p.keyless, p.connected])).toEqual([
      ["keyless", true, true],
      ["freekey", false, false],
    ]);
    await svc.connect("freekey", "secret-key");
    expect(svc.state().providers.find((p) => p.id === "freekey")?.connected).toBe(true);

    await svc.setup([]);
    const byId = new Map(svc.state().models.map((m) => [m.id, m]));
    expect(byId.get("kl/fast")?.ok).toBe(true);
    expect(byId.get("kl/broken")).toMatchObject({
      ok: false,
      error: "upstream down",
    });
    expect(byId.get("kl/chatty")).toMatchObject({
      ok: false,
      error: expect.stringMatching(/would forget its memory/),
    });
    expect(byId.has("paid/x")).toBe(false);
    expect(byId.get("kl/fast")).toMatchObject({ systemAsUser: false, inputLimitChars: null, agentReady: true });
    // Given the agent's instructions, fr/big answers without writing a cell: chained only as a later backup.
    expect(byId.get("fr/big")?.agentReady).toBe(false);
    // A hidden input limit is measured; one too small for the agent's instructions keeps the model out of the chain.
    expect(byId.get("fr/big")).toMatchObject({ ok: true, systemAsUser: true, inputLimitChars: 18000 });
    expect(byId.get("kl/tiny")).toMatchObject({ ok: false, inputLimitChars: 5000, error: expect.stringMatching(/accepts only about 5000 characters/) });
    // A model that answers short prompts but fails long ones without naming the size is measured all the same.
    expect(byId.get("kl/mute")).toMatchObject({ ok: false, inputLimitChars: 5000, error: expect.stringMatching(/accepts only about 5000 characters/) });

    // The agent's model is "OmniRoute", which stands for the chain; the model whose provider drops system prompts gets
    // the agent's instructions and memory inside the user message. The previous model is the last resort.
    expect(settings.roles.agent).toBe(OMNIROUTE_AUTO_MODEL_ID);
    expect(settings.omniroute.chain).toEqual(["omniroute:kl/fast", "omniroute:fr/big"]);
    expect(settings.roles.agentFallbacks).toEqual(["builtin:agent"]);
    expect(agentPickerModels(settings).map((m) => m.id)).toEqual([BUILTIN.agentModelId, OMNIROUTE_AUTO_MODEL_ID]);
    expect(settings.models.find((m) => m.id === OMNIROUTE_AUTO_MODEL_ID)).toMatchObject({ contextLength: 64000, maxOutputTokens: 8000 });
    const models = new Map(settings.models.filter((m) => m.providerId === "omniroute").map((m) => [m.id, m]));
    expect(models.get("omniroute:kl/fast")).toMatchObject({
      contextLength: 64000,
      maxOutputTokens: 8000,
    });
    expect(models.get("omniroute:kl/fast")?.systemAsUser).toBeUndefined();
    // 18000 measured characters = 5625 tokens of input, plus as much again for the reply.
    expect(models.get("omniroute:fr/big")).toMatchObject({ systemAsUser: true, contextLength: 11250, maxOutputTokens: 5625 });
    expect(
      svc
        .state()
        .models.filter((m) => m.chainIndex !== null)
        .map((m) => m.id),
    ).toEqual(["kl/fast", "fr/big"]);
    // The combo for other apps holds only models that honour system prompts.
    let fake = JSON.parse(readFileSync(path.join(dir, "data", "fake.json"), "utf8"));
    expect(fake.combos["rlm-free"]).toEqual({
      strategy: "priority",
      models: ["kl/fast"],
    });

    // A manual order replaces the chain; the previous non-gateway agent stays as the last resort.
    await svc.setup(["fr/big", "kl/chatty"]);
    expect(settings.roles.agent).toBe(OMNIROUTE_AUTO_MODEL_ID);
    expect(settings.omniroute.chain).toEqual(["omniroute:fr/big"]);
    expect(settings.roles.agentFallbacks).toEqual(["builtin:agent"]);
    expect(settings.models.filter((m) => m.providerId === "omniroute").map((m) => m.id)).toEqual(["omniroute:fr/big", OMNIROUTE_AUTO_MODEL_ID]);
    fake = JSON.parse(readFileSync(path.join(dir, "data", "fake.json"), "utf8"));
    expect(fake.combos["rlm-free"]).toBeUndefined();

    // The background setup after install refreshes the chain without taking the agent from the user's model.
    settings = { ...settings, roles: { ...settings.roles, agent: "builtin:agent", agentFallbacks: [] } };
    await svc.setup(["kl/fast"], false);
    expect(settings.roles.agent).toBe("builtin:agent");
    expect(settings.omniroute.chain).toEqual(["omniroute:kl/fast"]);

    // Restart reuses the stored key.
    await svc.stop();
    expect(svc.state().phase).toBe("stopped");
    await svc.start();
    expect(secrets.get("provider:omniroute")).toBe("sk-test-0");
    expect(keyPushes).toBe(1);
  }, 60_000);
});
