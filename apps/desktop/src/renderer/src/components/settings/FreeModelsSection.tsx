import { useEffect, useState } from "react";
import type { OmniRouteModelView, OmniRouteView, Settings } from "@rlm/contracts";
import { api, call, errText } from "../../api";
import { useApp } from "../../store";
import { IArrowDown, IArrowUp, ITrash } from "../../ui/icons";
import { Num, Row, Section, Select, Toggle } from "./fields";

function useOmniRoute(): OmniRouteView | null {
  const [s, set] = useState<OmniRouteView | null>(null);
  useEffect(() => {
    void call<OmniRouteView>("omniroute:state", {}).then(set, () => undefined);
    return api.onOmniRouteState(set);
  }, []);
  return s;
}

function Pill({ children, tone }: { children: React.ReactNode; tone?: "ok" | "warn" | "err" }) {
  const color = tone === "ok" ? "var(--ok)" : tone === "warn" ? "var(--warn)" : tone === "err" ? "var(--err)" : "var(--text-dim)";
  return (
    <span className="mono inline-flex shrink-0 items-center rounded-full border px-2 py-[1px] text-[10.5px]" style={{ borderColor: color, color }}>
      {children}
    </span>
  );
}

const PHASE: Record<OmniRouteView["phase"], [string, "ok" | "warn" | "err" | undefined]> = {
  not_installed: ["not installed", undefined],
  installing: ["installing", "warn"],
  stopped: ["stopped", undefined],
  starting: ["starting", "warn"],
  running: ["running", "ok"],
  error: ["error", "err"],
};

function move<T>(list: T[], i: number, d: -1 | 1): T[] {
  const j = i + d;
  if (j < 0 || j >= list.length) return list;
  const out = [...list];
  [out[i], out[j]] = [out[j]!, out[i]!];
  return out;
}

/** OmniRoute, run by the IDE: one local endpoint in front of many free providers. */
export function FreeModelsSection() {
  const toast = useApp((s) => s.toast);
  const v = useOmniRoute();
  const [pending, setPending] = useState("");
  const [showLog, setShowLog] = useState(false);
  if (!v) return null;

  const act = async (label: string, fn: () => Promise<unknown>, done?: string) => {
    setPending(label);
    try {
      await fn();
      if (done) toast(done);
    } catch (e) {
      toast(errText(e), "error");
    } finally {
      setPending("");
    }
  };
  const busy = !!pending || !!v.busy;
  const [phaseLabel, tone] = PHASE[v.phase];
  const running = v.phase === "running";

  return (
    <Section
      title="Free models"
      hint="The IDE runs OmniRoute on this computer: one private endpoint in front of dozens of free model providers. Set up picks the free models that answer right now and chains them, so when one hits a rate limit, goes down or stops mid-reply, the agent carries on with the next."
    >
      <div className="card px-4 py-3" data-testid="omniroute-card">
        <div className="flex items-center gap-3">
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2 text-[13.5px] font-semibold">
              OmniRoute <Pill tone={tone}>{phaseLabel}</Pill>
              {v.version && <span className="mono text-[11px] font-normal text-faint">v{v.version}</span>}
            </div>
            <div className="mt-[2px] text-[11.5px] text-dim">{running ? `${v.baseUrl} · local only, API key required` : v.node.detail}</div>
          </div>
          {v.phase === "not_installed" && (
            <button className="btn btn-primary btn-sm" disabled={busy} data-testid="omniroute-install" onClick={() => void act("install", () => call("omniroute:install", {}), "OmniRoute installed")}>
              {v.node.ok || v.node.detail === "not checked" ? "Install" : "Retry"}
            </button>
          )}
          {(v.phase === "stopped" || v.phase === "error") && (
            <button className="btn btn-primary btn-sm" disabled={busy} data-testid="omniroute-start" onClick={() => void act("start", () => call("omniroute:service", { action: "start" }))}>
              Start
            </button>
          )}
          {running && (
            <>
              <button
                className="btn btn-sm"
                disabled={busy}
                onClick={() =>
                  void act("dash", async () => toast((await call<{ copied: boolean }>("omniroute:dashboard", {})).copied ? "Dashboard opened; its password is on your clipboard" : "Dashboard opened"))
                }
              >
                Dashboard
              </button>
              <button className="btn btn-sm" disabled={busy} onClick={() => void act("stop", () => call("omniroute:service", { action: "stop" }))}>
                Stop
              </button>
            </>
          )}
        </div>
        {(v.busy || pending) && (
          <div className="mt-2 text-[11.5px] text-dim" data-testid="omniroute-busy">
            {v.busy ?? "Working…"}
          </div>
        )}
        {v.error && <pre className="mono mt-2 max-h-[120px] overflow-auto whitespace-pre-wrap text-[11px] text-err">{v.error}</pre>}
        {v.logTail && (
          <button className="btn btn-ghost btn-sm mt-1" onClick={() => setShowLog(!showLog)}>
            {showLog ? "Hide log" : "Show log"}
          </button>
        )}
        {showLog && <pre className="mono mt-1 max-h-[180px] overflow-auto whitespace-pre-wrap text-[10.5px] text-faint">{v.logTail}</pre>}
        {v.phase === "not_installed" && (
          <div className="mt-2 text-[11.5px] text-faint">
            Needs Node.js. The first install downloads about 3 GB into the app's data folder (the gateway plus a headless browser some free providers need).
          </div>
        )}
      </div>

      {running && <FreeProviders v={v} busy={busy} act={act} />}
      {running && <FreeChain v={v} busy={busy} act={act} />}
    </Section>
  );
}

type Act = (label: string, fn: () => Promise<unknown>, done?: string) => Promise<void>;

function FreeProviders({ v, busy, act }: { v: OmniRouteView; busy: boolean; act: Act }) {
  const [keys, setKeys] = useState<Record<string, string>>({});
  const keyless = v.providers.filter((p) => p.keyless);
  const keyed = v.providers.filter((p) => !p.keyless);
  return (
    <div className="card mt-3 px-4 py-3" data-testid="omniroute-providers">
      <div className="smallcaps mb-1">Free providers</div>
      <div className="mb-2 text-[11.5px] text-dim">
        No account: {keyless.length ? keyless.map((p) => p.name).join(", ") : "none"}. These drive public web playgrounds, so they are slow, can break without notice, and their terms may not allow
        automated use. A free key from one of the providers below is faster and more dependable; each key is handed to the gateway and kept there, encrypted.
      </div>
      {keyed.map((p) => {
        const conn = v.connections.find((c) => c.provider === p.id);
        return (
          <div key={p.id} className="flex items-center gap-3 border-b border-line py-2 last:border-b-0" data-provider={p.id}>
            <div className="min-w-0 flex-1">
              <div className="text-[12.5px] font-medium">{p.name}</div>
              {p.website && <div className="mono truncate text-[10.5px] text-faint">{p.website}</div>}
            </div>
            {conn ? (
              <>
                <Pill tone={conn.active && conn.status !== "error" ? "ok" : "err"}>{conn.lastError ? "error" : "connected"}</Pill>
                <button className="iconbtn" title="Disconnect" disabled={busy} onClick={() => void act("disc", () => call("omniroute:disconnect", { connectionId: conn.id }))}>
                  <ITrash size={14} />
                </button>
              </>
            ) : (
              <>
                <input
                  className="input mono"
                  style={{ width: 200 }}
                  type="password"
                  placeholder="free API key"
                  value={keys[p.id] ?? ""}
                  onChange={(e) => setKeys({ ...keys, [p.id]: e.target.value })}
                />
                <button
                  className="btn btn-sm"
                  disabled={busy || !(keys[p.id] ?? "").trim()}
                  onClick={() =>
                    void act(
                      "conn",
                      async () => {
                        await call("omniroute:connect", {
                          provider: p.id,
                          key: keys[p.id]!.trim(),
                        });
                        setKeys({ ...keys, [p.id]: "" });
                      },
                      `${p.name} connected. Run Set up again to use it.`,
                    )
                  }
                >
                  Connect
                </button>
              </>
            )}
          </div>
        );
      })}
    </div>
  );
}

function FreeChain({ v, busy, act }: { v: OmniRouteView; busy: boolean; act: Act }) {
  const chain = v.models.filter((m) => m.chainIndex !== null).sort((a, b) => a.chainIndex! - b.chainIndex!);
  const [order, setOrder] = useState<string[]>(chain.map((m) => m.id));
  const chainKey = chain.map((m) => m.id).join(",");
  useEffect(() => setOrder(chainKey ? chainKey.split(",") : []), [chainKey]);
  const byId = new Map(v.models.map((m) => [m.id, m]));
  const others = v.models.filter((m) => !order.includes(m.id)).sort((a, b) => Number(b.ok) - Number(a.ok) || (a.latencyMs ?? 1e9) - (b.latencyMs ?? 1e9));
  const dirty = order.join(",") !== chainKey;
  return (
    <div className="card mt-3 px-4 py-3" data-testid="omniroute-chain">
      <div className="flex items-center gap-2">
        <div className="smallcaps flex-1">Free model chain</div>
        <button className="btn btn-sm" disabled={busy} onClick={() => void act("probe", () => call("omniroute:probe", { models: [] }))}>
          Test models
        </button>
        {dirty && (
          <button className="btn btn-sm" disabled={busy || !order.length} onClick={() => void act("save", () => call("omniroute:setup", { models: order }), "Chain saved")}>
            Save order
          </button>
        )}
        <button
          className="btn btn-primary btn-sm"
          disabled={busy}
          data-testid="omniroute-setup"
          onClick={() => void act("setup", () => call("omniroute:setup", { models: [] }), "OmniRoute is now the agent's model")}
        >
          {chain.length ? "Set up again" : "Set up free models"}
        </button>
      </div>
      <div className="mt-1 text-[11.5px] text-dim">
        The agent uses the first model and switches down the list whenever one fails; a reply cut off halfway is resumed on the next model from where it stopped. Every model is tested for whether it
        receives the agent's instructions and memory; models marked “instructions in message” get them inside the user message because their provider drops system prompts.
        {v.comboName && ` Other apps can use the combo “${v.comboName}” on the gateway.`}
      </div>
      {chain.length > 0 && !chain.some((m) => m.agentReady === true) && chain.some((m) => m.agentReady === false) && (
        <div className="mt-2 text-[11.5px] text-err" data-testid="omniroute-chat-only">
          None of these models reliably does its work through the agent's tools; they often answer from memory instead. Chat works, but for real agent work connect a free provider with a key
          above (Gemini, Groq, OpenRouter, Mistral…).
        </div>
      )}
      {order.map((id, i) => {
        const m = byId.get(id);
        return m ? (
          <ModelLine key={id} m={m} index={i} onUp={() => setOrder(move(order, i, -1))} onDown={() => setOrder(move(order, i, 1))} onRemove={() => setOrder(order.filter((x) => x !== id))} />
        ) : null;
      })}
      {others.length > 0 && <div className="smallcaps mb-1 mt-3">Other tested models</div>}
      {others.map((m) => (
        <ModelLine key={m.id} m={m} onAdd={() => setOrder([...order, m.id])} />
      ))}
    </div>
  );
}

function ModelLine({ m, index, onUp, onDown, onRemove, onAdd }: { m: OmniRouteModelView; index?: number; onUp?: () => void; onDown?: () => void; onRemove?: () => void; onAdd?: () => void }) {
  const result = m.ok === null ? <Pill>untested</Pill> : m.ok ? <Pill tone="ok">{((m.latencyMs ?? 0) / 1000).toFixed(1)} s</Pill> : <Pill tone="err">failed</Pill>;
  return (
    <div className="flex items-center gap-3 border-b border-line py-2 last:border-b-0" data-chain-model={m.id}>
      {index !== undefined && <span className="mono w-4 text-right text-[11px] text-faint">{index + 1}</span>}
      <div className="min-w-0 flex-1">
        <div className="truncate text-[12.5px] font-medium">{m.name}</div>
        <div className="mono truncate text-[10.5px] text-faint" title={m.error ?? undefined}>
          {m.id} · {m.contextLength ? `${Math.round(m.contextLength / 1000)}k ctx` : "ctx ?"}
          {m.toolCalling ? " · tools" : ""}
          {m.systemAsUser ? " · instructions in message" : ""}
          {m.inputLimitChars ? ` · takes ~${Math.round(m.inputLimitChars / 1000)}k chars` : ""}
          {m.agentReady === true ? " · uses its tools" : m.agentReady === false ? " · often answers without its tools" : ""}
          {m.error ? ` · ${m.error}` : ""}
        </div>
      </div>
      {result}
      {onUp && (
        <button className="iconbtn" title="Earlier" onClick={onUp}>
          <IArrowUp size={13} />
        </button>
      )}
      {onDown && (
        <button className="iconbtn" title="Later" onClick={onDown}>
          <IArrowDown size={13} />
        </button>
      )}
      {onRemove && (
        <button className="iconbtn" title="Remove from chain" onClick={onRemove}>
          <ITrash size={13} />
        </button>
      )}
      {onAdd && (
        <button className="btn btn-ghost btn-sm" disabled={m.ok === false} onClick={onAdd}>
          Add
        </button>
      )}
    </div>
  );
}

/** Works for any models, free or paid: the agent's backups and how switching behaves. */
export function FailoverSection({ s }: { s: Settings }) {
  const { updateSettings } = useApp();
  const chat = s.models.filter((m) => m.kind === "chat");
  const label = (id: string) => chat.find((m) => m.id === id)?.label ?? id;
  const fallbacks = s.roles.agentFallbacks;
  const addable = chat.filter((m) => m.id !== s.roles.agent && !fallbacks.includes(m.id));
  const setFallbacks = (list: string[]) => void updateSettings({ roles: { agentFallbacks: list } });
  const f = s.failover;
  const setF = (p: Partial<Settings["failover"]>) => void updateSettings({ failover: p });
  return (
    <Section
      title="Automatic switching"
      hint="When the agent's model fails (rate limit, outage, stalled or dropped stream, empty reply) the turn moves to the next backup instead of stopping. A reply cut off halfway is continued, not restarted."
    >
      <Row label="Switch models on failure" hint="Off: a failure ends the turn with an error.">
        <Toggle value={f.enabled} onChange={(enabled) => setF({ enabled })} />
      </Row>
      <Row label="Continue cut-off replies" hint="Hands the partial reply to the next model to finish. Off: the next model starts the reply over.">
        <Toggle value={f.continueMidReply} onChange={(continueMidReply) => setF({ continueMidReply })} />
      </Row>
      <Row label="Wait when every model is down" hint="Seconds to wait for the first model to come back before giving up (rate limits usually clear within minutes).">
        <Num value={f.waitForRecoverySec} min={0} width={80} suffix="s" onChange={(n) => setF({ waitForRecoverySec: n ?? 0 })} />
      </Row>
      <Row label="First cooldown" hint="How long a failing model is skipped; doubles on repeated failures. A Retry-After from the server wins.">
        <Num value={f.cooldownSec} min={1} width={80} suffix="s" onChange={(n) => setF({ cooldownSec: n ?? 30 })} />
      </Row>
      <div className="card mt-2 px-4 py-3" data-testid="fallback-chain">
        <div className="smallcaps mb-1">Agent backups, in order</div>
        <div className="mb-1 text-[11.5px] text-dim">After {label(s.roles.agent)}.</div>
        {fallbacks.map((id, i) => (
          <div key={id} className="flex items-center gap-3 border-b border-line py-2 last:border-b-0" data-fallback={id}>
            <span className="mono w-4 text-right text-[11px] text-faint">{i + 1}</span>
            <div className="min-w-0 flex-1 truncate text-[12.5px]">{label(id)}</div>
            <button className="iconbtn" title="Earlier" onClick={() => setFallbacks(move(fallbacks, i, -1))}>
              <IArrowUp size={13} />
            </button>
            <button className="iconbtn" title="Later" onClick={() => setFallbacks(move(fallbacks, i, 1))}>
              <IArrowDown size={13} />
            </button>
            <button className="iconbtn" title="Remove" onClick={() => setFallbacks(fallbacks.filter((x) => x !== id))}>
              <ITrash size={13} />
            </button>
          </div>
        ))}
        {!fallbacks.length && <div className="py-2 text-[12px] text-faint">No backups: a failing model ends the turn.</div>}
        {addable.length > 0 && (
          <div className="mt-2">
            <Select value="" width={260} options={[["", "Add a backup…"], ...addable.map((m) => [m.id, m.label] as [string, string])]} onChange={(id) => id && setFallbacks([...fallbacks, id])} />
          </div>
        )}
      </div>
    </Section>
  );
}
