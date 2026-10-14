import { useEffect, useState } from "react";
import { BUILTIN, type LocalModelStatus, type ModelConfig, type ProviderConfig, type Settings } from "@rlm/contracts";
import { api, call, errText } from "../../api";
import { useApp } from "../../store";
import { IDownload, IPlus, ITrash } from "../../ui/icons";
import { Bytes, Num, Row, Section, Select, Text } from "./fields";

type Progress = { modelId: string; received: number; total: number; file: string; done: boolean; error?: string };

const PRESETS: Array<{ label: string; baseUrl: string; needsKey: boolean }> = [
  { label: "OpenAI", baseUrl: "https://api.openai.com/v1", needsKey: true },
  { label: "OpenRouter", baseUrl: "https://openrouter.ai/api/v1", needsKey: true },
  { label: "Ollama", baseUrl: "http://localhost:11434/v1", needsKey: false },
  { label: "LM Studio", baseUrl: "http://localhost:1234/v1", needsKey: false },
  { label: "vLLM", baseUrl: "http://localhost:8000/v1", needsKey: false },
  { label: "Custom", baseUrl: "", needsKey: true },
];

export function ModelsSection({ s }: { s: Settings }) {
  const { updateSettings, toast } = useApp();
  const chat = s.models.filter((m) => m.kind === "chat");
  const embed = s.models.filter((m) => m.kind === "embed");
  const setRole = (role: "agent" | "summarizer" | "embed", id: string) => void updateSettings({ roles: { [role]: id } });
  const patchModel = (id: string, p: Partial<ModelConfig>) => void updateSettings({ models: s.models.map((m) => (m.id === id ? { ...m, ...p } : m)) });

  return (
    <>
      <Section title="Roles" hint="Pick a model for each job. Any role can use the built-in local models or an OpenAI-compatible API.">
        <Row label="Agent" hint="Plans, writes REPL cells, and answers.">
          <Select value={s.roles.agent} width={260} options={chat.map((m) => [m.id, m.label])} onChange={(v) => setRole("agent", v)} />
        </Row>
        <Row label="Summarizer" hint="Writes compaction checkpoints and distills memory facts. Empty = same as agent.">
          <Select value={s.roles.summarizer} width={260} options={[["", "Same as agent"], ...chat.map((m) => [m.id, m.label] as [string, string])]} onChange={(v) => setRole("summarizer", v)} />
        </Row>
        <Row label="Embeddings" hint="Vectors for memory recall. Changing it starts a fresh memory table for that model.">
          <Select value={s.roles.embed} width={260} options={embed.map((m) => [m.id, m.label])} onChange={(v) => setRole("embed", v)} />
        </Row>
      </Section>

      <ModelManager />

      <Section title="Models" hint="Context length must be known before sending. Use Detect for servers that report it (vLLM, OpenRouter, Ollama) or enter it manually.">
        {s.models.map((m) => (
          <ModelRow key={m.id} m={m} providers={s.providers} onPatch={(p) => patchModel(m.id, p)} onRemove={() => void updateSettings({ models: s.models.filter((x) => x.id !== m.id) })} toast={toast} />
        ))}
        <AddModel s={s} />
      </Section>

      <Providers s={s} />
    </>
  );
}

function ModelManager() {
  const toast = useApp((s) => s.toast);
  const [models, setModels] = useState<LocalModelStatus[]>([]);
  const [progress, setProgress] = useState<Record<string, Progress>>({});
  const load = () => call<LocalModelStatus[]>("models:status", {}).then(setModels, () => undefined);
  useEffect(() => {
    void load();
    const off = api.onModelProgress((p) => {
      setProgress((prev) => ({ ...prev, [p.modelId]: p }));
      if (p.done) {
        if (p.error) toast(`Download failed: ${p.error}`, "error");
        void load();
      }
    });
    const t = setInterval(() => void load(), 5000);
    return () => (off(), clearInterval(t));
  }, []);

  const importDir = async (id: string) => {
    const dir = await call<string | null>("dialog:openFolder", {});
    if (!dir) return;
    try {
      await call("models:import", { modelId: id, sourceDir: dir });
      toast("Imported and verified");
      void load();
    } catch (e) {
      toast(errText(e), "error");
    }
  };

  return (
    <Section title="Built-in models" hint="Run fully offline. FrogNano 4B runs on your GPU through WebGPU, or on the CPU when no GPU is available. bge-m3 is bundled with the app.">
      {models.map((m) => {
        const p = progress[m.id];
        const downloading = p && !p.done;
        const pct = downloading ? p.received / Math.max(1, p.total) : m.bytesOnDisk / Math.max(1, m.expectedBytes);
        return (
          <div key={m.id} className="card mb-2 px-4 py-3">
            <div className="flex items-center gap-3">
              <div className="min-w-0 flex-1">
                <div className="text-[13.5px] font-semibold">{m.id === BUILTIN.agentModelId ? "FrogNano 4B" : m.id === BUILTIN.embedModelId ? "bge-m3" : m.id}</div>
                <div className="text-[11.5px] text-dim">
                  {m.id === BUILTIN.agentModelId ? "Agent LLM · q4f16 ONNX" : "Embeddings · 1024-dim"} · <Bytes n={m.expectedBytes} /> · {!m.installed ? "not installed" : m.runtime === "unavailable" ? (/not started|stopped/.test(m.detail) ? "loads on first use" : "runtime unavailable") : `runtime ${m.runtime}`}
                </div>
              </div>
              {m.installed ? (
                <span className="rounded-full border border-ok px-2.5 py-[2px] text-[10.5px] font-semibold uppercase tracking-[0.1em] text-ok">Installed</span>
              ) : (
                <>
                  <button className="btn btn-sm" onClick={() => void importDir(m.id)}>
                    Import folder
                  </button>
                  <button
                    className="btn btn-primary btn-sm"
                    disabled={!!downloading}
                    onClick={() => call("models:install", { modelId: m.id }).catch((e) => toast(errText(e), "error"))}
                  >
                    <IDownload size={12} /> {m.bytesOnDisk > 0 ? "Resume" : "Download"}
                  </button>
                </>
              )}
            </div>
            {(!m.installed || downloading) && (
              <div className="mt-3">
                <div className="h-[3px] rounded-full bg-surface-3">
                  <div className="h-full rounded-full bg-ink transition-[width]" style={{ width: `${Math.min(100, pct * 100)}%` }} />
                </div>
                <div className="mt-1 flex justify-between text-[10.5px] text-faint">
                  <span className="mono">{downloading ? p.file : ""}</span>
                  <span className="mono">
                    <Bytes n={downloading ? p.received : m.bytesOnDisk} /> / <Bytes n={m.expectedBytes} />
                  </span>
                </div>
              </div>
            )}
            {m.installed && m.detail && <div className="mt-1 text-[11px] text-faint">{m.detail}</div>}
          </div>
        );
      })}
    </Section>
  );
}

function ModelRow({ m, providers, onPatch, onRemove, toast }: { m: ModelConfig; providers: ProviderConfig[]; onPatch: (p: Partial<ModelConfig>) => void; onRemove: () => void; toast: (t: string, tone?: "info" | "error") => void }) {
  const [busy, setBusy] = useState<"" | "detect" | "test">("");
  const provider = providers.find((p) => p.id === m.providerId);
  const builtin = m.id.startsWith("builtin:");
  const detect = async () => {
    setBusy("detect");
    try {
      const l = await call<{ contextLength: number | null; maxOutputTokens: number | null; source?: string }>("models:detect", { modelId: m.id });
      toast(l.contextLength ? `Context ${l.contextLength} tokens${l.source ? ` (${l.source})` : ""}` : "The server did not report a context length; enter it manually", l.contextLength ? "info" : "error");
    } catch (e) {
      toast(errText(e), "error");
    } finally {
      setBusy("");
    }
  };
  const test = async () => {
    setBusy("test");
    try {
      const r = await call<{ ok: boolean; detail: string }>("models:test", { modelId: m.id });
      toast(`${m.label}: ${r.detail}`, r.ok ? "info" : "error");
    } catch (e) {
      toast(`${m.label}: ${errText(e)}`, "error");
    } finally {
      setBusy("");
    }
  };
  return (
    <div className="flex items-center gap-3 border-b border-line py-3 last:border-b-0" data-model={m.id}>
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2 text-[13px] font-medium">
          {m.label}
          <span className="rounded-full border border-line px-2 py-[1px] text-[10px] uppercase tracking-[0.1em] text-dim">{m.kind}</span>
        </div>
        <div className="mono truncate text-[11px] text-faint">
          {provider?.label ?? m.providerId} · {m.name}
        </div>
      </div>
      {m.kind === "chat" ? (
        <>
          <Num value={m.contextLength} onChange={(v) => onPatch({ contextLength: v })} min={512} width={90} suffix="ctx" />
          <Num value={m.maxOutputTokens} onChange={(v) => onPatch({ maxOutputTokens: v })} min={64} width={76} suffix="out" />
        </>
      ) : (
        <Num value={m.dim} onChange={(v) => onPatch({ dim: v })} min={8} width={76} suffix="dim" />
      )}
      {!builtin && m.kind === "chat" && (
        <button className="btn btn-sm" disabled={!!busy} onClick={() => void detect()}>
          {busy === "detect" ? "…" : "Detect"}
        </button>
      )}
      <button className="btn btn-sm" disabled={!!busy} onClick={() => void test()}>
        {busy === "test" ? "Testing…" : "Test"}
      </button>
      {!builtin && (
        <button className="iconbtn" title="Remove" onClick={onRemove}>
          <ITrash size={14} />
        </button>
      )}
    </div>
  );
}

function AddModel({ s }: { s: Settings }) {
  const { updateSettings, toast } = useApp();
  const apiProviders = s.providers.filter((p) => p.kind === "openai_compat");
  const [open, setOpen] = useState(false);
  const [providerId, setProviderId] = useState(apiProviders[0]?.id ?? "");
  const [name, setName] = useState("");
  const [kind, setKind] = useState<"chat" | "embed">("chat");
  if (!apiProviders.length) return <div className="pt-3 text-[12px] text-dim">Add an API provider below to use remote models.</div>;
  if (!open)
    return (
      <div className="pt-3">
        <button className="btn btn-sm" onClick={() => setOpen(true)}>
          <IPlus size={12} /> Add API model
        </button>
      </div>
    );
  const add = () => {
    const pid = providerId || apiProviders[0]!.id;
    const id = `${pid}:${name}`;
    if (s.models.some((m) => m.id === id)) return toast("That model already exists", "error");
    const m: ModelConfig = { id, providerId: pid, name, label: name, kind, contextLength: null, maxOutputTokens: null, dim: null };
    void updateSettings({ models: [...s.models, m] });
    setOpen(false);
    setName("");
  };
  return (
    <div className="card mt-3 flex flex-wrap items-center gap-2 p-3">
      <Select value={providerId || apiProviders[0]!.id} width={160} options={apiProviders.map((p) => [p.id, p.label])} onChange={setProviderId} />
      <input className="input mono" style={{ width: 240 }} placeholder="model name, e.g. gpt-4.1-mini" value={name} onChange={(e) => setName(e.target.value)} />
      <Select value={kind} width={100} options={[["chat", "chat"], ["embed", "embed"]]} onChange={setKind} />
      <button className="btn btn-primary btn-sm" disabled={!name.trim()} onClick={add}>
        Add
      </button>
      <button className="btn btn-ghost btn-sm" onClick={() => setOpen(false)}>
        Cancel
      </button>
    </div>
  );
}

function Providers({ s }: { s: Settings }) {
  const { updateSettings, toast } = useApp();
  const [preset, setPreset] = useState(0);
  const [label, setLabel] = useState("");
  const [baseUrl, setBaseUrl] = useState("");
  const [key, setKey] = useState("");
  const p = PRESETS[preset]!;

  const add = async () => {
    const id = (label || p.label).toLowerCase().replace(/[^a-z0-9]+/g, "-") + "-" + Math.random().toString(36).slice(2, 6);
    const prov: ProviderConfig = { id, kind: "openai_compat", label: label || p.label, baseUrl: (baseUrl || p.baseUrl).replace(/\/+$/, ""), hasKey: false };
    if (!prov.baseUrl) return toast("Base URL is required", "error");
    await updateSettings({ providers: [...s.providers, prov] });
    setLabel("");
    setBaseUrl("");
    setKey("");
    try {
      if (key.trim()) await call("secrets:setKey", { providerId: id, key: key.trim() });
      toast(`Added ${prov.label}. Now add a model for it above.`);
    } catch (e) {
      toast(`Added ${prov.label}, but the API key was not saved: ${errText(e)}`, "error");
    }
  };

  return (
    <Section title="API providers" hint="Any OpenAI-compatible endpoint. Keys are encrypted with the OS keychain and never leave the main process.">
      {s.providers.map((pr) => (
        <div key={pr.id} data-provider={pr.id}>
          <Row label={pr.label} hint={pr.kind === "local_onnx" ? "Runs on this machine" : pr.baseUrl}>
            {pr.kind === "openai_compat" && (
              <>
                <KeyInput providerId={pr.id} hasKey={pr.hasKey} />
                <button
                  className="iconbtn"
                  title="Remove provider"
                  onClick={() => {
                    if (!confirm(`Remove ${pr.label} and its models?`)) return;
                    void updateSettings({ providers: s.providers.filter((x) => x.id !== pr.id), models: s.models.filter((m) => m.providerId !== pr.id) });
                  }}
                >
                  <ITrash size={14} />
                </button>
              </>
            )}
          </Row>
        </div>
      ))}
      <div className="card mt-3 p-3">
        <div className="smallcaps mb-2">Add provider</div>
        <div className="mb-2 flex flex-wrap gap-1.5">
          {PRESETS.map((x, i) => (
            <button key={x.label} className={`tile ${i === preset ? "on" : ""}`} style={{ minHeight: 0, padding: "6px 12px", fontSize: 10 }} onClick={() => (setPreset(i), setBaseUrl(""))}>
              {x.label}
            </button>
          ))}
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <input className="input" style={{ width: 150 }} placeholder={p.label} value={label} onChange={(e) => setLabel(e.target.value)} />
          <input className="input mono" style={{ width: 260 }} placeholder={p.baseUrl || "https://host/v1"} value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} />
          <input className="input mono" style={{ width: 200 }} type="password" placeholder={p.needsKey ? "API key" : "API key (optional)"} value={key} onChange={(e) => setKey(e.target.value)} />
          <button className="btn btn-primary btn-sm" onClick={() => void add()}>
            Add
          </button>
        </div>
      </div>
    </Section>
  );
}

function KeyInput({ providerId, hasKey }: { providerId: string; hasKey: boolean }) {
  const toast = useApp((s) => s.toast);
  const [edit, setEdit] = useState(false);
  const save = async (key: string) => {
    try {
      await call("secrets:setKey", { providerId, key });
      toast(key ? "Key saved" : "Key removed");
    } catch (e) {
      toast(`Key not saved: ${errText(e)}`, "error");
    }
    setEdit(false);
  };
  if (!edit)
    return (
      <>
        <button className="btn btn-sm" data-testid="key-edit" onClick={() => setEdit(true)}>
          {hasKey ? "Key ••••••" : "Set key"}
        </button>
        {hasKey && (
          <button className="btn btn-ghost btn-sm" data-testid="key-remove" onClick={() => void save("")}>
            Remove key
          </button>
        )}
      </>
    );
  return (
    <Text
      type="password"
      mono
      width={220}
      value=""
      placeholder="paste API key, Enter to save"
      onChange={(v) => (v.trim() ? void save(v.trim()) : setEdit(false))}
    />
  );
}
