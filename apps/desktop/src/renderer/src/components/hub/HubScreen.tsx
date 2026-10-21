import { useEffect, useMemo, useState } from "react";
import type { DeployTargetView, HubDeployRequest, HubDeploymentView, HubJobView, HubModelView, HubRepoView, HubStateView, Settings } from "@rlm/contracts";
import { api, call, errText } from "../../api";
import { useApp } from "../../store";
import { Circles } from "../../ui/LineArt";
import { Bytes } from "../settings/fields";

function fmtBytes(n: number): string {
  const u = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  let v = n;
  while (v >= 1024 && i < u.length - 1) (v /= 1024), i++;
  return `${v.toFixed(i >= 2 ? 1 : 0)} ${u[i]}`;
}

function fmtEta(s: number | null): string {
  if (s === null || !Number.isFinite(s)) return "—";
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${s % 60}s`;
  return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
}

function useHub(): HubStateView | null {
  const [s, set] = useState<HubStateView | null>(null);
  useEffect(() => {
    void call<HubStateView>("hub:state", {}).then(set, () => undefined);
    return api.onHubState(set);
  }, []);
  return s;
}

function Pill({ children, tone }: { children: React.ReactNode; tone?: "ok" | "warn" | "err" }) {
  const color = tone === "ok" ? "var(--ok)" : tone === "warn" ? "var(--warn)" : tone === "err" ? "var(--err)" : "var(--text-dim)";
  return (
    <span className="mono inline-flex items-center rounded-full border px-2 py-[1px] text-[10.5px]" style={{ borderColor: color, color }}>
      {children}
    </span>
  );
}

function Block({ title, hint, children, testid }: { title: string; hint?: string; children: React.ReactNode; testid?: string }) {
  return (
    <section className="mb-10" data-testid={testid}>
      <h2 className="heading m-0 text-[24px]">{title}</h2>
      {hint && <p className="mb-0 mt-1 max-w-[700px] text-[12.5px] text-dim">{hint}</p>}
      <div className="mt-4">{children}</div>
    </section>
  );
}

function Bar({ value }: { value: number }) {
  return (
    <div className="h-[6px] w-full overflow-hidden rounded-full bg-surface-2">
      <div className="h-full rounded-full bg-ink transition-[width]" style={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }} />
    </div>
  );
}

/** Models hub: download any Hugging Face repo, deploy it with llama.cpp, transformers or ONNX Runtime. */
export function HubScreen() {
  const state = useHub();
  if (!state) return <div className="h-full bg-bg" />;
  return (
    <div className="h-full overflow-y-auto bg-bg" data-testid="hub-screen">
      <div className="relative overflow-hidden border-b border-line">
        <Circles count={3} size={640} />
        <div className="relative mx-auto max-w-[1000px] px-10 pb-7 pt-11">
          <div className="smallcaps">Any Hugging Face model, served on this machine</div>
          <h1 className="heading m-0 mt-1 text-[44px] leading-none">Models hub</h1>
          <p className="mt-3 max-w-[680px] text-[13px] text-dim">
            Type a repo like <span className="mono">org/name</span>, pick its files and download. Then deploy it: as GGUF with llama.cpp (any quantization llama.cpp offers), with transformers (optionally 8/4-bit), or with ONNX Runtime. Each deployment
            becomes an API-key-protected model in Settings → Models.
          </p>
        </div>
      </div>
      <div className="mx-auto max-w-[1000px] px-10 py-8">
        <Lookup />
        <Downloads jobs={state.jobs} />
        <Library models={state.models} />
        <Deployments deployments={state.deployments} />
        <Tools tools={state.tools} />
      </div>
    </div>
  );
}

// ---- lookup + file selection -----------------------------------------------------------------

function Lookup() {
  const toast = useApp((s) => s.toast);
  const [repo, setRepo] = useState("");
  const [revision, setRevision] = useState("");
  const [info, setInfo] = useState<HubRepoView | null>(null);
  const [sel, setSel] = useState<Set<string>>(new Set());
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [filter, setFilter] = useState("");

  const look = async () => {
    if (!repo.trim()) return;
    setBusy(true);
    setError("");
    setInfo(null);
    try {
      const r = await call<HubRepoView>("hub:lookup", { repo, revision });
      setInfo(r);
      setSel(new Set(r.defaultSelection));
    } catch (e) {
      setError(errText(e));
    } finally {
      setBusy(false);
    }
  };
  const download = async () => {
    if (!info) return;
    setBusy(true);
    try {
      await call("hub:download", { repo: info.repo, revision: info.revision, files: [...sel] });
      toast(`Downloading ${info.repo}`);
      setInfo(null);
      setRepo("");
    } catch (e) {
      setError(errText(e));
    } finally {
      setBusy(false);
    }
  };
  const shown = useMemo(() => (info ? info.files.filter((f) => !filter || f.path.toLowerCase().includes(filter.toLowerCase())) : []), [info, filter]);
  const selBytes = info ? info.files.filter((f) => sel.has(f.path)).reduce((n, f) => n + f.size, 0) : 0;
  const p = info?.profile;

  return (
    <Block title="Get a model" hint="Repo id, or a huggingface.co link. Gated and private repos need a token (Tools, below)." testid="hub-lookup-block">
      <div className="flex gap-2">
        <input
          data-testid="hub-repo"
          className="input mono flex-1"
          placeholder="google/embeddinggemma-300m"
          value={repo}
          onChange={(e) => setRepo(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && void look()}
        />
        <input data-testid="hub-revision" className="input mono shrink-0" style={{ width: 160 }} placeholder="revision (main)" value={revision} onChange={(e) => setRevision(e.target.value)} />
        <button data-testid="hub-lookup" className="btn btn-primary" disabled={busy || !repo.trim()} onClick={() => void look()}>
          {busy && !info ? "Looking up…" : "Look up"}
        </button>
      </div>
      {error && (
        <div data-testid="hub-error" className="mt-3 whitespace-pre-wrap rounded-[12px] border border-[var(--err)] px-3 py-2 text-[12px] text-[var(--err)]">
          {error}
        </div>
      )}
      {info && p && (
        <div className="mt-4 rounded-[16px] border border-line bg-surface p-4" data-testid="hub-repo-card" data-repo={info.repo}>
          <div className="flex flex-wrap items-baseline gap-2">
            <a className="text-[16px] font-semibold" href={`https://huggingface.co/${info.repo}`} target="_blank" rel="noreferrer">
              {info.repo}
            </a>
            <span className="mono text-[11px] text-faint">{info.sha.slice(0, 10)}</span>
            <Pill tone={p.kind === "chat" || p.kind === "embed" ? "ok" : p.kind === "other" ? "err" : "warn"}>{p.kind}</Pill>
            {p.pipelineTag && <Pill>{p.pipelineTag}</Pill>}
            {p.libraryName && <Pill>{p.libraryName}</Pill>}
            {p.formats.map((f) => (
              <Pill key={f}>{f}</Pill>
            ))}
            {p.contextLength && <Pill>ctx {p.contextLength.toLocaleString()}</Pill>}
            {p.parameters && <Pill>{(p.parameters / 1e6).toFixed(p.parameters > 1e9 ? 0 : 1)}M params</Pill>}
            {info.gated && <Pill tone="warn">gated</Pill>}
          </div>
          <div className="mt-1 text-[12px] text-dim">
            {p.kindReason}
            {p.architectures.length ? ` · ${p.architectures.join(", ")}` : ""}
          </div>
          {p.notes.map((n) => (
            <div key={n} className="mt-1 text-[12px] text-dim">
              · {n}
            </div>
          ))}
          {info.access !== "ok" && (
            <div data-testid="hub-access" className="mt-3 rounded-[10px] border border-[var(--warn)] px-3 py-2 text-[12px]">
              {info.accessDetail}
            </div>
          )}
          <div className="mt-4 flex items-center gap-2">
            <input className="input flex-1 text-[12px]" placeholder="filter files" value={filter} onChange={(e) => setFilter(e.target.value)} />
            <button className="btn btn-sm" onClick={() => setSel(new Set([...sel, ...shown.map((f) => f.path)]))}>
              Select shown
            </button>
            <button className="btn btn-sm" onClick={() => setSel(new Set([...sel].filter((x) => !shown.some((f) => f.path === x))))}>
              Clear shown
            </button>
            <button className="btn btn-sm" onClick={() => setSel(new Set(info.defaultSelection))}>
              Default
            </button>
          </div>
          <div className="mt-2 max-h-[300px] overflow-y-auto rounded-[10px] border border-line" data-testid="hub-files">
            {shown.map((f) => (
              <label key={f.path} className="flex cursor-pointer items-center gap-2 border-b border-line px-3 py-[5px] text-[12px] last:border-b-0 hover:bg-surface-2">
                <input type="checkbox" data-file={f.path} checked={sel.has(f.path)} onChange={(e) => setSel((s) => (e.target.checked ? new Set([...s, f.path]) : new Set([...s].filter((x) => x !== f.path))))} />
                <span className="mono min-w-0 flex-1 truncate">{f.path}</span>
                <span className="mono text-faint">{fmtBytes(f.size)}</span>
              </label>
            ))}
          </div>
          <div className="mt-3 flex items-center gap-3">
            <span className="text-[12px] text-dim" data-testid="hub-selected">
              {sel.size} of {info.files.length} files · {fmtBytes(selBytes)} of {fmtBytes(info.totalBytes)}
            </span>
            <div className="flex-1" />
            <button data-testid="hub-download" className="btn btn-primary" disabled={busy || !sel.size || info.access !== "ok"} onClick={() => void download()}>
              Download {fmtBytes(selBytes)}
            </button>
          </div>
        </div>
      )}
    </Block>
  );
}

// ---- downloads ---------------------------------------------------------------------------------

function Downloads({ jobs }: { jobs: HubJobView[] }) {
  const toast = useApp((s) => s.toast);
  if (!jobs.length) return null;
  const act = (id: string, action: "pause" | "resume" | "cancel" | "remove") => call("hub:job", { id, action }).catch((e) => toast(errText(e)));
  return (
    <Block title="Downloads" testid="hub-downloads">
      <div className="flex flex-col gap-2">
        {jobs.map((j) => {
          const frac = j.bytesTotal ? j.bytesDone / j.bytesTotal : j.status === "done" ? 1 : 0;
          return (
            <div key={j.id} data-job={j.id} data-repo={j.repo} data-status={j.status} className="rounded-[14px] border border-line bg-surface px-4 py-3">
              <div className="flex items-center gap-3">
                <span className="font-semibold">{j.repo}</span>
                <Pill tone={j.status === "done" ? "ok" : j.status === "error" ? "err" : j.status === "running" ? undefined : "warn"}>{j.status}</Pill>
                <div className="flex-1" />
                <span className="mono text-[11.5px]" data-testid="job-percent">
                  {(frac * 100).toFixed(1)}%
                </span>
                {j.status === "running" && (
                  <span className="mono text-[11.5px] text-dim" data-testid="job-speed">
                    {fmtBytes(j.speedBps)}/s · ETA {fmtEta(j.etaSec)}
                  </span>
                )}
                {j.status === "running" && (
                  <button className="btn btn-sm" onClick={() => void act(j.id, "pause")}>
                    Pause
                  </button>
                )}
                {["paused", "error", "cancelled"].includes(j.status) && (
                  <button className="btn btn-sm" data-testid="job-resume" onClick={() => void act(j.id, "resume")}>
                    Resume
                  </button>
                )}
                {["running", "paused"].includes(j.status) && (
                  <button className="btn btn-sm btn-ghost" onClick={() => void act(j.id, "cancel")}>
                    Cancel
                  </button>
                )}
                {!["running"].includes(j.status) && (
                  <button className="btn btn-sm btn-ghost" onClick={() => void act(j.id, "remove")}>
                    Clear
                  </button>
                )}
              </div>
              <div className="mt-2">
                <Bar value={frac} />
              </div>
              <div className="mt-1 flex gap-3 text-[11.5px] text-dim">
                <span className="mono">
                  {fmtBytes(j.bytesDone)} / {fmtBytes(j.bytesTotal)}
                </span>
                <span>
                  {j.filesDone}/{j.files} files
                </span>
                {j.current.length > 0 && <span className="mono truncate">{j.current.join(", ")}</span>}
              </div>
              {j.error && <div className="mt-1 whitespace-pre-wrap text-[12px] text-[var(--err)]">{j.error}</div>}
            </div>
          );
        })}
      </div>
    </Block>
  );
}

// ---- library + deploy form -------------------------------------------------------------------

function Library({ models }: { models: HubModelView[] }) {
  const [open, setOpen] = useState<string | null>(null);
  const toast = useApp((s) => s.toast);
  return (
    <Block title="Library" hint="Downloaded models. Deploy one to serve it; the deploy options come from the model's own files and the tools installed." testid="hub-library">
      {!models.length && <div className="text-[12.5px] text-faint">Nothing downloaded yet.</div>}
      <div className="flex flex-col gap-2">
        {models.map((m) => (
          <div key={m.id} data-model={m.id} className="rounded-[14px] border border-line bg-surface px-4 py-3">
            <div className="flex flex-wrap items-center gap-2">
              <span className="font-semibold">{m.repo}</span>
              <Pill tone={m.profile.kind === "other" ? "err" : "ok"}>{m.profile.kind}</Pill>
              {m.profile.formats.map((f) => (
                <Pill key={f}>{f}</Pill>
              ))}
              <span className="mono text-[11.5px] text-dim">
                <Bytes n={m.bytes} /> · {m.files.length} files
              </span>
              <div className="flex-1" />
              <button className="btn btn-sm btn-primary" data-testid="model-deploy" onClick={() => setOpen(open === m.id ? null : m.id)}>
                {open === m.id ? "Close" : "Deploy…"}
              </button>
              <button
                className="btn btn-sm btn-ghost"
                data-testid="model-delete"
                onClick={() => {
                  if (confirm(`Delete ${m.repo} (${fmtBytes(m.bytes)}) from disk?`)) void call("hub:deleteModel", { modelId: m.id }).catch((e) => toast(errText(e)));
                }}
              >
                Delete
              </button>
            </div>
            <div className="mono mt-1 truncate text-[11px] text-faint">{m.dir}</div>
            {open === m.id && <DeployForm model={m} onDone={() => setOpen(null)} />}
          </div>
        ))}
      </div>
    </Block>
  );
}

function DeployForm({ model, onDone }: { model: HubModelView; onDone: () => void }) {
  const toast = useApp((s) => s.toast);
  const [targets, setTargets] = useState<DeployTargetView[] | null>(null);
  const [err, setErr] = useState("");
  const [target, setTarget] = useState<DeployTargetView["id"] | "">("");
  const [source, setSource] = useState("");
  const [quant, setQuant] = useState("");
  const [kind, setKind] = useState<"chat" | "embed">("chat");
  const [device, setDevice] = useState<"cpu" | "cuda" | "gpu">("cpu");
  const [dtype, setDtype] = useState("auto");
  const [ctx, setCtx] = useState("");
  const [remote, setRemote] = useState(false);
  const [busy, setBusy] = useState(false);

  const load = () =>
    call<DeployTargetView[]>("hub:targets", { modelId: model.id }).then((t) => {
      setTargets(t);
      const first = t.find((x) => x.available && !x.needs.length) ?? t.find((x) => x.available) ?? t[0];
      if (first && !target) pick(first);
    }, (e) => setErr(errText(e)));
  useEffect(() => void load(), [model.id]);

  const pick = (t: DeployTargetView) => {
    setTarget(t.id);
    const s = t.sources[0]?.id ?? "";
    setSource(s);
    setQuant(t.quantizations[s]?.[0]?.id ?? "");
    setKind(t.kinds[0] ?? "chat");
    setDevice(t.devices[0] ?? "cpu");
  };
  const t = targets?.find((x) => x.id === target);
  const quants = t?.quantizations[source] ?? [];

  const install = async (need: string) => {
    setBusy(true);
    try {
      if (need === "llamacpp") await call("hub:tools", { action: "downloadLlamaCpp", backend: "auto", packages: [] });
      else if (need === "python") await call("hub:tools", { action: "setupPython", packages: [], backend: "cpu" });
      else {
        const env = need.startsWith("onnx:") ? "onnx" : "main";
        await call("hub:tools", { action: "installPackages", packages: [need.replace(/^(python|onnx):/, "")], backend: "cpu", env });
      }
      await load();
    } catch (e) {
      setErr(errText(e));
    } finally {
      setBusy(false);
    }
  };

  const deploy = async () => {
    if (!t) return;
    setBusy(true);
    setErr("");
    try {
      const req: HubDeployRequest = { modelId: model.id, target: t.id, source, quant, kind, device, dtype, contextLength: Number(ctx) || 0, trustRemoteCode: remote, label: "" };
      await call("hub:deploy", req);
      toast(`Deploying ${model.repo} with ${t.label}`);
      onDone();
    } catch (e) {
      setErr(errText(e));
    } finally {
      setBusy(false);
    }
  };

  if (!targets) return <div className="mt-3 text-[12px] text-faint">{err || "Checking what this model can do…"}</div>;
  return (
    <div className="mt-3 rounded-[12px] border border-line p-3" data-testid="deploy-form">
      <div className="flex flex-wrap gap-2">
        {targets.map((x) => (
          <button
            key={x.id}
            data-target={x.id}
            className="rounded-[10px] border px-3 py-2 text-left text-[12px]"
            style={{ borderColor: target === x.id ? "var(--text)" : "var(--border)", opacity: x.available ? 1 : 0.55 }}
            onClick={() => pick(x)}
          >
            <div className="font-semibold">{x.label}</div>
            <div className="text-[11px] text-dim">{!x.available ? "not available" : x.needs.length ? `needs ${x.needs.join(", ")}` : "ready"}</div>
          </button>
        ))}
      </div>
      {t && (
        <div className="mt-3 flex flex-col gap-2 text-[12.5px]">
          {t.reason && <div className="text-dim">{t.reason}</div>}
          {t.needs.map((n) => (
            <div key={n} className="flex items-center gap-2">
              <span>
                Needs <span className="mono">{n}</span>
              </span>
              <button className="btn btn-sm" data-need={n} disabled={busy} onClick={() => void install(n)}>
                Install
              </button>
            </div>
          ))}
          {t.available && (
            <>
              <label className="flex items-center gap-3">
                <span className="w-[110px] text-dim">Source</span>
                <select className="input flex-1" data-testid="deploy-source" value={source} onChange={(e) => (setSource(e.target.value), setQuant(t.quantizations[e.target.value]?.[0]?.id ?? ""))}>
                  {t.sources.map((s) => (
                    <option key={s.id} value={s.id}>
                      {s.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="flex items-center gap-3">
                <span className="w-[110px] text-dim">Quantization</span>
                <select className="input flex-1" data-testid="deploy-quant" value={quant} onChange={(e) => setQuant(e.target.value)}>
                  {quants.map((q) => (
                    <option key={q.id} value={q.id}>
                      {q.label}
                      {q.detail ? ` — ${q.detail}` : ""}
                    </option>
                  ))}
                </select>
              </label>
              <label className="flex items-center gap-3">
                <span className="w-[110px] text-dim">Serve as</span>
                <select className="input shrink-0" style={{ width: 160 }} data-testid="deploy-kind" value={kind} onChange={(e) => setKind(e.target.value as "chat" | "embed")}>
                  {t.kinds.map((k) => (
                    <option key={k} value={k}>
                      {k === "chat" ? "chat model" : "embedding model"}
                    </option>
                  ))}
                </select>
                <span className="w-[60px] text-right text-dim">Device</span>
                <select className="input shrink-0" style={{ width: 100 }} data-testid="deploy-device" value={device} onChange={(e) => setDevice(e.target.value as "cpu" | "cuda" | "gpu")}>
                  {t.devices.map((d) => (
                    <option key={d} value={d}>
                      {d === "gpu" ? "GPU" : d}
                    </option>
                  ))}
                </select>
                {t.dtypes.length > 0 && (
                  <>
                    <span className="w-[50px] text-right text-dim">dtype</span>
                    <select className="input shrink-0" style={{ width: 110 }} value={dtype} onChange={(e) => setDtype(e.target.value)}>
                      {t.dtypes.map((d) => (
                        <option key={d}>{d}</option>
                      ))}
                    </select>
                  </>
                )}
              </label>
              <label className="flex items-center gap-3">
                <span className="w-[110px] text-dim">Context</span>
                <input className="input mono shrink-0" style={{ width: 120 }} data-testid="deploy-ctx" placeholder={String(model.profile.contextLength ?? "auto")} value={ctx} onChange={(e) => setCtx(e.target.value.replace(/\D/g, ""))} />
                <span className="text-[11px] text-faint">tokens; empty = the model's own, capped in settings</span>
              </label>
              {model.profile.remoteCode && (
                <label className="flex items-center gap-2">
                  <input type="checkbox" checked={remote} onChange={(e) => setRemote(e.target.checked)} />
                  <span>Trust remote code (runs Python shipped with the model)</span>
                </label>
              )}
              <div className="flex justify-end">
                <button className="btn btn-primary" data-testid="deploy-go" disabled={busy || !!t.needs.length || !quant || !source} onClick={() => void deploy()}>
                  Deploy
                </button>
              </div>
            </>
          )}
        </div>
      )}
      {err && <div className="mt-2 whitespace-pre-wrap text-[12px] text-[var(--err)]">{err}</div>}
    </div>
  );
}

// ---- deployments -------------------------------------------------------------------------------

function Deployments({ deployments }: { deployments: HubDeploymentView[] }) {
  const toast = useApp((s) => s.toast);
  const [log, setLog] = useState<{ id: string; text: string } | null>(null);
  const [settings, setSettings] = useState<Settings | null>(null);
  useEffect(() => {
    void call<Settings>("settings:get", {}).then(setSettings);
    return api.onSettings(setSettings);
  }, []);
  if (!deployments.length) return null;
  const act = async (id: string, action: "start" | "stop" | "remove" | "test") => {
    try {
      const r = await call<{ ok?: boolean; detail?: string } | null>("hub:deployment", { id, action });
      if (action === "test" && r) toast(`${r.ok ? "OK" : "Failed"}: ${r.detail}`);
    } catch (e) {
      toast(errText(e));
    }
  };
  const useAs = async (d: HubDeploymentView, role: "agent" | "summarizer" | "embed") => {
    if (!settings || !d.settingsModelId) return;
    await call("settings:update", { patch: { roles: { ...settings.roles, [role]: d.settingsModelId } } });
    toast(`${d.servedName} is now the ${role} model`);
  };
  return (
    <Block title="Deployments" hint="Each running deployment is an OpenAI-compatible endpoint on 127.0.0.1 that only answers with its API key (kept in the OS keychain)." testid="hub-deployments">
      <div className="flex flex-col gap-2">
        {deployments.map((d) => {
          const roles = settings ? (Object.entries(settings.roles).filter(([, v]) => v === d.settingsModelId).map(([k]) => k) as string[]) : [];
          return (
            <div key={d.id} data-deployment={d.id} data-status={d.status} data-served={d.servedName} className="rounded-[14px] border border-line bg-surface px-4 py-3">
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-semibold">{d.servedName}</span>
                <Pill tone={d.status === "running" ? "ok" : d.status === "error" ? "err" : "warn"}>{d.status}</Pill>
                <Pill>{d.request.target}</Pill>
                <Pill>{d.request.kind}</Pill>
                <Pill>{d.request.device}</Pill>
                {d.contextLength && <Pill>ctx {d.contextLength}</Pill>}
                {d.dim && <Pill>{d.dim}-dim</Pill>}
                {roles.map((r) => (
                  <Pill key={r} tone="ok">
                    {r}
                  </Pill>
                ))}
                <div className="flex-1" />
                {d.status === "running" && d.request.kind === "chat" && (
                  <>
                    <button className="btn btn-sm" data-role="agent" onClick={() => void useAs(d, "agent")}>
                      Use as agent
                    </button>
                    <button className="btn btn-sm" data-role="summarizer" onClick={() => void useAs(d, "summarizer")}>
                      Summarizer
                    </button>
                  </>
                )}
                {d.status === "running" && d.request.kind === "embed" && (
                  <button className="btn btn-sm" data-role="embed" onClick={() => void useAs(d, "embed")}>
                    Use for memory
                  </button>
                )}
                {d.status === "running" && (
                  <button className="btn btn-sm" data-action="test" onClick={() => void act(d.id, "test")}>
                    Test
                  </button>
                )}
                {(d.status === "running" || d.status === "starting") && (
                  <button className="btn btn-sm" data-action="stop" onClick={() => void act(d.id, "stop")}>
                    Stop
                  </button>
                )}
                {(d.status === "stopped" || d.status === "error") && (
                  <button className="btn btn-sm" data-action="start" onClick={() => void act(d.id, "start")}>
                    Start
                  </button>
                )}
                <button className="btn btn-sm btn-ghost" onClick={() => void call<string>("hub:log", { id: d.id }).then((text) => setLog(log?.id === d.id ? null : { id: d.id, text }))}>
                  Log
                </button>
                <button className="btn btn-sm btn-ghost" data-action="remove" onClick={() => void act(d.id, "remove")}>
                  Remove
                </button>
              </div>
              <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[11.5px]">
                {d.steps.map((s, i) => (
                  <span key={i} className="flex items-center gap-1" title={s.detail}>
                    <span style={{ color: s.status === "done" ? "var(--ok)" : s.status === "error" ? "var(--err)" : s.status === "running" ? "var(--text)" : "var(--text-faint)" }}>
                      {s.status === "done" ? "✓" : s.status === "error" ? "✕" : s.status === "running" ? "●" : s.status === "skipped" ? "↷" : "○"}
                    </span>
                    {s.label}
                    {s.status === "running" && s.detail && <span className="mono max-w-[340px] truncate text-faint">{s.detail}</span>}
                    {s.status === "done" && s.detail && <span className="max-w-[420px] truncate text-faint">{s.detail}</span>}
                  </span>
                ))}
              </div>
              {d.baseUrl && d.status === "running" && <div className="mono mt-1 text-[11px] text-dim">{d.baseUrl} · model "{d.servedName}" · in Settings → Models</div>}
              {d.error && <div className="mono mt-2 max-h-[160px] overflow-y-auto whitespace-pre-wrap text-[11px] text-[var(--err)]">{d.error}</div>}
              {log?.id === d.id && <pre className="mono mt-2 max-h-[300px] overflow-auto rounded-[8px] bg-surface-2 p-2 text-[10.5px]">{log.text || d.logTail}</pre>}
            </div>
          );
        })}
      </div>
    </Block>
  );
}

// ---- tools -------------------------------------------------------------------------------------

function Tools({ tools }: { tools: HubStateView["tools"] }) {
  const toast = useApp((s) => s.toast);
  const [token, setToken] = useState("");
  const [pkgs, setPkgs] = useState("");
  const run = (payload: { action: "probe" | "setupPython" | "installPackages" | "installLlamaCpp" | "downloadLlamaCpp"; packages?: string[]; backend?: "cpu" | "cuda" | "gpu" | "auto"; env?: "main" | "onnx" }) =>
    call("hub:tools", { packages: [], backend: "cpu", env: "main", ...payload }).catch((e) => toast(errText(e)));
  const py = tools.python;
  const ox = tools.onnxPython;
  const l = tools.llamaCpp;
  return (
    <Block title="Tools" hint="What deployments run with. Everything is detected from what is installed; the IDE can set up its own Python environment and build llama.cpp." testid="hub-tools">
      <div className="flex flex-col gap-3 text-[12.5px]">
        <div className="rounded-[14px] border border-line bg-surface px-4 py-3">
          <div className="flex items-center gap-2">
            <span className="font-semibold">Hugging Face token</span>
            <Pill tone={tools.hfToken ? "ok" : undefined}>{tools.hfToken ? "set" : "not set"}</Pill>
            <div className="flex-1" />
            <input className="input mono shrink-0" style={{ width: 280 }} type="password" data-testid="hf-token" placeholder="hf_…" value={token} onChange={(e) => setToken(e.target.value)} />
            <button className="btn btn-sm" disabled={!token} onClick={() => void call("hub:setToken", { token }).then(() => (setToken(""), toast("Token saved in the OS keychain")))}>
              Save
            </button>
            {tools.hfToken && (
              <button className="btn btn-sm btn-ghost" onClick={() => void call("hub:setToken", { token: "" })}>
                Remove
              </button>
            )}
          </div>
          <div className="mt-1 text-[11.5px] text-dim">Needed for gated and private repos. Sent only to huggingface.co.</div>
        </div>

        <div className="rounded-[14px] border border-line bg-surface px-4 py-3" data-testid="tools-python">
          <div className="flex items-center gap-2">
            <span className="font-semibold">Python</span>
            {py ? <Pill tone="ok">{py.version}</Pill> : tools.probing ? <Pill>checking…</Pill> : <Pill tone="err">not found</Pill>}
            {py?.own && <Pill>IDE environment</Pill>}
            {py?.cuda && <Pill tone="ok">CUDA{py.gpu ? ` · ${py.gpu.name} · ${py.gpu.freeMb} MB free` : ""}</Pill>}
            <div className="flex-1" />
            <button className="btn btn-sm" onClick={() => void run({ action: "probe" })}>
              Re-check
            </button>
            {!py?.own && (
              <button className="btn btn-sm" onClick={() => void run({ action: "setupPython" })}>
                Create IDE environment
              </button>
            )}
          </div>
          {py && <div className="mono mt-1 truncate text-[11px] text-faint">{py.path}</div>}
          {tools.pythonError && <div className="mt-1 text-[11.5px] text-[var(--err)]">{tools.pythonError}</div>}
          {py && (
            <div className="mt-2 flex flex-wrap gap-1.5">
              {Object.entries(py.packages).map(([k, v]) => (
                <Pill key={k} tone={v ? "ok" : undefined}>
                  {k} {v ?? "—"}
                </Pill>
              ))}
            </div>
          )}
          {py && (
            <div className="mt-2 text-[11.5px] text-dim">
              transformers knows {py.transformersModelTypes} model types · optimum exports {py.onnxExportModelTypes} to ONNX · ONNX quantizers: {py.onnxQuantizers.join(", ") || "none"}
            </div>
          )}
          <div className="mt-2 flex gap-2">
            <input className="input mono flex-1" placeholder="pip packages, e.g. bitsandbytes" value={pkgs} onChange={(e) => setPkgs(e.target.value)} />
            <button className="btn btn-sm" disabled={!pkgs.trim()} onClick={() => void run({ action: "installPackages", packages: pkgs.split(/\s+/).filter(Boolean) }).then(() => setPkgs(""))}>
              pip install
            </button>
          </div>
        </div>

        <div className="rounded-[14px] border border-line bg-surface px-4 py-3" data-testid="tools-onnx">
          <div className="flex items-center gap-2">
            <span className="font-semibold">ONNX environment</span>
            {ox ? <Pill tone="ok">{ox.packages["optimum-onnx"] ?? ox.packages.optimum ? "ready" : "no optimum"}</Pill> : <Pill>shares Python</Pill>}
            <div className="flex-1" />
            {!ox?.packages["optimum-onnx"] && (
              <button className="btn btn-sm" data-testid="tools-onnx-install" onClick={() => void run({ action: "installPackages", packages: ["optimum-onnx[onnxruntime]"], env: "onnx" })}>
                Set up
              </button>
            )}
          </div>
          <div className="mt-1 text-[11.5px] text-dim">
            A separate environment, because optimum pins its own transformers version.
            {ox && ` optimum exports ${ox.onnxExportModelTypes} model types · quantizers: ${ox.onnxQuantizers.join(", ") || "none"}`}
          </div>
        </div>

        <div className="rounded-[14px] border border-line bg-surface px-4 py-3" data-testid="tools-llama">
          <div className="flex items-center gap-2">
            <span className="font-semibold">llama.cpp</span>
            {l.server ? <Pill tone="ok">{l.backend}</Pill> : <Pill>not installed</Pill>}
            {l.version && <span className="mono text-[11px] text-faint">{l.version}</span>}
            <div className="flex-1" />
            <button
              className="btn btn-sm btn-primary"
              data-testid="llama-download"
              title="Prebuilt for this computer: CUDA on NVIDIA GPUs, Vulkan on other GPUs, Metal on Macs"
              disabled={!!tools.installing}
              onClick={() => void run({ action: "downloadLlamaCpp", backend: "auto" })}
            >
              {l.server ? "Update" : "Download"}
            </button>
            <button className="btn btn-sm" data-testid="llama-download-cpu" disabled={!!tools.installing} onClick={() => void run({ action: "downloadLlamaCpp", backend: "cpu" })}>
              CPU only
            </button>
            <details className="relative">
              <summary className="btn btn-sm cursor-pointer list-none">Build from source</summary>
              <div className="absolute right-0 z-10 mt-1 flex gap-1 rounded-[10px] border border-line bg-surface p-2 shadow">
                <button className="btn btn-sm" data-testid="llama-build-cpu" disabled={!!tools.installing} onClick={() => void run({ action: "installLlamaCpp", backend: "cpu" })}>
                  CPU
                </button>
                {tools.cudaToolkit && (
                  <button className="btn btn-sm" disabled={!!tools.installing} onClick={() => void run({ action: "installLlamaCpp", backend: "cuda" })}>
                    CUDA
                  </button>
                )}
              </div>
            </details>
          </div>
          {l.dir && <div className="mono mt-1 truncate text-[11px] text-faint">{l.dir}</div>}
          {l.error && <div className="mt-1 text-[11.5px] text-err">{l.error}</div>}
          {l.server && (
            <div className="mt-1 text-[11.5px] text-dim">
              {l.quantTypes.length} quantization types · converter knows {l.convertArchitectures} architectures
            </div>
          )}
          {l.quantTypes.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-1">
              {l.quantTypes.map((q) => (
                <span key={q.id} title={q.detail} className="mono rounded-[6px] bg-surface-2 px-1.5 text-[10.5px]">
                  {q.id}
                </span>
              ))}
            </div>
          )}
        </div>

        {tools.installing && (
          <div className="rounded-[14px] border border-line bg-surface px-4 py-3" data-testid="tools-installing">
            <div className="font-semibold">Installing {tools.installing.tool}…</div>
            <pre className="mono mt-2 max-h-[220px] overflow-auto whitespace-pre-wrap text-[10.5px] text-dim">{tools.installing.log.split("\n").slice(-40).join("\n")}</pre>
          </div>
        )}
      </div>
    </Block>
  );
}
