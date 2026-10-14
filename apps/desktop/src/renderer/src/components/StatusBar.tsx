import { useApp } from "../store";
import { IGit } from "../ui/icons";

function Dot({ tone }: { tone: "ok" | "warn" | "err" | "dim" }) {
  const c = tone === "ok" ? "var(--ok)" : tone === "warn" ? "var(--warn)" : tone === "err" ? "var(--err)" : "var(--text-faint)";
  return <span className="inline-block h-[7px] w-[7px] rounded-full" style={{ background: c }} />;
}

export function StatusBar() {
  const { status, git, workspace, settings, set, tabs, activeTab } = useApp();
  const tab = tabs.find((t) => t.id === activeTab);
  const agentModel = settings?.models.find((m) => m.id === settings.roles.agent);
  const inf = status.inference;
  const isLocal = settings?.providers.find((p) => p.id === agentModel?.providerId)?.kind === "local_onnx";
  return (
    <footer className="flex shrink-0 items-center gap-4 border-t border-line bg-bg px-3 text-[11.5px] text-dim" style={{ height: "var(--statusbar-h)" }}>
      {workspace && git.branch && (
        <span className="flex items-center gap-1.5">
          <IGit size={12} /> {git.branch}
          {git.dirty > 0 && <span className="text-faint">· {git.dirty} changed</span>}
        </span>
      )}
      {tab && tab.kind === "file" && <span className="mono text-faint">{tab.path}</span>}
      <div className="flex-1" />
      <button className="flex items-center gap-1.5 hover:text-ink" onClick={() => set({ activity: "settings" })} title="Agent model">
        <Dot tone={isLocal ? (inf?.status === "ready" ? "ok" : inf?.status === "error" ? "err" : inf ? "warn" : "dim") : "ok"} />
        {agentModel?.label ?? "no model"}
        {isLocal && inf && <span className="text-faint">· {inf.device === "none" ? inf.status : inf.device}</span>}
        {status.contextLength ? <span className="text-faint">· {Math.round(status.contextLength / 1024)}k ctx</span> : null}
      </button>
      <span className="flex items-center gap-1.5" title="Python kernel">
        <Dot tone={status.kernel === "busy" ? "warn" : status.kernel === "down" ? "err" : "ok"} /> kernel {status.kernel}
      </span>
      <button className="flex items-center gap-1.5 hover:text-ink" title="Memory" onClick={() => set({ activity: "memory" })}>
        <Dot tone={status.memory === "online" ? "ok" : status.memory === "offline" ? "err" : "dim"} /> memory {status.memory}
      </button>
    </footer>
  );
}
