import type { ContextUsage } from "../../chat/model";

function k(n: number): string {
  return n >= 1000 ? `${(n / 1000).toFixed(n >= 10000 ? 0 : 1)}k` : String(n);
}

/** Player-style progress bar: the prompt of the latest model call against the context window,
 * with a tick where the agent compacts its transcript. */
export function ContextMeter({ usage, contextLength }: { usage: ContextUsage | undefined; contextLength: number | null }) {
  const window = usage?.contextLength ?? contextLength ?? 0;
  const used = usage?.usedTokens ?? 0;
  const pct = window ? Math.min(1, used / window) : 0;
  const trig = window && usage ? Math.min(1, usage.triggerTokens / window) : 0.9;
  const near = pct >= trig;
  const title = usage
    ? `Prompt ${used.toLocaleString()} of ${window.toLocaleString()} tokens (${Math.round(pct * 100)}%) · compacts at ${usage.triggerTokens.toLocaleString()} · hard limit ${(usage.inputLimit ?? window).toLocaleString()}`
    : "Context usage";
  return (
    <div className="px-1" title={title} data-testid="context-meter" data-used={used} data-window={window} data-trigger={usage?.triggerTokens ?? ""}>
      <div className="relative h-[3px] rounded-full bg-surface-3">
        <div className="absolute inset-y-0 left-0 rounded-full" style={{ width: `${pct * 100}%`, background: near ? "var(--warn)" : "var(--text)" }} />
        <div className="absolute top-[-3px] h-[9px] w-px bg-faint" style={{ left: `${trig * 100}%` }} />
        <div className="absolute top-1/2 h-[11px] w-[11px] -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-bg" style={{ left: `${pct * 100}%`, background: near ? "var(--warn)" : "var(--text)" }} />
      </div>
      <div className="mt-1.5 flex justify-between text-[10.5px] text-faint">
        <span className="mono">
          {k(used)}
          {window ? ` · ${Math.round(pct * 100)}%` : ""}
        </span>
        <span className="smallcaps" style={{ fontSize: 9.5 }}>
          context
        </span>
        <span className="mono">{window ? k(window) : "—"}</span>
      </div>
    </div>
  );
}
