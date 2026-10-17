import { memo, useState } from "react";
import { call, errText } from "../../api";
import type { Block } from "../../chat/model";
import { useApp } from "../../store";
import { Markdown } from "../../ui/Markdown";
import { IBrain, ICheck, IChevron, ICode, IEdit, ISpark, IWrench } from "../../ui/icons";

type ToolBlock = Extract<Block, { kind: "tool" }>;
export type RenderBlock = Block | { kind: "cellGroup"; cell: Extract<Block, { kind: "cell" }>; tools: ToolBlock[] };

/** Attach the tool calls a cell made (they stream right after it, same agent) to that cell. */
export function groupBlocks(blocks: Block[]): RenderBlock[] {
  const out: RenderBlock[] = [];
  for (const b of blocks) {
    const prev = out[out.length - 1];
    if (b.kind === "cell") out.push({ kind: "cellGroup", cell: b, tools: [] });
    else if (b.kind === "tool" && prev?.kind === "cellGroup" && prev.cell.agentPath.join("/") === b.agentPath.join("/")) prev.tools.push(b);
    else out.push(b);
  }
  return out;
}

export const BlockView = memo(function BlockView({ block, running, last }: { block: RenderBlock; running: boolean; last: boolean }) {
  switch (block.kind) {
    case "cellGroup":
      return <Cell b={block.cell} tools={block.tools} />;
    case "text":
      return <Markdown text={block.text} />;
    case "thinking":
      return <Thinking text={block.text} live={running && last} />;
    case "status":
      return <div className="pulse text-[12px] text-dim">{block.text}</div>;
    case "switch":
      return (
        <div className="flex items-center gap-1.5 text-[11.5px] text-faint" data-testid="model-switch">
          <ISpark size={11} className="shrink-0" />
          <span className="min-w-0 truncate">
            Switched to <span className="text-dim">{block.to}</span> · {block.from}: {block.reason}
            {block.resumed ? " · continued the reply" : ""}
          </span>
        </div>
      );
    case "cell":
      return <Cell b={block} />;
    case "tool":
      return <Tool b={block} />;
    case "approval":
      return <Approval b={block} />;
    case "compaction":
      return <Compaction b={block} />;
    case "memory":
      return <Memory b={block} />;
    case "subagent":
      return <SubAgent b={block} />;
    case "edit":
      return <Edit b={block} />;
    case "error":
      return (
        <div className="selectable rounded-[10px] border px-3 py-2 text-[12.5px]" style={{ borderColor: "var(--err)", color: "var(--err)" }}>
          {block.text}
        </div>
      );
  }
});

function Disclosure({ label, icon, right, children, defaultOpen = false, tone }: { label: React.ReactNode; icon?: React.ReactNode; right?: React.ReactNode; children: React.ReactNode; defaultOpen?: boolean; tone?: "err" | "ok" }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="overflow-hidden rounded-[12px] border border-line bg-surface">
      <button className="flex w-full items-center gap-2 px-3 py-[7px] text-left text-[12px]" onClick={() => setOpen(!open)}>
        <IChevron size={12} className="shrink-0 text-faint" style={{ transform: open ? "rotate(90deg)" : undefined, transition: "transform .12s" }} />
        {icon && <span className="shrink-0 text-dim">{icon}</span>}
        <span className="min-w-0 flex-1 truncate" style={{ color: tone === "err" ? "var(--err)" : undefined }}>
          {label}
        </span>
        {right}
      </button>
      {open && <div className="border-t border-line">{children}</div>}
    </div>
  );
}

function Thinking({ text, live }: { text: string; live: boolean }) {
  const [open, setOpen] = useState(false);
  return (
    <div>
      <button className="flex items-center gap-1.5 text-[11.5px] text-faint hover:text-dim" onClick={() => setOpen(!open)}>
        <IChevron size={11} style={{ transform: open ? "rotate(90deg)" : undefined }} />
        <span className={live ? "pulse" : ""}>{live ? "Thinking…" : "Thought"}</span>
        <span className="text-faint">· {Math.max(1, Math.round(text.length / 4))} tokens</span>
      </button>
      {open && <div className="selectable mt-1 whitespace-pre-wrap border-l border-line-strong pl-3 text-[12px] italic text-faint">{text.trim()}</div>}
    </div>
  );
}

function Depth({ path, children }: { path: string[]; children: React.ReactNode }) {
  return <div style={{ marginLeft: Math.max(0, path.length - 1) * 14 }}>{children}</div>;
}

function Cell({ b, tools = [] }: { b: Extract<Block, { kind: "cell" }>; tools?: ToolBlock[] }) {
  const running = b.ok === undefined;
  const toolFailed = tools.some((t) => t.ok === false);
  const status = running ? (
    <span className="pulse text-[11px] text-dim">running</span>
  ) : b.ok && !toolFailed ? (
    <ICheck size={13} className="text-ok" />
  ) : (
    <span className="text-[11px] text-err">{b.ok ? "tool error" : "failed"}</span>
  );
  const firstLine = b.code.split("\n").find((l) => l.trim() && !l.trim().startsWith("#")) ?? b.code.split("\n")[0] ?? "";
  const label = tools.length ? (
    <span className="flex min-w-0 items-center gap-1.5 overflow-hidden">
      {tools.slice(0, 3).map((t) => (
        <span key={t.call.callId} className="flex min-w-0 items-center gap-1">
          <span className="font-semibold">{t.call.toolId}</span>
          <span className="mono truncate text-[11px] text-dim">{argSummary(t.call.args).slice(0, 48)}</span>
        </span>
      ))}
      {tools.length > 3 && <span className="text-[11px] text-faint">+{tools.length - 3}</span>}
    </span>
  ) : (
    <span className="mono text-[11.5px]">{firstLine.trim().slice(0, 90)}</span>
  );
  return (
    <Depth path={b.agentPath}>
      <Disclosure
        icon={tools.length ? <IWrench size={13} /> : <ICode size={13} />}
        label={label}
        tone={b.ok === false || toolFailed ? "err" : undefined}
        right={
          <span className="flex shrink-0 items-center gap-2 text-[11px] text-faint">
            {b.agentPath.length > 1 && <span>{b.agentPath[b.agentPath.length - 1]}</span>}
            {b.durationMs !== undefined && <span>{(b.durationMs / 1000).toFixed(1)}s</span>}
            {status}
          </span>
        }
        defaultOpen={running}
      >
        <pre className="selectable mono m-0 max-h-[280px] overflow-auto bg-surface-2 px-3 py-2 text-[11.5px] leading-[1.55]">{b.code}</pre>
        {tools.length > 0 && (
          <div className="flex flex-col gap-1 border-b border-line px-2 py-2">
            {tools.map((t) => (
              <Tool key={t.call.callId} b={t} />
            ))}
          </div>
        )}
        {(b.output || b.error) && (
          <pre className="selectable mono m-0 max-h-[220px] overflow-auto px-3 py-2 text-[11.5px] leading-[1.55] text-dim">
            {b.output}
            {b.error && <span className="text-err">{(b.output ? "\n" : "") + b.error}</span>}
          </pre>
        )}
      </Disclosure>
    </Depth>
  );
}

function argSummary(args: Record<string, unknown>): string {
  const v = args.path ?? args.pattern ?? args.command ?? args.query ?? args.message ?? Object.values(args)[0];
  return typeof v === "string" ? v : v === undefined ? "" : JSON.stringify(v);
}

function Tool({ b }: { b: Extract<Block, { kind: "tool" }> }) {
  const done = b.ok !== undefined;
  const result = typeof b.result === "string" ? b.result : JSON.stringify(b.result, null, 2);
  const image = b.ok && b.result && typeof b.result === "object" ? (b.result as { image?: unknown }).image : undefined;
  const shot = typeof image === "string" && /^rlm-attach:\/\/img\/[\w-]+\.png$/.test(image) ? image : null;
  return (
    <Depth path={b.agentPath}>
      {shot && <img src={shot} alt="Browser screenshot" className="md-img mb-1" />}
      <Disclosure
        icon={<IWrench size={13} />}
        tone={b.ok === false ? "err" : undefined}
        label={
          <span>
            <span className="font-semibold">{b.call.toolId}</span> <span className="mono text-[11px] text-dim">{argSummary(b.call.args).slice(0, 80)}</span>
          </span>
        }
        right={
          <span className="flex shrink-0 items-center gap-2 text-[11px] text-faint">
            {b.deduped && <span>cached</span>}
            {b.durationMs !== undefined && <span>{b.durationMs}ms</span>}
            {!done ? <span className="pulse">…</span> : b.ok ? <ICheck size={13} className="text-ok" /> : <span className="text-err">error</span>}
          </span>
        }
      >
        <pre className="selectable mono m-0 max-h-[120px] overflow-auto bg-surface-2 px-3 py-2 text-[11px]">{JSON.stringify(b.call.args, null, 2)}</pre>
        {done && <pre className="selectable mono m-0 max-h-[240px] overflow-auto px-3 py-2 text-[11px] text-dim">{(result ?? "").slice(0, 20000)}</pre>}
      </Disclosure>
    </Depth>
  );
}

function Approval({ b }: { b: Extract<Block, { kind: "approval" }> }) {
  const toast = useApp((s) => s.toast);
  const [busy, setBusy] = useState(false);
  const answer = async (approved: boolean, remember: "once" | "session" | "workspace" = "once") => {
    setBusy(true);
    try {
      await call("chat:approve", { approvalId: b.approvalId, approved, remember });
    } catch (e) {
      toast(errText(e), "error");
      setBusy(false);
    }
  };
  return (
    <div className="rounded-[14px] border-[1.5px] p-3" style={{ borderColor: b.resolved ? "var(--border)" : "var(--text)" }}>
      <div className="flex items-center gap-2">
        <span className="smallcaps" style={{ color: "var(--text)" }}>
          Approval
        </span>
        <span className="font-semibold">{b.call.toolId}</span>
        <span className="mono truncate text-[11px] text-dim">{argSummary(b.call.args).slice(0, 80)}</span>
      </div>
      {b.preview && <pre className="selectable mono mt-2 max-h-[260px] overflow-auto rounded-[8px] bg-surface-2 px-3 py-2 text-[11px] leading-[1.5]">{colorDiff(b.preview)}</pre>}
      {b.resolved ? (
        <div className="mt-2 text-[12px]" style={{ color: b.approved ? "var(--ok)" : "var(--err)" }}>
          {b.approved ? "Allowed" : "Denied"}
        </div>
      ) : (
        <div className="mt-3 flex flex-wrap gap-2">
          <button className="btn btn-primary btn-sm" disabled={busy} onClick={() => void answer(true)}>
            Allow once
          </button>
          <button className="btn btn-sm" disabled={busy} onClick={() => void answer(true, "session")}>
            This chat
          </button>
          <button className="btn btn-sm" disabled={busy} onClick={() => void answer(true, "workspace")}>
            Always here
          </button>
          <button className="btn btn-sm btn-danger ml-auto" disabled={busy} onClick={() => void answer(false)}>
            Deny
          </button>
        </div>
      )}
    </div>
  );
}

function colorDiff(text: string) {
  return text.split("\n").map((l, i) => (
    <div key={i} style={{ background: l.startsWith("+") && !l.startsWith("+++") ? "var(--diff-add)" : l.startsWith("-") && !l.startsWith("---") ? "var(--diff-del)" : undefined }}>
      {l || " "}
    </div>
  ));
}

function Compaction({ b }: { b: Extract<Block, { kind: "compaction" }> }) {
  const [open, setOpen] = useState(false);
  const what = b.checkpoint.compactedBy === "session" ? "Earlier conversation summarized" : "Working notes compacted";
  return (
    <div className="my-1">
      <button className="flex w-full items-center gap-3 text-[11px] text-faint" onClick={() => setOpen(!open)}>
        <span className="h-px flex-1 bg-line" />
        <span className="smallcaps">
          {what}
          {b.checkpoint.foldedTurns ? ` · ${b.checkpoint.foldedTurns} folded` : ""}
        </span>
        <span className="h-px flex-1 bg-line" />
      </button>
      {open && (
        <div className="card mt-2 max-h-[300px] overflow-auto px-3 py-2">
          <Markdown text={b.checkpoint.summary} />
        </div>
      )}
    </div>
  );
}

function Memory({ b }: { b: Extract<Block, { kind: "memory" }> }) {
  if (b.offline)
    return (
      <div className="flex items-center gap-1.5 text-[11.5px] text-warn">
        <IBrain size={13} /> Memory offline — answering without recall
      </div>
    );
  return (
    <Disclosure icon={<IBrain size={13} />} label={<span className="text-dim">Recalled {b.items.length} memor{b.items.length === 1 ? "y" : "ies"}</span>}>
      <div className="px-3 py-2">
        {b.items.map((m) => (
          <div key={m.id} className="flex gap-2 py-[3px] text-[12px]">
            <span className="smallcaps w-[60px] shrink-0">{m.memoryType === "long_term" ? "fact" : "recent"}</span>
            <span className="selectable flex-1">{m.text}</span>
            <span className="mono text-[10.5px] text-faint">{m.score.toFixed(2)}</span>
          </div>
        ))}
      </div>
    </Disclosure>
  );
}

function SubAgent({ b }: { b: Extract<Block, { kind: "subagent" }> }) {
  return (
    <div style={{ marginLeft: Math.max(0, b.parentPath.length - 1) * 14 }}>
      <Disclosure
        icon={<ISpark size={13} />}
        tone={b.done && !b.ok ? "err" : undefined}
        label={
          <span>
            <span className="font-semibold">Sub-agent</span> <span className="text-dim">{b.task.slice(0, 90)}</span>
          </span>
        }
        right={<span className="text-[11px]">{!b.done ? <span className="pulse text-dim">working</span> : b.ok ? <ICheck size={13} className="text-ok" /> : <span className="text-err">failed</span>}</span>}
      >
        <div className="px-3 py-2">
          <div className="smallcaps mb-1">Task</div>
          <div className="selectable whitespace-pre-wrap text-[12px]">{b.task}</div>
          {b.result && (
            <>
              <div className="smallcaps mb-1 mt-2">Result</div>
              <Markdown text={b.result} />
            </>
          )}
        </div>
      </Disclosure>
    </div>
  );
}

function Edit({ b }: { b: Extract<Block, { kind: "edit" }> }) {
  const openDiff = useApp((s) => s.openDiff);
  const added = b.after.split("\n").length - (b.before ? b.before.split("\n").length : 0);
  return (
    <div className="flex items-center gap-2 rounded-[12px] border border-line bg-surface px-3 py-[7px] text-[12px]">
      <IEdit size={13} className="text-dim" />
      <span className="mono truncate">{b.path}</span>
      <span className="text-[11px] text-faint">{b.before ? `${added >= 0 ? "+" : ""}${added} lines` : "new file"}</span>
      <div className="flex-1" />
      <button className="btn btn-sm" onClick={() => openDiff(b.path, b.before, b.after, b.editId)}>
        Review
      </button>
    </div>
  );
}
