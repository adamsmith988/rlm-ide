import { useEffect, useState } from "react";
import type { MemoryFactView } from "@rlm/contracts";
import { call, errText } from "../api";
import { useApp } from "../store";
import { Circles } from "../ui/LineArt";
import { IBrain, IEdit, IPlus, ITrash } from "../ui/icons";

interface Stats {
  facts: number;
  chats: number;
  collections: string[];
}

/** Memory "profile" screen: what the agent knows about you, editable. */
export function MemoryScreen() {
  const { toast, status, settings } = useApp();
  const [q, setQ] = useState("");
  const [facts, setFacts] = useState<MemoryFactView[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);
  const [adding, setAdding] = useState(false);
  const [editing, setEditing] = useState<string | null>(null);

  const load = async () => {
    try {
      setFacts(await call<MemoryFactView[]>("memory:list", { query: q }));
      setStats(await call<Stats>("memory:stats", {}));
    } catch (e) {
      toast(errText(e), "error");
    }
  };
  useEffect(() => {
    const t = setTimeout(() => void load(), q ? 250 : 0);
    return () => clearTimeout(t);
  }, [q]);

  const run = async (fn: () => Promise<unknown>, ok?: string) => {
    try {
      await fn();
      if (ok) toast(ok);
      await load();
    } catch (e) {
      toast(errText(e), "error");
    }
  };

  const exportAll = async () => {
    const json = await call<string>("memory:export", {});
    const url = URL.createObjectURL(new Blob([json], { type: "application/json" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = "rlm-memory.json";
    a.click();
    URL.revokeObjectURL(url);
  };

  const importFile = () => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "application/json,.json";
    input.onchange = async () => {
      const f = input.files?.[0];
      if (!f) return;
      const json = await f.text();
      await run(() => call("memory:import", { json }), "Imported");
    };
    input.click();
  };

  return (
    <div className="h-full overflow-y-auto">
      <div className="relative overflow-hidden border-b border-line bg-bg">
        <Circles count={3} size={700} />
        <div className="relative mx-auto flex max-w-[860px] items-end gap-8 px-10 pb-8 pt-12">
          <div className="flex h-[96px] w-[96px] shrink-0 items-center justify-center rounded-full border-[1.5px] border-ink">
            <IBrain size={40} />
          </div>
          <div className="flex-1">
            <div className="smallcaps">Long-term memory · {settings?.memory.longTermScope === "workspace" ? "per project" : "all projects"}</div>
            <h1 className="heading m-0 mt-1 text-[44px] leading-none">What I remember</h1>
            <div className="mt-4 flex gap-8 text-[12px] text-dim">
              <Stat n={stats?.facts ?? 0} label="facts" />
              <Stat n={stats?.chats ?? 0} label="chats" />
              <div>
                <div className="text-[20px] font-semibold capitalize text-ink">{status.memory}</div>
                <div className="smallcaps">status</div>
              </div>
            </div>
          </div>
          <div className="flex flex-col gap-2">
            <button className="btn btn-primary" onClick={() => setAdding(true)}>
              <IPlus size={14} /> Add memory
            </button>
            <div className="flex gap-2">
              <button className="btn btn-sm" onClick={() => void exportAll()}>
                Export
              </button>
              <button className="btn btn-sm" onClick={importFile}>
                Import
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="mx-auto max-w-[860px] px-10 py-6">
        <div className="mb-4 flex gap-3">
          <input className="input flex-1" placeholder="Search memories" value={q} onChange={(e) => setQ(e.target.value)} />
          <button
            className="btn btn-sm btn-danger"
            onClick={() => {
              if (confirm("Forget everything? This deletes all long-term facts and chat memory.")) void run(() => call("memory:forgetAll", {}), "Memory cleared");
            }}
          >
            Forget all
          </button>
        </div>

        {adding && (
          <FactEditor
            initial=""
            onCancel={() => setAdding(false)}
            onSave={(text) => {
              setAdding(false);
              void run(() => call("memory:add", { text, category: "preference" }), "Saved");
            }}
          />
        )}

        <div className="flex flex-col gap-2">
          {facts.map((f) =>
            editing === f.id ? (
              <FactEditor
                key={f.id}
                initial={f.text}
                onCancel={() => setEditing(null)}
                onSave={(text) => {
                  setEditing(null);
                  void run(() => call("memory:update", { id: f.id, text }));
                }}
              />
            ) : (
              <div key={f.id} className="card group flex items-start gap-4 px-4 py-3">
                <span className="smallcaps mt-[3px] w-[86px] shrink-0">{f.category}</span>
                <div className="min-w-0 flex-1">
                  <div className="selectable text-[13.5px]">{f.text}</div>
                  <div className="mt-1 text-[11px] text-faint">
                    {new Date(f.updatedAt).toLocaleDateString()}
                    {f.versions > 0 && ` · edited ${f.versions}×`}
                    {f.workspaceId && ` · ${f.workspaceId.split(/[\\/]/).pop()}`}
                  </div>
                </div>
                <div className="flex gap-1 opacity-0 transition-opacity group-hover:opacity-100">
                  <button className="iconbtn" title="Edit" onClick={() => setEditing(f.id)}>
                    <IEdit size={14} />
                  </button>
                  <button className="iconbtn" title="Delete" onClick={() => void run(() => call("memory:delete", { id: f.id }))}>
                    <ITrash size={14} />
                  </button>
                </div>
              </div>
            ),
          )}
          {!facts.length && (
            <div className="py-10 text-center text-dim">
              <div className="heading text-[22px] text-ink">{q ? "Nothing matches" : "No memories yet"}</div>
              <div className="mt-1 text-[12.5px]">Durable facts you share in chats (preferences, conventions, decisions) show up here.</div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function Stat({ n, label }: { n: number; label: string }) {
  return (
    <div>
      <div className="text-[20px] font-semibold text-ink">{n}</div>
      <div className="smallcaps">{label}</div>
    </div>
  );
}

function FactEditor({ initial, onSave, onCancel }: { initial: string; onSave: (t: string) => void; onCancel: () => void }) {
  const [t, setT] = useState(initial);
  return (
    <div className="card mb-2 p-3">
      <textarea autoFocus className="input h-[70px] resize-none" value={t} onChange={(e) => setT(e.target.value)} placeholder="e.g. I prefer pnpm and strict TypeScript" />
      <div className="mt-2 flex justify-end gap-2">
        <button className="btn btn-sm btn-ghost" onClick={onCancel}>
          Cancel
        </button>
        <button className="btn btn-sm btn-primary" disabled={!t.trim()} onClick={() => onSave(t.trim())}>
          Save
        </button>
      </div>
    </div>
  );
}
