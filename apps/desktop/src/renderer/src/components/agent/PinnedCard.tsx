import { useEffect, useState } from "react";
import type { PinnedFactView } from "@rlm/contracts";
import { call, errText } from "../../api";
import { useApp } from "../../store";
import { IClose, IPin, IPlus } from "../../ui/icons";

/** The chat goal as a quote card (reference: the motivational quote screen), plus constraints/decisions. */
export function PinnedCard({ chatId, version }: { chatId: string; version: number }) {
  const toast = useApp((s) => s.toast);
  const [facts, setFacts] = useState<PinnedFactView[]>([]);
  const [open, setOpen] = useState(false);
  const [editing, setEditing] = useState<string | "new" | null>(null);
  const load = () => call<PinnedFactView[]>("pinned:list", { chatId }).then(setFacts, () => setFacts([]));
  useEffect(() => {
    void load();
  }, [chatId, version]);

  const goal = facts.find((f) => f.kind === "goal");
  const rest = facts.filter((f) => f.kind !== "goal");
  const save = async (id: string | undefined, kind: PinnedFactView["kind"], text: string) => {
    try {
      if (text.trim()) await call("pinned:upsert", { chatId, id, kind, text: text.trim() });
      else if (id) await call("pinned:delete", { id });
      setEditing(null);
      await load();
    } catch (e) {
      toast(errText(e), "error");
    }
  };

  if (!goal && !rest.length) return null;
  return (
    <div className="relative mx-3 mb-2 overflow-hidden rounded-[16px] bg-ink px-4 py-3 text-bg">
      <div className="flex items-center gap-2">
        <IPin size={12} />
        <span className="text-[9.5px] font-bold uppercase tracking-[0.16em] opacity-70">Goal</span>
        <div className="flex-1" />
        <button className="text-[10.5px] opacity-60 hover:opacity-100" onClick={() => setOpen(!open)}>
          {open ? "Hide" : rest.length ? `${rest.length} pinned` : "Pin"}
        </button>
      </div>
      {editing === goal?.id && goal ? (
        <textarea
          autoFocus
          className="mt-1 w-full resize-none rounded-[8px] bg-transparent text-[15px] outline-none"
          defaultValue={goal.text}
          onBlur={(e) => void save(goal.id, "goal", e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && (e.preventDefault(), e.currentTarget.blur())}
        />
      ) : (
        <div className="heading mt-1 line-clamp-3 cursor-text text-[17px] leading-[1.25]" onDoubleClick={() => goal && setEditing(goal.id)} title="Double-click to edit">
          “{goal?.text ?? "No goal yet"}”
        </div>
      )}
      {open && (
        <div className="mt-2 border-t border-bg/20 pt-2">
          {rest.map((f) => (
            <div key={f.id} className="group flex items-start gap-2 py-[2px] text-[12px]">
              <span className="w-[66px] shrink-0 text-[9.5px] font-bold uppercase tracking-[0.14em] opacity-60">{f.kind}</span>
              <span className="flex-1">{f.text}</span>
              <button className="opacity-0 group-hover:opacity-70" onClick={() => void save(f.id, f.kind, "")}>
                <IClose size={12} />
              </button>
            </div>
          ))}
          {editing === "new" ? (
            <NewPin onDone={(kind, text) => (text ? void save(undefined, kind, text) : setEditing(null))} />
          ) : (
            <button className="mt-1 flex items-center gap-1 text-[11px] opacity-70 hover:opacity-100" onClick={() => setEditing("new")}>
              <IPlus size={12} /> Add constraint or decision
            </button>
          )}
        </div>
      )}
    </div>
  );
}

function NewPin({ onDone }: { onDone: (kind: "constraint" | "decision", text: string) => void }) {
  const [kind, setKind] = useState<"constraint" | "decision">("constraint");
  return (
    <div className="mt-1 flex gap-2">
      <select className="rounded-[6px] bg-bg/10 px-1 text-[11px]" value={kind} onChange={(e) => setKind(e.target.value as never)}>
        <option value="constraint">constraint</option>
        <option value="decision">decision</option>
      </select>
      <input
        autoFocus
        className="flex-1 rounded-[6px] bg-bg/10 px-2 py-[2px] text-[12px] outline-none"
        onKeyDown={(e) => {
          if (e.key === "Enter") onDone(kind, e.currentTarget.value.trim());
          if (e.key === "Escape") onDone(kind, "");
        }}
      />
    </div>
  );
}
