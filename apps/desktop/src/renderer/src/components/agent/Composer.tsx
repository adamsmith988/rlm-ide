import { useEffect, useMemo, useRef, useState } from "react";
import { call } from "../../api";
import { useApp } from "../../store";
import { IClose, IPick, ISend, IStop } from "../../ui/icons";

let fileCache: { at: number; files: string[] } | null = null;
async function listFiles(): Promise<string[]> {
  if (fileCache && Date.now() - fileCache.at < 15000) return fileCache.files;
  const files = await call<string[]>("fs:listFiles", { limit: 20000 }).catch(() => []);
  fileCache = { at: Date.now(), files };
  return files;
}

export function Composer({ running, disabledReason }: { running: boolean; disabledReason: string | null }) {
  const { sendMessage, stopTurn, tabs, activeTab, selection, attachments, detach } = useApp();
  const [text, setText] = useState("");
  const [mention, setMention] = useState<{ q: string; start: number } | null>(null);
  const [files, setFiles] = useState<string[]>([]);
  const [pick, setPick] = useState(0);
  const ta = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const el = ta.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(220, el.scrollHeight)}px`;
  }, [text]);

  useEffect(() => {
    if (mention) void listFiles().then(setFiles);
  }, [mention?.start]);

  const matches = useMemo(() => {
    if (!mention) return [];
    const q = mention.q.toLowerCase();
    return files
      .filter((f) => f.toLowerCase().includes(q))
      .sort((a, b) => score(a, q) - score(b, q))
      .slice(0, 8);
  }, [mention, files]);

  const onChange = (v: string, caret: number) => {
    setText(v);
    const before = v.slice(0, caret);
    const m = /(^|\s)@([\w./-]*)$/.exec(before);
    setMention(m ? { q: m[2]!, start: caret - m[2]!.length - 1 } : null);
    setPick(0);
  };

  const insert = (file: string) => {
    if (!mention) return;
    const end = mention.start + 1 + mention.q.length;
    const next = `${text.slice(0, mention.start)}@${file} ${text.slice(end)}`;
    setText(next);
    setMention(null);
    setTimeout(() => {
      const pos = mention.start + file.length + 2;
      ta.current?.setSelectionRange(pos, pos);
      ta.current?.focus();
    }, 0);
  };

  const submit = () => {
    const t = text.trim();
    if ((!t && !attachments.length) || running || disabledReason) return;
    const mentions = [...t.matchAll(/(?:^|\s)@([\w./-]+)/g)].map((m) => m[1]!);
    setText("");
    setMention(null);
    void sendMessage(t, mentions).then((ok) => {
      if (!ok) setText((cur) => cur || t);
    });
  };

  const tab = tabs.find((t) => t.id === activeTab);
  return (
    <div className="relative">
      {mention && matches.length > 0 && (
        <div className="card absolute bottom-full left-0 right-0 z-20 mb-2 overflow-hidden py-1" style={{ boxShadow: "var(--shadow-lg)" }}>
          {matches.map((f, i) => (
            <button
              key={f}
              className="flex w-full gap-2 px-3 py-[5px] text-left text-[12px]"
              style={{ background: i === pick ? "var(--surface-2)" : undefined }}
              onMouseDown={(e) => (e.preventDefault(), insert(f))}
            >
              <span className="font-medium">{f.split("/").pop()}</span>
              <span className="truncate text-faint">{f}</span>
            </button>
          ))}
        </div>
      )}
      <div className="rounded-[18px] border-[1.5px] border-line-strong bg-surface p-2 focus-within:border-ink">
        {attachments.length > 0 && (
          <div className="flex flex-wrap gap-1.5 px-1 pb-1.5" data-testid="attachments">
            {attachments.map((a) => (
              <div key={a.id} className="flex max-w-full items-center gap-2 rounded-full border-[1.5px] border-ink py-[2px] pl-[3px] pr-1 text-[11px]">
                {a.image ? <img src={a.image} alt="" className="h-6 w-9 rounded-full border border-line object-cover object-top" /> : <IPick size={14} />}
                <span className="min-w-0 truncate">
                  {a.kind === "screenshot" ? (
                    <>
                      <b>Screenshot</b> · {a.title || a.pageUrl}
                    </>
                  ) : (
                    <>
                      <b className="mono">{a.selector.length > 40 ? `…${a.selector.slice(-40)}` : a.selector}</b> · {a.title || a.pageUrl}
                    </>
                  )}
                </span>
                <button className="iconbtn h-5 w-5 shrink-0" title="Remove" onClick={() => detach(a.id)}>
                  <IClose size={11} />
                </button>
              </div>
            ))}
          </div>
        )}
        <textarea
          id="composer"
          ref={ta}
          rows={1}
          value={text}
          placeholder={disabledReason ?? "Ask, plan, or build…  (@ to mention a file)"}
          disabled={!!disabledReason}
          className="block max-h-[220px] min-h-[40px] w-full resize-none bg-transparent px-2 py-1.5 text-[13px] outline-none placeholder:text-faint"
          onChange={(e) => onChange(e.target.value, e.target.selectionStart)}
          onKeyDown={(e) => {
            if (mention && matches.length) {
              if (e.key === "ArrowDown") return e.preventDefault(), setPick((pick + 1) % matches.length);
              if (e.key === "ArrowUp") return e.preventDefault(), setPick((pick - 1 + matches.length) % matches.length);
              if (e.key === "Enter" || e.key === "Tab") return e.preventDefault(), insert(matches[pick]!);
              if (e.key === "Escape") return setMention(null);
            }
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
        />
        <div className="flex items-center gap-2 px-1 pt-1">
          <div className="flex min-w-0 flex-1 items-center gap-1.5 overflow-hidden text-[11px] text-faint">
            {tab?.kind === "file" && <span className="truncate rounded-full border border-line px-2 py-[1px]">{tab.title}</span>}
            {selection && <span className="shrink-0 rounded-full border border-line px-2 py-[1px]">{selection.split("\n").length} lines selected</span>}
          </div>
          {running ? (
            <button className="round-btn on" style={{ width: 34, height: 34 }} title="Stop" onClick={() => void stopTurn()}>
              <IStop size={15} />
            </button>
          ) : (
            <button className="round-btn on" style={{ width: 34, height: 34, opacity: (text.trim() || attachments.length) && !disabledReason ? 1 : 0.4 }} title="Send (Enter)" onClick={submit}>
              <ISend size={15} />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

function score(path: string, q: string): number {
  const name = path.split("/").pop()!.toLowerCase();
  if (name.startsWith(q)) return 0;
  if (name.includes(q)) return 1 + path.length / 1000;
  return 2 + path.length / 1000;
}
