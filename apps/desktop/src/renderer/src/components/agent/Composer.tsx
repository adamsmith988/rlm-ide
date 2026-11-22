import { useEffect, useMemo, useRef, useState } from "react";
import { call } from "../../api";
import { useApp } from "../../store";
import { IClose, IPaperclip, IPick, ISend, IStop } from "../../ui/icons";

let fileCache: { at: number; files: string[] } | null = null;
async function listFiles(): Promise<string[]> {
  if (fileCache && Date.now() - fileCache.at < 15000) return fileCache.files;
  const files = await call<string[]>("fs:listFiles", { limit: 20000 }).catch(() => []);
  fileCache = { at: Date.now(), files };
  return files;
}

interface SlashItem {
  id: string;
  label: string;
  hint: string;
  run: () => void;
}

let attachSeq = 0;

export function Composer({ running, disabledReason }: { running: boolean; disabledReason: string | null }) {
  const { sendMessage, stopTurn, tabs, activeTab, selection, attachments, detach, attach, settings } = useApp();
  const [text, setText] = useState("");
  const [mention, setMention] = useState<{ q: string; start: number } | null>(null);
  const [slash, setSlash] = useState<{ q: string; start: number } | null>(null);
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

  const pickFiles = async (images: boolean) => {
    const files = await call<Array<{ kind: "image" | "file"; name: string; path: string; image?: string; text?: string }>>("dialog:openFiles", {
      images,
      title: images ? "Attach images" : "Attach files",
    }).catch(() => []);
    for (const f of files) {
      if (f.kind === "image" && f.image) attach({ id: `a${++attachSeq}`, kind: "image", name: f.name, path: f.path, image: f.image });
      else if (f.kind === "file" && f.text !== undefined) attach({ id: `a${++attachSeq}`, kind: "file", name: f.name, path: f.path, text: f.text });
    }
  };

  const onChange = (v: string, caret: number) => {
    setText(v);
    const before = v.slice(0, caret);
    const m = /(^|\s)@([\w./-]*)$/.exec(before);
    const sl = /(^|\s)\/([\w.-]*)$/.exec(before);
    setMention(m && !sl ? { q: m[2]!, start: caret - m[2]!.length - 1 } : null);
    setSlash(sl ? { q: sl[2]!, start: caret - sl[2]!.length - 1 } : null);
    setPick(0);
  };

  const replaceSlash = (insertText: string) => {
    if (!slash) return;
    const end = slash.start + 1 + slash.q.length;
    const next = `${text.slice(0, slash.start)}${insertText}${text.slice(end)}`;
    setText(next);
    setSlash(null);
    setTimeout(() => ta.current?.focus(), 0);
  };

  const commands: SlashItem[] = [
    { id: "file", label: "/file", hint: "Attach a file", run: () => (replaceSlash(""), void pickFiles(false)) },
    { id: "image", label: "/image", hint: "Attach an image", run: () => (replaceSlash(""), void pickFiles(true)) },
    { id: "attach", label: "/attach", hint: "Attach a file", run: () => (replaceSlash(""), void pickFiles(false)) },
    ...(settings?.skills ?? [])
      .filter((s) => s.enabled)
      .map((s) => ({ id: s.name, label: `/${s.name}`, hint: s.description || "Skill", run: () => replaceSlash(`${s.body.trim()}\n\n`) })),
  ];
  const slashMatches = slash ? commands.filter((c) => c.label.slice(1).toLowerCase().includes(slash.q.toLowerCase())).slice(0, 8) : [];

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
    setSlash(null);
    void sendMessage(t, mentions).then((ok) => {
      if (!ok) setText((cur) => cur || t);
    });
  };

  const tab = tabs.find((t) => t.id === activeTab);
  return (
    <div className="relative">
      {slash && slashMatches.length > 0 && (
        <div className="card absolute bottom-full left-0 right-0 z-20 mb-2 overflow-hidden py-1" style={{ boxShadow: "var(--shadow-lg)" }} data-testid="slash-menu">
          {slashMatches.map((c, i) => (
            <button
              key={c.id}
              className="flex w-full gap-2 px-3 py-[5px] text-left text-[12px]"
              style={{ background: i === pick ? "var(--surface-2)" : undefined }}
              onMouseDown={(e) => {
                e.preventDefault();
                setSlash(null);
                c.run();
              }}
            >
              <span className="font-medium">{c.label}</span>
              <span className="truncate text-faint">{c.hint}</span>
            </button>
          ))}
        </div>
      )}
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
                {"image" in a && a.image ? <img src={a.image} alt="" className="h-6 w-9 rounded-full border border-line object-cover object-top" /> : <IPick size={14} />}
                <span className="min-w-0 truncate">
                  {a.kind === "file" || a.kind === "image" ? (
                    <b>{a.name}</b>
                  ) : a.kind === "screenshot" ? (
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
          placeholder={disabledReason ?? "Ask, plan, or build…  (@ file, / skill)"}
          disabled={!!disabledReason}
          className="block max-h-[220px] min-h-[40px] w-full resize-none bg-transparent px-2 py-1.5 text-[13px] outline-none placeholder:text-faint"
          onChange={(e) => onChange(e.target.value, e.target.selectionStart)}
          onPaste={(e) => {
            const img = [...e.clipboardData.files].find((f) => f.type.startsWith("image/"));
            if (!img) return;
            e.preventDefault();
            const reader = new FileReader();
            reader.onload = () => attach({ id: `a${++attachSeq}`, kind: "image", name: img.name || "pasted.png", path: "clipboard", image: String(reader.result) });
            reader.readAsDataURL(img);
          }}
          onKeyDown={(e) => {
            if (slash && slashMatches.length) {
              if (e.key === "ArrowDown") return e.preventDefault(), setPick((pick + 1) % slashMatches.length);
              if (e.key === "ArrowUp") return e.preventDefault(), setPick((pick - 1 + slashMatches.length) % slashMatches.length);
              if (e.key === "Enter" || e.key === "Tab") {
                e.preventDefault();
                const c = slashMatches[pick]!;
                setSlash(null);
                c.run();
                return;
              }
              if (e.key === "Escape") return setSlash(null);
            }
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
          <button className="iconbtn h-7 w-7 shrink-0" title="Attach a file or image" disabled={!!disabledReason} onClick={() => void pickFiles(false)} data-testid="attach">
            <IPaperclip size={15} />
          </button>
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
