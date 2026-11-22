import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from "react";
import { agentPickerModels } from "@rlm/contracts";
import { call } from "../../api";
import { selectAgentModel } from "../../models";
import { useApp } from "../../store";
import { IAsk, IBug, IChevron, IClose, ICube, INodes, IPaperclip, IPick, IPlan, IPlug, IPlus, ISend, IStop } from "../../ui/icons";

let fileCache: { at: number; files: string[] } | null = null;
async function listFiles(): Promise<string[]> {
  if (fileCache && Date.now() - fileCache.at < 15000) return fileCache.files;
  const files = await call<string[]>("fs:listFiles", { limit: 20000 }).catch(() => []);
  fileCache = { at: Date.now(), files };
  return files;
}

type ModeId = "plan" | "debug" | "multitask" | "ask";
type Pane = "main" | "model" | "mcp" | "skills";
type SkillRow = { name: string; description: string; source: "user" | "project"; files: string[] };

const skillPreface = (name: string) =>
  `**Skill: ${name}.** Follow the ${name} skill completely. Call skills.read(name="${name}") and follow every step, including extra files via skills.read_file.`;

const MODES: Array<{ id: ModeId; label: string; hint: string; preface: string; color: string; Icon: typeof IPlan }> = [
  { id: "plan", label: "Plan", hint: "Generate an implementation plan", preface: "**Mode: Plan.** Generate an implementation plan. Do not edit files or apply changes.", color: "#d9a441", Icon: IPlan },
  { id: "debug", label: "Debug", hint: "Pinpoint the root cause of an issue", preface: "**Mode: Debug.** Find the root cause before changing code, then fix it.", color: "#d46a6a", Icon: IBug },
  { id: "multitask", label: "Multitask", hint: "Orchestrate multiple subagents in parallel", preface: "**Mode: Multitask.** Split the independent parts across sub-agents with spawn_agents and run them in parallel.", color: "#8b7cf7", Icon: INodes },
  { id: "ask", label: "Ask", hint: "Answer questions without making edits", preface: "**Mode: Ask.** Answer the question. Do not edit files or run commands that change the project.", color: "#6faf7a", Icon: IAsk },
];

type Row =
  | { key: string; label: string; hint: string; color?: string; Icon: typeof IPlan; chevron?: boolean; run: () => void };

let attachSeq = 0;

export function Composer({ running, disabledReason }: { running: boolean; disabledReason: string | null }) {
  const { sendMessage, stopTurn, tabs, activeTab, selection, attachments, detach, attach, settings, chats, selectChat, set } = useApp();
  const hasMessages = useApp((s) => (s.chatId ? (s.messages[s.chatId]?.length ?? 0) > 0 : false));
  const [text, setText] = useState("");
  const [mention, setMention] = useState<{ q: string; start: number } | null>(null);
  const [slash, setSlash] = useState<{ q: string; start: number } | null>(null);
  const [plus, setPlus] = useState(false);
  const [pane, setPane] = useState<Pane>("main");
  const [plusQuery, setPlusQuery] = useState("");
  const [mode, setMode] = useState<ModeId | null>(null);
  const [files, setFiles] = useState<string[]>([]);
  const [skillCatalog, setSkillCatalog] = useState<SkillRow[]>([]);
  const [skill, setSkill] = useState<string | null>(null);
  const [pick, setPick] = useState(0);
  const ta = useRef<HTMLTextAreaElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const el = ta.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(220, el.scrollHeight)}px`;
  }, [text]);

  const menuOpen = plus || !!slash;
  useEffect(() => {
    if (menuOpen) void listFiles().then(setFiles);
  }, [menuOpen]);
  useEffect(() => {
    if (!menuOpen) return;
    void call<SkillRow[]>("skills:list", {})
      .then(setSkillCatalog)
      .catch(() => setSkillCatalog([]));
  }, [menuOpen, settings?.skills]);
  useEffect(() => {
    if (plus) searchRef.current?.focus();
  }, [plus, pane]);

  const query = (slash ? slash.q : plusQuery).trim().toLowerCase();
  const models = settings ? agentPickerModels(settings) : [];
  const modelLabel = models.find((m) => m.id === settings?.roles.agent)?.label ?? "Model";

  const closeMenu = () => {
    setPlus(false);
    setSlash(null);
    setPane("main");
    setPlusQuery("");
    setPick(0);
  };

  const clearSlashToken = () => {
    if (!slash) return;
    const end = slash.start + 1 + slash.q.length;
    setText(`${text.slice(0, slash.start)}${text.slice(end)}`);
    setSlash(null);
  };

  const pickFiles = async (images: boolean) => {
    const picked = await call<Array<{ kind: "image" | "file"; name: string; path: string; image?: string; text?: string }>>("dialog:openFiles", {
      images,
      title: images ? "Attach images" : "Attach files",
    }).catch(() => []);
    for (const f of picked) {
      if (f.kind === "image" && f.image) attach({ id: `a${++attachSeq}`, kind: "image", name: f.name, path: f.path, image: f.image });
      else if (f.kind === "file" && f.text !== undefined) attach({ id: `a${++attachSeq}`, kind: "file", name: f.name, path: f.path, text: f.text });
    }
  };

  const attachProjectFile = async (path: string) => {
    const name = path.split("/").pop() || path;
    if (/\.(png|jpe?g|gif|webp|bmp)$/i.test(name)) {
      closeMenu();
      await pickFiles(true);
      return;
    }
    const body = await call<{ content: string; binary: boolean }>("fs:readFile", { path }).catch(() => null);
    if (!body) return;
    const text = body.binary ? "(binary file, not inlined)" : body.content.slice(0, 80_000);
    attach({ id: `a${++attachSeq}`, kind: "file", name, path, text });
  };

  const insertText = (insert: string) => {
    if (slash) {
      const end = slash.start + 1 + slash.q.length;
      const next = `${text.slice(0, slash.start)}${insert}${text.slice(end)}`;
      setText(next);
      setSlash(null);
    } else {
      setText((cur) => (cur && !cur.endsWith("\n") ? `${cur}\n${insert}` : `${cur}${insert}`));
    }
    closeMenu();
    setTimeout(() => ta.current?.focus(), 0);
  };

  const chooseMode = (id: ModeId) => {
    setMode(id);
    clearSlashToken();
    closeMenu();
    setTimeout(() => ta.current?.focus(), 0);
  };

  const chooseSkill = (name: string) => {
    setSkill(name);
    clearSlashToken();
    closeMenu();
    setTimeout(() => ta.current?.focus(), 0);
  };

  const skillRows = (hit: (label: string, hint?: string) => boolean): Row[] =>
    skillCatalog
      .filter((s) => hit(s.name, s.description))
      .map((s) => ({
        key: `skill-${s.source}-${s.name}`,
        label: s.name,
        hint: s.description || (s.source === "project" ? "Project skill" : "Your skill"),
        Icon: IPlan,
        color: "#d9a441",
        run: () => chooseSkill(s.name),
      }));

  const rows = useMemo<Row[]>(() => {
    const hit = (label: string, hint = "") => !query || `${label} ${hint}`.toLowerCase().includes(query);
    if (pane === "model") {
      return models.filter((m) => hit(m.label)).map((m) => ({
        key: m.id,
        label: m.label,
        hint: m.id === settings?.roles.agent ? "Current model" : "",
        Icon: ICube,
        color: "#9aa0a6",
        run: () => void selectAgentModel(m.id).then(closeMenu),
      }));
    }
    if (pane === "skills") {
      const items = skillRows(hit);
      items.push({
        key: "skill-settings",
        label: skillCatalog.length ? "Manage skills" : "Add a skill",
        hint: "Settings",
        Icon: IPlan,
        color: "#d9a441",
        run: () => (closeMenu(), set({ activity: "settings" })),
      });
      return items;
    }
    if (pane === "mcp") {
      const servers = (settings?.mcp ?? []).filter((s) => hit(s.name, s.command));
      const items: Row[] = servers.map((s) => ({
        key: s.id,
        label: s.name,
        hint: s.enabled ? s.command : "Disabled",
        Icon: IPlug,
        color: "#9aa0a6",
        run: () => insertText(`Use the ${s.name} MCP server. `),
      }));
      items.push({
        key: "mcp-settings",
        label: servers.length ? "Manage MCP servers" : "Add an MCP server",
        hint: "Settings",
        Icon: IPlug,
        color: "#9aa0a6",
        run: () => (closeMenu(), set({ activity: "settings" })),
      });
      return items;
    }
    const main: Row[] = [];
    if (!query) {
      for (const m of MODES) main.push({ key: m.id, label: m.label, hint: m.hint, color: m.color, Icon: m.Icon, run: () => chooseMode(m.id) });
      main.push(
        { key: "files", label: "Files", hint: "", Icon: IPaperclip, color: "#b0b4b8", run: () => (clearSlashToken(), closeMenu(), void pickFiles(false)) },
        { key: "skills", label: "Skills", hint: skillCatalog.length ? `${skillCatalog.length}` : "", Icon: IPlan, color: "#d9a441", chevron: true, run: () => (setPane("skills"), setPick(0)) },
        { key: "model", label: "Model", hint: modelLabel, Icon: ICube, color: "#b0b4b8", chevron: true, run: () => (setPane("model"), setPick(0)) },
        { key: "mcp", label: "MCP", hint: "", Icon: IPlug, color: "#b0b4b8", chevron: true, run: () => (setPane("mcp"), setPick(0)) },
      );
      return main;
    }
    for (const m of MODES) if (hit(m.label, m.hint)) main.push({ key: m.id, label: m.label, hint: m.hint, color: m.color, Icon: m.Icon, run: () => chooseMode(m.id) });
    main.push(...skillRows(hit));
    if (hit("Files", "attach")) main.push({ key: "files", label: "Files", hint: "Attach a file or image", Icon: IPaperclip, color: "#b0b4b8", run: () => (clearSlashToken(), closeMenu(), void pickFiles(false)) });
    for (const f of files.filter((f) => f.toLowerCase().includes(query)).slice(0, 6)) {
      main.push({ key: `file-${f}`, label: f.split("/").pop() || f, hint: f, Icon: IPaperclip, color: "#b0b4b8", run: () => void attachProjectFile(f).then(closeMenu) });
    }
    for (const c of chats.filter((c) => hit(c.title)).slice(0, 4)) {
      main.push({ key: `chat-${c.id}`, label: c.title || "Chat", hint: "Chat", Icon: IAsk, color: "#6faf7a", run: () => (closeMenu(), void selectChat(c.id)) });
    }
    if (hit("Skills")) main.push({ key: "skills", label: "Skills", hint: "", Icon: IPlan, color: "#d9a441", chevron: true, run: () => (setPane("skills"), setPick(0)) });
    if (hit("Model", modelLabel)) main.push({ key: "model", label: "Model", hint: modelLabel, Icon: ICube, color: "#b0b4b8", chevron: true, run: () => (setPane("model"), setPick(0)) });
    if (hit("MCP")) main.push({ key: "mcp", label: "MCP", hint: "", Icon: IPlug, color: "#b0b4b8", chevron: true, run: () => (setPane("mcp"), setPick(0)) });
    return main.slice(0, 12);
  }, [pane, query, settings, files, chats, modelLabel, text, slash, skillCatalog]);

  const onChange = (v: string, caret: number) => {
    setText(v);
    const before = v.slice(0, caret);
    const m = /(^|\s)@([\w./-]*)$/.exec(before);
    const sl = /(^|\s)\/([\w.-]*)$/.exec(before);
    setMention(m && !sl ? { q: m[2]!, start: caret - m[2]!.length - 1 } : null);
    if (sl) {
      setSlash({ q: sl[2]!, start: caret - sl[2]!.length - 1 });
      setPlus(false);
      setPane("main");
    } else setSlash(null);
    setPick(0);
  };

  const matches = useMemo(() => {
    if (!mention) return [];
    const q = mention.q.toLowerCase();
    return files
      .filter((f) => f.toLowerCase().includes(q))
      .sort((a, b) => score(a, q) - score(b, q))
      .slice(0, 8);
  }, [mention, files]);

  const insertMention = (file: string) => {
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

  const runRow = (row: Row) => {
    row.run();
  };

  const onMenuKey = (e: KeyboardEvent) => {
    if (!menuOpen || mention) return false;
    if (!rows.length) {
      if (e.key === "Escape") closeMenu();
      return e.key === "Escape";
    }
    if (e.key === "ArrowDown") return e.preventDefault(), setPick((pick + 1) % rows.length), true;
    if (e.key === "ArrowUp") return e.preventDefault(), setPick((pick - 1 + rows.length) % rows.length), true;
    if (e.key === "Enter" || e.key === "Tab") return e.preventDefault(), runRow(rows[Math.min(pick, rows.length - 1)]!), true;
    if (e.key === "Escape") return closeMenu(), true;
    return false;
  };

  const submit = () => {
    const t = text.trim();
    if ((!t && !attachments.length) || running || disabledReason) return;
    const mentions = [...t.matchAll(/(?:^|\s)@([\w./-]+)/g)].map((m) => m[1]!);
    const modeLine = mode ? MODES.find((m) => m.id === mode)?.preface : "";
    const skillLine = skill ? skillPreface(skill) : "";
    const preface = [skillLine, modeLine].filter(Boolean).join("\n\n");
    const payload = preface ? `${preface}\n\n${t}` : t;
    setText("");
    setMention(null);
    closeMenu();
    void sendMessage(payload, mentions).then((ok) => {
      if (!ok) setText((cur) => cur || t);
    });
  };

  const tab = tabs.find((t) => t.id === activeTab);
  const activeMode = MODES.find((m) => m.id === mode);
  return (
    <div className="relative">
      {menuOpen && (
        <div className="absolute bottom-full left-0 right-0 z-20 mb-2 overflow-hidden rounded-[16px] border border-line bg-surface" style={{ boxShadow: "var(--shadow-lg)" }} data-testid="slash-menu">
          <div className="p-2 pb-1">
            <input
              ref={searchRef}
              className="h-8 w-full rounded-[10px] border border-line bg-bg px-3 text-[13px] outline-none placeholder:text-faint"
              placeholder="Search skills, context, chats..."
              value={slash ? slash.q : plusQuery}
              onChange={(e) => {
                if (slash) {
                  const end = slash.start + 1 + slash.q.length;
                  const next = `${text.slice(0, slash.start)}/${e.target.value}${text.slice(end)}`;
                  setText(next);
                  setSlash({ q: e.target.value, start: slash.start });
                } else setPlusQuery(e.target.value);
                setPane("main");
                setPick(0);
              }}
              onKeyDown={(e) => {
                if (onMenuKey(e)) e.stopPropagation();
              }}
            />
          </div>
          {pane !== "main" && (
            <button className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-[12px] text-dim hover:bg-surface-2" onMouseDown={(e) => (e.preventDefault(), setPane("main"), setPick(0))}>
              <IChevron size={13} style={{ transform: "rotate(180deg)" }} /> Back
            </button>
          )}
          <div className="max-h-[280px] overflow-y-auto py-1">
            {rows.map((row, i) => (
              <span key={row.key}>
                {pane === "main" && !query && row.key === "files" && <div className="mx-3 my-1 border-t border-line" />}
                <button
                  className="flex w-full items-center gap-2.5 px-3 py-[6px] text-left text-[13px]"
                  style={{ background: i === pick ? "var(--surface-2)" : undefined }}
                  onMouseDown={(e) => {
                    e.preventDefault();
                    runRow(row);
                  }}
                  onMouseEnter={() => setPick(i)}
                >
                  <row.Icon size={15} style={{ color: row.color }} />
                  <span className="font-medium">{row.label}</span>
                  {row.hint && <span className="min-w-0 flex-1 truncate text-dim">{row.hint}</span>}
                  {row.chevron && <IChevron size={14} className="ml-auto text-faint" />}
                </button>
              </span>
            ))}
            {!rows.length && <div className="px-3 py-2 text-[12px] text-faint">Nothing matches.</div>}
          </div>
        </div>
      )}
      {mention && matches.length > 0 && !menuOpen && (
        <div className="card absolute bottom-full left-0 right-0 z-20 mb-2 overflow-hidden py-1" style={{ boxShadow: "var(--shadow-lg)" }}>
          {matches.map((f, i) => (
            <button
              key={f}
              className="flex w-full gap-2 px-3 py-[5px] text-left text-[12px]"
              style={{ background: i === pick ? "var(--surface-2)" : undefined }}
              onMouseDown={(e) => (e.preventDefault(), insertMention(f))}
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
          placeholder={disabledReason ?? (hasMessages ? "Send follow-up" : "Ask, plan, or build…")}
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
            if (slash && onMenuKey(e)) return;
            if (mention && matches.length) {
              if (e.key === "ArrowDown") return e.preventDefault(), setPick((pick + 1) % matches.length);
              if (e.key === "ArrowUp") return e.preventDefault(), setPick((pick - 1 + matches.length) % matches.length);
              if (e.key === "Enter" || e.key === "Tab") return e.preventDefault(), insertMention(matches[pick]!);
              if (e.key === "Escape") return setMention(null);
            }
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
        />
        <div className="flex items-center gap-2 px-1 pt-1">
          <button
            className="iconbtn h-7 w-7 shrink-0"
            title="Attach, skills, and tools"
            disabled={!!disabledReason}
            data-testid="attach"
            onClick={() => {
              if (plus) closeMenu();
              else {
                setPlus(true);
                setSlash(null);
                setPane("main");
                setPlusQuery("");
                setPick(0);
              }
            }}
          >
            <IPlus size={16} style={{ transform: plus ? "rotate(45deg)" : undefined }} />
          </button>
          {activeMode && (
            <button className="flex items-center gap-1 rounded-full border border-line px-2 py-[2px] text-[11px]" title="Clear mode" onClick={() => setMode(null)}>
              <activeMode.Icon size={12} style={{ color: activeMode.color }} />
              {activeMode.label}
              <IClose size={10} />
            </button>
          )}
          {skill && (
            <button className="flex items-center gap-1 rounded-full border border-line px-2 py-[2px] text-[11px]" title="Clear skill" data-testid="skill-chip" onClick={() => setSkill(null)}>
              <IPlan size={12} style={{ color: "#d9a441" }} />
              {skill}
              <IClose size={10} />
            </button>
          )}
          <div className="flex min-w-0 flex-1 items-center gap-1.5 overflow-hidden text-[11px] text-faint">
            {tab?.kind === "file" && <span className="truncate rounded-full border border-line px-2 py-[1px]">{tab.title}</span>}
            {selection && <span className="shrink-0 rounded-full border border-line px-2 py-[1px]">{selection.split("\n").length} lines selected</span>}
          </div>
          <button
            className="flex max-w-[140px] items-center gap-1 truncate rounded-full px-1 text-[12px] text-dim hover:text-ink"
            title="Model"
            onClick={() => {
              setPlus(true);
              setPane("model");
              setPick(0);
            }}
          >
            <span className="truncate">{modelLabel}</span>
            <IChevron size={12} style={{ transform: "rotate(90deg)" }} />
          </button>
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
