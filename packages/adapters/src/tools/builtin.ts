import { promises as fs } from "node:fs";
import path from "node:path";
import type { ToolContext, ToolImpl, ToolRegistryPort, ToolDescriptor } from "@rlm/core";
import { globToRegExp, isProbablyBinary, relPath, resolveInside, walk } from "../fs/confine";
import { ProcessManager } from "./shell";
import { syntaxProblem } from "./syntax";
import { describeRanges, errorHint, errorLocations, isSmallFixByRewrite, rewriteStats } from "./rewrite_guard";

/** Max chars a single read returns; the model is told to page with line ranges. */
const READ_MAX_CHARS = 60000;
const GREP_FILE_MAX_BYTES = 2_000_000;

export interface ToolHostServices {
  processes: ProcessManager;
  shellTimeoutSec: () => number;
  /** The REPL cell time limit; blocking shell calls are kept under it so the kernel is not restarted. */
  cellTimeoutSec?: () => number;
  editorState: () => { activeFile?: string; selection?: string; openFiles?: string[]; diagnostics?: Array<{ path: string; line: number; severity: string; message: string }> };
  searchSession: (chatId: string, query: string, limit: number) => Promise<Array<{ turn: number; role: string; excerpt: string }>>;
  /** Loads an HTML file in a hidden offline browser (desktop host only). */
  checkPage?: (args: { path: string; waitMs: number; clicks: string[] }) => Promise<Record<string, unknown>>;
  /** Drives the IDE's visible built-in browser (desktop host only). */
  browser?: (op: string, args: Record<string, unknown>) => Promise<unknown>;
}

const s = (description: string, extra: Record<string, unknown> = {}) => ({ type: "string" as const, description, ...extra });
const i = (description: string, def?: number) => ({ type: "integer" as const, description, ...(def === undefined ? {} : { default: def }) });
const b = (description: string, def = false) => ({ type: "boolean" as const, description, default: def });

function tool(d: Omit<ToolDescriptor, "timeoutSec"> & { timeoutSec?: number }, run: ToolImpl["run"], preview?: ToolImpl["preview"], precheck?: ToolImpl["precheck"]): ToolImpl {
  return { descriptor: { timeoutSec: 30, ...d }, run, preview, precheck };
}

async function readText(abs: string): Promise<string> {
  return fs.readFile(abs, "utf8");
}

/** The block of the file that best matches `oldS` line by line (whitespace-insensitive), numbered. */
function closestRegion(text: string, oldS: string): string {
  const norm = (l: string) => l.replace(/\s+/g, " ").trim();
  const want = oldS.split("\n").map(norm);
  const lines = text.split("\n");
  const n = Math.max(1, want.length);
  let best = -1;
  let bestScore = 0;
  for (let i = 0; i + 1 <= lines.length; i++) {
    let score = 0;
    for (let k = 0; k < n && i + k < lines.length; k++) {
      const w = want[k]!;
      const l = norm(lines[i + k]!);
      if (!w) continue;
      if (l === w) score += 2;
      else if (l && (l.includes(w) || w.includes(l))) score += 1;
    }
    if (score > bestScore) {
      bestScore = score;
      best = i;
    }
  }
  if (best < 0 || bestScore < Math.max(1, Math.ceil(want.filter(Boolean).length / 3))) return "";
  const from = Math.max(0, best - 1);
  const to = Math.min(lines.length, best + n + 1);
  return lines
    .slice(from, to)
    .map((l, k) => `${String(from + k + 1).padStart(5)}| ${l}`)
    .join("\n")
    .slice(0, 3000);
}

function editResult(before: string, oldS: string, newS: string, replaceAll: boolean): string {
  if (!oldS) throw new Error("old_string must not be empty; use fs.write_file to create a file");
  const count = before.split(oldS).length - 1;
  if (count === 0) {
    const trimmed = oldS.trim();
    const hint = trimmed && before.includes(trimmed) ? " (it matches after trimming whitespace; copy indentation exactly)" : "";
    const near = closestRegion(before, oldS);
    throw new Error(`old_string not found in file${hint}.${near ? ` The closest text in the file is:\n${near}\nCopy the exact current text from there (without the line numbers) and retry fs.edit_file.` : " Read the file again and copy the exact text."}`);
  }
  if (count > 1 && !replaceAll) throw new Error(`old_string occurs ${count} times; add more surrounding context or pass replace_all=True`);
  return replaceAll ? before.split(oldS).join(newS) : before.replace(oldS, () => newS);
}

async function git(args: string[], ctx: ToolContext, pm: ProcessManager): Promise<string> {
  const r = await pm.run(`git ${args.map((a) => JSON.stringify(a)).join(" ")}`, ctx.workspaceRoot, 60, ctx.signal);
  if (r.exit_code !== 0) throw new Error((r.stderr || r.stdout).trim() || `git exited ${r.exit_code}`);
  return r.stdout;
}

export function createBuiltinTools(svc: ToolHostServices): ToolImpl[] {
  const pm = svc.processes;
  /** Per chat and file: rewrites that changed only a few lines. */
  const smallRewrites = new Map<string, number>();
  const blockingCap = () => {
    const cell = svc.cellTimeoutSec?.();
    return cell ? Math.max(5, cell - 15) : Number.POSITIVE_INFINITY;
  };
  return [
    tool(
      {
        id: "fs.read_file",
        group: "fs",
        description: "Read a text file. Returns the text of the requested lines. Use start_line/end_line to page through large files.",
        parameters: { type: "object", properties: { path: s("File path relative to the project root"), start_line: i("First line (1-based)", 1), end_line: i("Last line (inclusive); omit for end of file") }, required: ["path"] },
        returns: "str",
        sideEffects: false,
      },
      async (a, ctx) => {
        const abs = resolveInside(ctx.workspaceRoot, String(a.path));
        if (await isProbablyBinary(abs)) throw new Error("binary file; not readable as text");
        const lines = (await readText(abs)).split("\n");
        const start = Math.max(1, Number(a.start_line ?? 1));
        const end = Math.min(lines.length, a.end_line ? Number(a.end_line) : lines.length);
        let text = lines.slice(start - 1, end).join("\n");
        if (text.length > READ_MAX_CHARS) {
          const shown = text.slice(0, READ_MAX_CHARS).split("\n").length;
          text = `${text.slice(0, READ_MAX_CHARS)}\n[truncated: file has ${lines.length} lines; showed lines ${start}-${start + shown - 1}. Read further with start_line=${start + shown - 1}]`;
        } else if (start > 1 || end < lines.length) {
          text += `\n[lines ${start}-${end} of ${lines.length}]`;
        }
        return text;
      },
    ),
    tool(
      {
        id: "fs.list_dir",
        group: "fs",
        description: "List a directory. Directories end with '/'. VCS metadata (.git) is omitted.",
        parameters: { type: "object", properties: { path: s("Directory relative to the project root", { default: "." }) } },
        returns: "list[str]",
        sideEffects: false,
      },
      async (a, ctx) => {
        const abs = resolveInside(ctx.workspaceRoot, String(a.path ?? "."));
        const entries = await fs.readdir(abs, { withFileTypes: true });
        return entries
          .filter((e) => !(e.isDirectory() && (e.name === ".git" || e.name === ".hg" || e.name === ".svn")))
          .sort((x, y) => Number(y.isDirectory()) - Number(x.isDirectory()) || x.name.localeCompare(y.name))
          .map((e) => (e.isDirectory() ? `${e.name}/` : e.name));
      },
    ),
    tool(
      {
        id: "fs.glob",
        group: "fs",
        description: "Find files by glob pattern, e.g. 'src/**/*.tsx'. Skips node_modules, .git and build output.",
        parameters: { type: "object", properties: { pattern: s("Glob pattern relative to the project root"), max_results: i("Maximum paths", 500) }, required: ["pattern"] },
        returns: "list[str]",
        sideEffects: false,
      },
      async (a, ctx) => {
        const re = globToRegExp(String(a.pattern).replace(/^\.\//, ""));
        const out: string[] = [];
        const max = Number(a.max_results ?? 500);
        for await (const f of walk(ctx.workspaceRoot, ctx.workspaceRoot, { n: 50000 })) {
          const rel = relPath(ctx.workspaceRoot, f);
          if (re.test(rel)) out.push(rel);
          if (out.length >= max) break;
        }
        return out;
      },
    ),
    tool(
      {
        id: "search.grep",
        group: "search",
        description: "Search file contents with a regular expression. Returns 'path:line: text' matches.",
        parameters: {
          type: "object",
          properties: {
            pattern: s("Regular expression (JavaScript syntax)"),
            path: s("Directory or file to search", { default: "." }),
            glob: s("Only files matching this glob, e.g. '**/*.ts'"),
            ignore_case: b("Case-insensitive"),
            max_results: i("Maximum matches", 100),
          },
          required: ["pattern"],
        },
        returns: "list[str]",
        sideEffects: false,
      },
      async (a, ctx) => {
        let re: RegExp;
        try {
          re = new RegExp(String(a.pattern), a.ignore_case ? "i" : "");
        } catch (e) {
          throw new Error(`invalid regex: ${String(e)}`);
        }
        const base = resolveInside(ctx.workspaceRoot, String(a.path ?? "."));
        const globRe = a.glob ? globToRegExp(String(a.glob)) : null;
        const max = Number(a.max_results ?? 100);
        const out: string[] = [];
        const stat = await fs.stat(base);
        const files = stat.isFile() ? [base] : walk(ctx.workspaceRoot, base, { n: 50000 });
        for await (const f of files as AsyncIterable<string> | string[]) {
          const rel = relPath(ctx.workspaceRoot, f);
          if (globRe && !globRe.test(rel)) continue;
          const st = await fs.stat(f).catch(() => null);
          if (!st || st.size > GREP_FILE_MAX_BYTES) continue;
          if (await isProbablyBinary(f).catch(() => true)) continue;
          const lines = (await readText(f)).split("\n");
          for (let n = 0; n < lines.length; n++) {
            if (re.test(lines[n]!)) {
              out.push(`${rel}:${n + 1}: ${lines[n]!.trim().slice(0, 300)}`);
              if (out.length >= max) return out;
            }
          }
        }
        return out;
      },
    ),
    tool(
      {
        id: "fs.write_file",
        group: "fs",
        description:
          "Create a file, or replace one whose content mostly changes. To fix or change part of an existing file use fs.edit_file. Rewriting an existing file to change a few lines is refused after the first time unless overwrite=True. Parent folders are created.",
        parameters: {
          type: "object",
          properties: {
            path: s("File path relative to the project root"),
            content: s("Complete new file content"),
            overwrite: b("Confirm replacing an existing file even though most of it stays the same"),
          },
          required: ["path", "content"],
        },
        returns: "str",
        sideEffects: true,
        editsFiles: true,
      },
      async (a, ctx) => {
        const abs = resolveInside(ctx.workspaceRoot, String(a.path));
        const rel = relPath(ctx.workspaceRoot, abs);
        const content = String(a.content);
        const before = await readText(abs).catch(() => null);
        await fs.mkdir(path.dirname(abs), { recursive: true });
        await fs.writeFile(abs, content, "utf8");
        const notes: string[] = [];
        if (before !== null) {
          const st = rewriteStats(before, content);
          if (isSmallFixByRewrite(st)) {
            const key = `${ctx.chatId}\u0000${abs}`;
            const n = (smallRewrites.get(key) ?? 0) + 1;
            smallRewrites.set(key, n);
            if (n === 1 && !a.overwrite)
              notes.push(
                `you rewrote all ${st.afterLines} lines of ${rel} but only ${st.changedLines} changed (lines ${describeRanges(st.ranges)}). Make changes like this with fs.edit_file(path, old_string=<the exact current lines>, new_string=<the fixed lines>): it keeps the working parts intact. Another rewrite like this is refused.`,
              );
          }
        }
        const problem = await syntaxProblem(abs, content, rel);
        if (problem) notes.push(problem);
        for (const n of notes) ctx.notice?.(n);
        return `wrote ${rel} (${content.split("\n").length} lines)${notes.length ? `; ${notes.join("; ")}` : ""}`;
      },
      async (a, ctx) => {
        const abs = resolveInside(ctx.workspaceRoot, String(a.path));
        const before = await readText(abs).catch(() => "");
        return { path: relPath(ctx.workspaceRoot, abs), before, after: String(a.content) };
      },
      async (a, ctx) => {
        if (a.overwrite) return null;
        const abs = resolveInside(ctx.workspaceRoot, String(a.path));
        if (!smallRewrites.get(`${ctx.chatId}\u0000${abs}`)) return null;
        const before = await readText(abs).catch(() => null);
        if (before === null) return null;
        const st = rewriteStats(before, String(a.content));
        if (!isSmallFixByRewrite(st)) return null;
        const rel = relPath(ctx.workspaceRoot, abs);
        return `not written: this rewrite of ${rel} changes only ${st.changedLines} of ${st.afterLines} lines (lines ${describeRanges(st.ranges)} of your new content). Change just those lines with fs.edit_file(path="${rel}", old_string=..., new_string=...), one call per changed block. Pass overwrite=True only if the file really has to be replaced as a whole.`;
      },
    ),
    tool(
      {
        id: "fs.edit_file",
        group: "fs",
        description: "Replace an exact text snippet in a file. old_string must match exactly once (including indentation) unless replace_all is true.",
        parameters: {
          type: "object",
          properties: { path: s("File path relative to the project root"), old_string: s("Exact existing text"), new_string: s("Replacement text"), replace_all: b("Replace every occurrence") },
          required: ["path", "old_string", "new_string"],
        },
        returns: "str",
        sideEffects: true,
        editsFiles: true,
      },
      async (a, ctx) => {
        const abs = resolveInside(ctx.workspaceRoot, String(a.path));
        const before = await readText(abs);
        const after = editResult(before, String(a.old_string), String(a.new_string), Boolean(a.replace_all));
        await fs.writeFile(abs, after, "utf8");
        const rel = relPath(ctx.workspaceRoot, abs);
        const problem = await syntaxProblem(abs, after, rel);
        if (problem) ctx.notice?.(problem);
        return `edited ${rel}${problem ? `; ${problem}` : ""}`;
      },
      async (a, ctx) => {
        const abs = resolveInside(ctx.workspaceRoot, String(a.path));
        const before = await readText(abs);
        return { path: relPath(ctx.workspaceRoot, abs), before, after: editResult(before, String(a.old_string), String(a.new_string), Boolean(a.replace_all)) };
      },
    ),
    tool(
      {
        id: "fs.delete",
        group: "fs",
        description: "Delete a file or an empty-or-not directory inside the project.",
        parameters: { type: "object", properties: { path: s("Path relative to the project root") }, required: ["path"] },
        returns: "str",
        sideEffects: true,
      },
      async (a, ctx) => {
        const abs = resolveInside(ctx.workspaceRoot, String(a.path));
        if (path.resolve(abs) === path.resolve(ctx.workspaceRoot)) throw new Error("refusing to delete the project root");
        await fs.rm(abs, { recursive: true });
        return `deleted ${relPath(ctx.workspaceRoot, abs)}`;
      },
    ),
    tool(
      {
        id: "fs.move",
        group: "fs",
        description: "Move or rename a file or directory inside the project.",
        parameters: { type: "object", properties: { src: s("Existing path"), dst: s("New path") }, required: ["src", "dst"] },
        returns: "str",
        sideEffects: true,
      },
      async (a, ctx) => {
        const src = resolveInside(ctx.workspaceRoot, String(a.src));
        const dst = resolveInside(ctx.workspaceRoot, String(a.dst));
        await fs.mkdir(path.dirname(dst), { recursive: true });
        await fs.rename(src, dst);
        return `moved ${relPath(ctx.workspaceRoot, src)} -> ${relPath(ctx.workspaceRoot, dst)}`;
      },
    ),
    tool(
      {
        id: "shell.run",
        group: "shell",
        description:
          "Run a shell command in the project (bash on Linux, PowerShell on Windows) and wait for it. For servers and watchers pass background=True: it returns the first output and a pid; stop it with shell.stop.",
        parameters: {
          type: "object",
          properties: {
            command: s("Command line"),
            cwd: s("Working directory relative to the project root", { default: "." }),
            timeout_sec: i("Timeout in seconds for foreground commands"),
            background: b("Start without waiting (dev servers)"),
            wait_sec: i("For background: seconds to wait before returning output", 6),
          },
          required: ["command"],
        },
        returns: "dict(exit_code, stdout, stderr, timed_out[, pid, running])",
        sideEffects: true,
        timeoutSec: 1800,
      },
      async (a, ctx) => {
        const cwd = resolveInside(ctx.workspaceRoot, String(a.cwd ?? "."));
        const cap = blockingCap();
        if (a.background) return pm.start(String(a.command), cwd, Math.min(Number(a.wait_sec ?? 6), cap));
        const asked = Number(a.timeout_sec ?? svc.shellTimeoutSec());
        const r = await pm.run(String(a.command), cwd, Math.min(asked, cap), ctx.signal);
        if (r.timed_out && asked > cap)
          return { ...r, note: `stopped after ${cap}s to stay inside the cell time limit. Run long commands with background=True and wait with shell.wait(pid=...).` };
        if (r.exit_code !== 0 && !r.timed_out) {
          const hint = errorHint(errorLocations(`${r.stdout}\n${r.stderr}`, ctx.workspaceRoot));
          if (hint) {
            ctx.notice?.(hint);
            return { ...r, hint };
          }
        }
        return r;
      },
    ),
    tool(
      {
        id: "shell.wait",
        group: "shell",
        description:
          "Wait for a background process from shell.run(background=True) to finish, up to timeout_sec (kept under the cell time limit). Returns its status and recent output; call it again while running is True.",
        parameters: { type: "object", properties: { pid: i("Process id from shell.run"), timeout_sec: i("Max seconds to wait in this call") }, required: ["pid"] },
        returns: "dict(pid, running, exit_code, elapsed_sec, silent_sec, output_tail[, note])",
        sideEffects: false,
        timeoutSec: 1800,
      },
      async (a, ctx) => {
        const r = await pm.wait(Number(a.pid), Math.min(Number(a.timeout_sec ?? blockingCap()), blockingCap()), ctx.signal);
        if (!r) throw new Error(`no background process with pid ${a.pid}; see shell.processes()`);
        if (!r.running && r.exit_code !== 0) {
          const hint = errorHint(errorLocations(r.output_tail, ctx.workspaceRoot));
          if (hint) {
            ctx.notice?.(hint);
            return { ...r, hint };
          }
        }
        return r;
      },
    ),
    tool(
      {
        id: "shell.processes",
        group: "shell",
        description: "List background processes started with shell.run(background=True), with their recent output.",
        parameters: { type: "object", properties: {} },
        returns: "list[dict(pid, command, running, exit_code, elapsed_sec, silent_sec, output_tail[, note])]",
        sideEffects: false,
      },
      async () => pm.list(),
    ),
    tool(
      {
        id: "shell.stop",
        group: "shell",
        description: "Stop a background process by pid.",
        parameters: { type: "object", properties: { pid: i("Process id from shell.run") }, required: ["pid"] },
        returns: "bool",
        sideEffects: false,
      },
      async (a) => pm.stop(Number(a.pid)),
    ),
    tool(
      {
        id: "git.status",
        group: "git",
        description: "Short git status of the project.",
        parameters: { type: "object", properties: {} },
        returns: "str",
        sideEffects: false,
      },
      async (_a, ctx) => git(["status", "--short", "--branch"], ctx, pm),
    ),
    tool(
      {
        id: "git.diff",
        group: "git",
        description: "Git diff of working changes (or staged changes).",
        parameters: { type: "object", properties: { path: s("Limit to a path"), staged: b("Show staged changes") } },
        returns: "str",
        sideEffects: false,
      },
      async (a, ctx) => {
        const args = ["diff", "--no-color"];
        if (a.staged) args.push("--staged");
        if (a.path) args.push("--", relPath(ctx.workspaceRoot, resolveInside(ctx.workspaceRoot, String(a.path))));
        return git(args, ctx, pm);
      },
    ),
    tool(
      {
        id: "git.log",
        group: "git",
        description: "Recent commits (one line each).",
        parameters: { type: "object", properties: { n: i("Number of commits", 10) } },
        returns: "str",
        sideEffects: false,
      },
      async (a, ctx) => git(["log", "--oneline", "-n", String(a.n ?? 10)], ctx, pm),
    ),
    tool(
      {
        id: "git.commit",
        group: "git",
        description: "Commit changes. By default stages all changes first.",
        parameters: { type: "object", properties: { message: s("Commit message"), add_all: b("Stage all changes first", true) }, required: ["message"] },
        returns: "str",
        sideEffects: true,
      },
      async (a, ctx) => {
        if (a.add_all !== false) await git(["add", "-A"], ctx, pm);
        return git(["commit", "-m", String(a.message)], ctx, pm);
      },
    ),
    tool(
      {
        id: "git.checkout",
        group: "git",
        description: "Switch branch (or create it).",
        parameters: { type: "object", properties: { ref: s("Branch or commit"), create: b("Create a new branch") }, required: ["ref"] },
        returns: "str",
        sideEffects: true,
      },
      async (a, ctx) => git(a.create ? ["checkout", "-b", String(a.ref)] : ["checkout", String(a.ref)], ctx, pm),
    ),
    tool(
      {
        id: "editor.get_context",
        group: "editor",
        description: "What the user sees in the editor: active file, open files and selected text.",
        parameters: { type: "object", properties: {} },
        returns: "dict(active_file, open_files, selection)",
        sideEffects: false,
      },
      async () => {
        const e = svc.editorState();
        return { active_file: e.activeFile ?? null, open_files: e.openFiles ?? [], selection: e.selection ?? "" };
      },
    ),
    tool(
      {
        id: "diagnostics.get",
        group: "editor",
        description: "Problems (errors, warnings) the editor currently reports, optionally for one file.",
        parameters: { type: "object", properties: { path: s("Only this file") } },
        returns: "list[dict(path, line, severity, message)]",
        sideEffects: false,
      },
      async (a) => {
        const d = svc.editorState().diagnostics ?? [];
        return a.path ? d.filter((x) => x.path === a.path) : d;
      },
    ),
    tool(
      {
        id: "session.read",
        group: "session",
        description: "Search this chat's full stored history, including turns folded into the checkpoint, for exact earlier details.",
        parameters: { type: "object", properties: { query: s("Words or identifier to look for"), limit: i("Maximum excerpts", 8) }, required: ["query"] },
        returns: "list[dict(turn, role, excerpt)]",
        sideEffects: false,
      },
      async (a, ctx) => svc.searchSession(ctx.chatId, String(a.query), Number(a.limit ?? 8)),
    ),
    ...(svc.checkPage ? [webCheck(svc.checkPage)] : []),
    ...(svc.browser ? browserTools(svc.browser) : []),
  ];
}

function browserTools(drive: NonNullable<ToolHostServices["browser"]>): ToolImpl[] {
  const target = s("Element ref from browser.snapshot (e.g. 'e12') or a CSS selector");
  const op = (name: string) => (a: Record<string, unknown>) => drive(name, a);
  return [
    tool(
      {
        id: "browser.navigate",
        group: "browser",
        description:
          "Open a URL in the IDE's built-in browser, which the user can see. Accepts http(s) URLs, localhost addresses and project file paths (e.g. 'site/index.html'). Returns the page text, interactive elements (with refs like e3) and console errors.",
        parameters: { type: "object", properties: { url: s("URL, host name or project-relative HTML path"), wait_ms: i("Extra time to let the page settle", 600) }, required: ["url"] },
        returns: "dict(url, title, error, console, text, elements)",
        sideEffects: true,
        timeoutSec: 60,
      },
      async (a, ctx) => {
        const u = String(a.url).trim();
        let url = u;
        if (!/^[a-z][\w+.-]*:/i.test(u)) {
          try {
            const abs = resolveInside(ctx.workspaceRoot, u);
            await fs.access(abs);
            url = abs;
          } catch {
            /* not a project file: a host name or search */
          }
        }
        return drive("navigate", { ...a, url });
      },
    ),
    tool(
      {
        id: "browser.snapshot",
        group: "browser",
        description: "Read the current page in the built-in browser: URL, title, headings, visible text and a numbered list of interactive elements (refs e1, e2, ...) to use with browser.click / browser.type.",
        parameters: { type: "object", properties: { max_text: i("Max characters of page text", 4000) } },
        returns: "dict(url, title, headings, elements, text, scroll, console)",
        sideEffects: false,
      },
      op("snapshot"),
    ),
    tool(
      {
        id: "browser.click",
        group: "browser",
        description: "Click an element in the built-in browser with a real mouse event. Returns whether the page navigated and new console errors.",
        parameters: { type: "object", properties: { target }, required: ["target"] },
        returns: "dict(clicked, url, navigated, console)",
        sideEffects: true,
        repeatable: true,
      },
      op("click"),
    ),
    tool(
      {
        id: "browser.type",
        group: "browser",
        description: "Type text into an input, textarea or editable element in the built-in browser. Clears it first unless clear=False; submit=True presses Enter afterwards.",
        parameters: { type: "object", properties: { target, text: s("Text to type"), clear: b("Clear the field first", true), submit: b("Press Enter after typing") }, required: ["target", "text"] },
        returns: "dict(typed_into, url, console)",
        sideEffects: true,
        repeatable: true,
      },
      op("type"),
    ),
    tool(
      {
        id: "browser.press",
        group: "browser",
        description: "Press a key in the built-in browser (e.g. 'Enter', 'Escape', 'Tab', 'ArrowDown', 'a').",
        parameters: { type: "object", properties: { key: s("Key name") }, required: ["key"] },
        returns: "dict(pressed, url)",
        sideEffects: true,
        repeatable: true,
      },
      op("press"),
    ),
    tool(
      {
        id: "browser.scroll",
        group: "browser",
        description: "Scroll the page in the built-in browser by dy pixels (negative scrolls up).",
        parameters: { type: "object", properties: { dy: i("Pixels to scroll", 600) } },
        returns: "dict(y, height)",
        sideEffects: false,
      },
      op("scroll"),
    ),
    tool(
      {
        id: "browser.eval",
        group: "browser",
        description: "Evaluate a JavaScript expression in the page (like the DevTools console) and return its JSON value. Use for reading state the snapshot does not show.",
        parameters: { type: "object", properties: { code: s("JavaScript expression, e.g. 'document.querySelectorAll(\".item\").length'") }, required: ["code"] },
        returns: "JSON value",
        sideEffects: true,
      },
      op("eval"),
    ),
    tool(
      {
        id: "browser.screenshot",
        group: "browser",
        description: "Capture the visible page of the built-in browser. The image is shown to the user in the chat; returns its reference and size.",
        parameters: { type: "object", properties: {} },
        returns: "dict(image, width, height, url, title)",
        sideEffects: false,
      },
      op("screenshot"),
    ),
    tool(
      {
        id: "browser.console",
        group: "browser",
        description: "Recent console messages (errors, warnings, logs) of the page in the built-in browser.",
        parameters: { type: "object", properties: { limit: i("Max messages", 40) } },
        returns: "list[str]",
        sideEffects: false,
      },
      op("console"),
    ),
  ];
}

function webCheck(checkPage: NonNullable<ToolHostServices["checkPage"]>): ToolImpl {
  return tool(
    {
      id: "web.check",
      group: "web",
      description:
        "Open an HTML page from the project in a hidden, offline browser, like a developer checking DevTools. Reports console errors and uncaught exceptions, missing local files, the visible text, element counts and text that is unreadable against its background. Optionally clicks elements (CSS selectors, in order) and reports whether the page text changed. Use it to verify web pages you build.",
      parameters: {
        type: "object",
        properties: {
          path: s("HTML file relative to the project root"),
          click: { type: "array", items: { type: "string" }, description: "CSS selectors to click in order, e.g. ['#next', '.tab:nth-child(2)']" },
          wait_ms: i("How long to let scripts run after load", 800),
        },
        required: ["path"],
      },
      returns: "dict(loaded, errors, warnings, missing_files, blocked_requests, title, text, counts, clicks, unreadable_text)",
      sideEffects: false,
      timeoutSec: 40,
    },
    async (a, ctx) => {
      const abs = resolveInside(ctx.workspaceRoot, String(a.path));
      await fs.access(abs).catch(() => {
        throw new Error(`${String(a.path)} does not exist`);
      });
      const clicks = Array.isArray(a.click) ? a.click.map(String) : typeof a.click === "string" ? [a.click] : [];
      const r = await checkPage({ path: abs, waitMs: Math.min(5000, Math.max(100, Number(a.wait_ms ?? 800))), clicks });
      const root = ctx.workspaceRoot.endsWith(path.sep) ? ctx.workspaceRoot : ctx.workspaceRoot + path.sep;
      const missing = ((r.missingFiles as string[]) ?? []).map((f) => (f.startsWith(root) ? f.slice(root.length) : f));
      return {
        loaded: r.loaded,
        ...(r.loadError ? { load_error: r.loadError } : {}),
        errors: r.errors,
        warnings: r.warnings,
        missing_files: missing,
        blocked_requests: r.blockedRequests,
        title: r.title,
        text: r.text,
        counts: r.counts,
        clicks: r.clicks,
        unreadable_text: r.unreadableText ?? [],
      };
    },
  );
}

export class ToolRegistry implements ToolRegistryPort {
  private readonly byId: Map<string, ToolImpl>;
  constructor(tools: ToolImpl[]) {
    this.byId = new Map(tools.map((t) => [t.descriptor.id, t]));
  }
  list(): ToolDescriptor[] {
    return [...this.byId.values()].map((t) => t.descriptor);
  }
  get(id: string): ToolImpl | undefined {
    return this.byId.get(id);
  }
}
