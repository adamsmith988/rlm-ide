import type { Severity } from "./types";

/** What the reporter knows. Everything except the title is optional. */
export interface BugReportInput {
  title: string;
  summary?: string;
  severity?: Severity;
  /** Error message and stack trace, verbatim. */
  error?: string;
  file?: string;
  line?: number;
  /** Branch or commit the bug was seen on (defaults to the project's default branch). */
  ref?: string;
  /** Commit the reporter blames explicitly (skips blame lookup). */
  commit?: string;
  steps?: string[];
  expected?: string;
  actual?: string;
  environment?: Record<string, string>;
  logs?: string;
  /** Extra markdown (screenshots, links, notes). */
  notes?: string;
}

/** What the hub found out by looking at the repository. */
export interface BugContext {
  project: string;
  number: number;
  reporter: { name: string; email: string };
  createdAt: number;
  ref: string;
  blame: { commit: string; author: string; email: string; time: number; summary: string } | null;
  excerpt: { startLine: number; lines: string[] } | null;
  commitInfo: { commit: string; author: string; email: string; time: number; subject: string; files: string[] } | null;
  routedTo: { name: string; email: string; via: string } | null;
  pull: { number: number; title: string; author: string } | null;
  pushedBy: { name: string; at: number } | null;
}

const LANG: Record<string, string> = {
  ts: "ts", tsx: "tsx", js: "js", jsx: "jsx", mjs: "js", cjs: "js", py: "python", rs: "rust", go: "go", java: "java", kt: "kotlin",
  c: "c", h: "c", cpp: "cpp", hpp: "cpp", cs: "csharp", rb: "ruby", php: "php", swift: "swift", css: "css", html: "html", json: "json",
  yml: "yaml", yaml: "yaml", md: "markdown", sh: "bash", sql: "sql", vue: "vue", svelte: "svelte",
};

export function langOf(file: string): string {
  return LANG[file.split(".").pop()?.toLowerCase() ?? ""] ?? "";
}

function fence(body: string, lang = ""): string {
  const ticks = body.includes("```") ? "~~~~" : "```";
  return `${ticks}${lang}\n${body.replace(/\s+$/, "")}\n${ticks}`;
}

const iso = (t: number) => new Date(t).toISOString().replace("T", " ").slice(0, 16) + " UTC";
const short = (sha: string) => sha.slice(0, 10);

/** The complete bug report a developer receives. Every section the data supports is filled in. */
export function renderBugReport(input: BugReportInput, ctx: BugContext): string {
  const sev = input.severity ?? "medium";
  const where = input.file ? `\`${input.file}${input.line ? `:${input.line}` : ""}\`` : "not specified";
  const out: string[] = [];
  out.push(`# Bug #${ctx.number}: ${input.title.trim()}`);
  out.push("");
  out.push("| Field | Details |", "|---|---|");
  out.push(`| **Project** | ${ctx.project} |`);
  out.push(`| **Severity** | ${sev} |`);
  out.push(`| **Reported by** | ${ctx.reporter.name} <${ctx.reporter.email}> |`);
  out.push(`| **Reported at** | ${iso(ctx.createdAt)} |`);
  out.push(`| **Seen on** | \`${ctx.ref}\` |`);
  out.push(`| **Location** | ${where} |`);
  if (ctx.routedTo) out.push(`| **Assigned to** | ${ctx.routedTo.name} <${ctx.routedTo.email}> (${ctx.routedTo.via}) |`);
  out.push("");

  out.push("## Summary", "", input.summary?.trim() || input.title.trim(), "");

  if (input.error?.trim()) out.push("## Error", "", fence(input.error.trim(), "text"), "");

  if (ctx.excerpt && input.file) {
    const width = String(ctx.excerpt.startLine + ctx.excerpt.lines.length).length;
    const code = ctx.excerpt.lines.map((l, i) => {
      const n = ctx.excerpt!.startLine + i;
      return `${n === input.line ? ">" : " "} ${String(n).padStart(width)} | ${l}`;
    });
    out.push(`## Code at ${where} (\`${ctx.ref}\`)`, "", fence(code.join("\n"), langOf(input.file)), "");
  }

  if (ctx.blame || ctx.commitInfo || ctx.pull || ctx.pushedBy) {
    out.push("## Origin", "");
    if (ctx.blame) {
      out.push(`- **Last change to this line:** commit \`${short(ctx.blame.commit)}\` by ${ctx.blame.author} <${ctx.blame.email}> on ${iso(ctx.blame.time)}`);
      out.push(`- **Commit message:** ${ctx.blame.summary}`);
    }
    if (ctx.commitInfo && (!ctx.blame || ctx.commitInfo.commit !== ctx.blame.commit)) {
      out.push(`- **Commit:** \`${short(ctx.commitInfo.commit)}\` "${ctx.commitInfo.subject}" by ${ctx.commitInfo.author} <${ctx.commitInfo.email}> on ${iso(ctx.commitInfo.time)}`);
    }
    if (ctx.commitInfo?.files.length) out.push(`- **Files changed in that commit:** ${ctx.commitInfo.files.slice(0, 15).map((f) => `\`${f}\``).join(", ")}${ctx.commitInfo.files.length > 15 ? ", …" : ""}`);
    if (ctx.pull) out.push(`- **Merged through pull request** !${ctx.pull.number} "${ctx.pull.title}" by ${ctx.pull.author}`);
    if (ctx.pushedBy) out.push(`- **Pushed to the hub by** ${ctx.pushedBy.name} on ${iso(ctx.pushedBy.at)}`);
    out.push("");
  }

  if (input.steps?.length) out.push("## Steps to reproduce", "", ...input.steps.map((s, i) => `${i + 1}. ${s.trim()}`), "");
  if (input.expected?.trim() || input.actual?.trim()) {
    out.push("## Expected vs actual", "");
    if (input.expected?.trim()) out.push(`**Expected:** ${input.expected.trim()}`, "");
    if (input.actual?.trim()) out.push(`**Actual:** ${input.actual.trim()}`, "");
  }
  const env = Object.entries(input.environment ?? {}).filter(([, v]) => v);
  if (env.length) out.push("## Environment", "", "| Key | Value |", "|---|---|", ...env.map(([k, v]) => `| ${k} | ${String(v).replace(/\|/g, "\\|")} |`), "");
  if (input.logs?.trim()) out.push("## Logs", "", fence(input.logs.trim().slice(-20000), "text"), "");
  if (input.notes?.trim()) out.push("## Notes", "", input.notes.trim(), "");

  out.push("## For the developer", "");
  out.push(
    `- [ ] Reproduce on \`${ctx.ref}\``,
    `- [ ] Fix${input.file ? ` ${where}` : ""} and add a test that fails without the fix`,
    `- [ ] Reply on this report with the fix commit or pull request`,
  );
  out.push("");
  return out.join("\n");
}
