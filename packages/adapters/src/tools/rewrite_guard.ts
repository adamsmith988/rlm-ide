import path from "node:path";
export { REWRITE_MAX_CHANGED, REWRITE_MIN_LINES, describeRanges, isSmallFixByRewrite, rewriteStats, type RewriteStats } from "@rlm/core";

export interface ErrorLocation {
  path: string;
  line: number;
}

const PY_FRAME = /File "([^"]+)", line (\d+)/g;
const JS_FRAME = /\(?((?:\/|[A-Za-z]:\\|\.{1,2}\/)?[^\s():]+\.(?:m?[jt]sx?|cjs|py|rs|go|java|rb|php|c|cc|cpp|h|hpp|cs|kt|swift)):(\d+)(?::\d+)?\)?/g;

/** Source locations inside the project that an error output points at, most relevant first. */
export function errorLocations(output: string, root: string, max = 3): ErrorLocation[] {
  const inside = (p: string): string | null => {
    const abs = path.resolve(root, p);
    const rel = path.relative(root, abs);
    if (!rel || rel.startsWith("..") || path.isAbsolute(rel)) return null;
    if (/(^|[\\/])(node_modules|site-packages|dist-packages|\.venv|venv|__pycache__)([\\/]|$)/.test(rel)) return null;
    return rel.split(path.sep).join("/");
  };
  const found: ErrorLocation[] = [];
  const push = (p: string, line: number) => {
    const rel = inside(p);
    if (rel && line > 0 && !found.some((f) => f.path === rel && f.line === line)) found.push({ path: rel, line });
  };
  const py = [...output.matchAll(PY_FRAME)];
  // Python prints the innermost frame last; that is where the fix usually goes.
  for (const m of py.reverse()) push(m[1]!, Number(m[2]));
  if (!py.length) for (const m of output.matchAll(JS_FRAME)) push(m[1]!, Number(m[2]));
  return found.slice(0, max);
}

export function errorHint(locs: ErrorLocation[]): string {
  if (!locs.length) return "";
  const first = locs[0]!;
  const others = locs.slice(1).map((l) => `${l.path}:${l.line}`).join(", ");
  return `The error points at ${first.path} line ${first.line}${others ? ` (also ${others})` : ""}. Read the lines around it (fs.read_file with start_line/end_line), then change only the faulty lines with fs.edit_file instead of rewriting the file.`;
}
