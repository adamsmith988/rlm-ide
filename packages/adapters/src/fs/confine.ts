import { promises as fs, realpathSync } from "node:fs";
import path from "node:path";

export class PathEscapeError extends Error {}

/** Directories never walked by listing/search tools. */
export const IGNORED_DIRS = new Set([".git", "node_modules", "dist", "out", "build", ".next", ".vite", "__pycache__", ".venv", "venv", "target", "coverage", ".cache"]);

/** Resolve `p` inside `root`, rejecting `..` escapes and symlinks that point outside. */
export function resolveInside(root: string, p: string): string {
  const realRoot = realpathSync(root);
  const abs = path.resolve(realRoot, p || ".");
  const rel = path.relative(realRoot, abs);
  if (rel.startsWith("..") || path.isAbsolute(rel)) throw new PathEscapeError(`path '${p}' is outside the workspace`);
  // Resolve the deepest existing ancestor through symlinks.
  let probe = abs;
  for (;;) {
    try {
      const real = realpathSync(probe);
      const r = path.relative(realRoot, real);
      if (r.startsWith("..") || path.isAbsolute(r)) throw new PathEscapeError(`path '${p}' resolves outside the workspace`);
      break;
    } catch (e) {
      if (e instanceof PathEscapeError) throw e;
      const parent = path.dirname(probe);
      if (parent === probe) break;
      probe = parent;
    }
  }
  return abs;
}

export function relPath(root: string, abs: string): string {
  return path.relative(realpathSync(root), abs).split(path.sep).join("/") || ".";
}

export async function* walk(root: string, dir: string, limit: { n: number }): AsyncGenerator<string> {
  let entries: import("node:fs").Dirent[];
  try {
    entries = await fs.readdir(dir, { withFileTypes: true });
  } catch {
    return;
  }
  entries.sort((a, b) => a.name.localeCompare(b.name));
  for (const e of entries) {
    if (limit.n <= 0) return;
    const full = path.join(dir, e.name);
    if (e.isDirectory()) {
      if (IGNORED_DIRS.has(e.name)) continue;
      yield* walk(root, full, limit);
    } else if (e.isFile()) {
      limit.n--;
      yield full;
    }
  }
}

/** Minimal glob -> RegExp (supports **, *, ?, {a,b}). Matched against workspace-relative paths. */
export function globToRegExp(glob: string): RegExp {
  let re = "";
  for (let i = 0; i < glob.length; i++) {
    const c = glob[i]!;
    if (c === "*") {
      if (glob[i + 1] === "*") {
        re += glob[i + 2] === "/" ? "(?:.*/)?" : ".*";
        i += glob[i + 2] === "/" ? 2 : 1;
      } else re += "[^/]*";
    } else if (c === "?") re += "[^/]";
    else if (c === "{") {
      const end = glob.indexOf("}", i);
      if (end > i) {
        re += `(?:${glob.slice(i + 1, end).split(",").map(escapeRe).join("|")})`;
        i = end;
      } else re += "\\{";
    } else re += escapeRe(c);
  }
  return new RegExp(`^${re}$`);
}

function escapeRe(s: string): string {
  return s.replace(/[.+^${}()|[\]\\]/g, "\\$&");
}

export async function isProbablyBinary(file: string): Promise<boolean> {
  const fh = await fs.open(file, "r");
  try {
    const buf = Buffer.alloc(4096);
    const { bytesRead } = await fh.read(buf, 0, 4096, 0);
    return buf.subarray(0, bytesRead).includes(0);
  } finally {
    await fh.close();
  }
}
