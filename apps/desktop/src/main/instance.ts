import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readdirSync, readFileSync, realpathSync, renameSync, rmSync, statSync, unlinkSync, writeFileSync } from "node:fs";
import path from "node:path";

/** Several copies of the IDE may run on one profile, one project folder each. The first copy (the one holding the
 * app's single-instance lock) runs the shared background services; the others use them. */

export function isAlive(pid: number): boolean {
  if (!Number.isInteger(pid) || pid <= 0) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch (e) {
    // EPERM: the process exists but belongs to someone else.
    return (e as NodeJS.ErrnoException).code === "EPERM";
  }
}

/** Windows and macOS paths are case-insensitive: one folder must map to one project however it was typed. */
export function canonicalFolder(root: string, platform: NodeJS.Platform = process.platform): string {
  const P = platform === "win32" ? path.win32 : path.posix;
  let p = P.resolve(root);
  try {
    if (platform === process.platform) p = realpathSync.native(p);
  } catch {
    /* keep the resolved path; the caller reports a missing folder */
  }
  if (platform === "win32" && /^[a-z]:/.test(p)) p = p[0]!.toUpperCase() + p.slice(1);
  return p;
}

function folderKey(root: string, platform: NodeJS.Platform): string {
  const k = platform === "win32" || platform === "darwin" ? root.toLowerCase() : root;
  return createHash("sha1").update(k).digest("hex").slice(0, 20);
}

/** Replaces a file atomically. Windows refuses to rename over a file another process has open for a moment. */
export function writeFileAtomic(file: string, data: string, mode?: number): void {
  mkdirSync(path.dirname(file), { recursive: true });
  const tmp = `${file}.${process.pid}.tmp`;
  writeFileSync(tmp, data, mode === undefined ? undefined : { mode });
  for (let i = 0; ; i++) {
    try {
      renameSync(tmp, file);
      return;
    } catch (e) {
      const code = (e as NodeJS.ErrnoException).code;
      if (i >= 40 || !(code === "EPERM" || code === "EBUSY" || code === "EACCES")) {
        try {
          unlinkSync(tmp);
        } catch {
          /* already gone */
        }
        throw e;
      }
      Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 25);
    }
  }
}

/** One project folder is open in at most one copy of the IDE, so its chats and scheduled wake-ups never run twice. */
export class FolderLocks {
  private held: string | null = null;

  constructor(
    private readonly dir: string,
    private readonly platform: NodeJS.Platform = process.platform,
  ) {}

  private file(root: string): string {
    return path.join(this.dir, `${folderKey(root, this.platform)}.json`);
  }

  /** Takes the folder for this process; returns the pid of another live copy that has it open instead. */
  acquire(root: string): number | null {
    const f = this.file(root);
    try {
      const other = JSON.parse(readFileSync(f, "utf8")) as { pid: number };
      if (other.pid !== process.pid && isAlive(other.pid)) return other.pid;
    } catch {
      /* free, or a stale/corrupt lock */
    }
    writeFileAtomic(f, JSON.stringify({ pid: process.pid, root }));
    if (this.held && this.held !== f) this.drop(this.held);
    this.held = f;
    return null;
  }

  releaseAll(): void {
    if (this.held) this.drop(this.held);
    this.held = null;
  }

  private drop(f: string): void {
    try {
      const cur = JSON.parse(readFileSync(f, "utf8")) as { pid: number };
      if (cur.pid === process.pid) unlinkSync(f);
    } catch {
      /* already gone */
    }
  }
}

/** Chromium keeps caches and local storage in the session folder and locks it (hard locks on Windows). Every copy
 * after the first gets its own; folders of copies that have exited are removed. */
export function secondarySessionDir(userData: string): string {
  const root = path.join(userData, "instances");
  mkdirSync(root, { recursive: true });
  for (const name of readdirSync(root)) {
    const pid = Number(name);
    if (pid !== process.pid && !isAlive(pid)) {
      try {
        rmSync(path.join(root, name), { recursive: true, force: true, maxRetries: 3 });
      } catch {
        /* still locked; next start cleans it */
      }
    }
  }
  const dir = path.join(root, String(process.pid));
  mkdirSync(dir, { recursive: true });
  return dir;
}

/** A folder passed on the command line ("RLM IDE.exe C:\work\app", Explorer's "Open with", a new-window request). */
export function folderFromArgv(argv: string[], cwd: string, appPath: string): string | null {
  const app = path.resolve(appPath);
  for (const a of argv.slice(1)) {
    if (!a || a.startsWith("-")) continue;
    const p = path.resolve(cwd, a);
    if (p === app || /\.(asar|js|cjs|mjs|exe)$/i.test(p)) continue;
    try {
      if (existsSync(p) && statSync(p).isDirectory()) return p;
    } catch {
      /* not a folder */
    }
  }
  return null;
}
