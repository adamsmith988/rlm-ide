import { mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { canonicalFolder, FolderLocks, folderFromArgv, isAlive, secondarySessionDir, writeFileAtomic } from "../src/main/instance";

const dirs: string[] = [];
const tmp = () => {
  const d = mkdtempSync(path.join(tmpdir(), "rlm-instance-"));
  dirs.push(d);
  return d;
};
afterEach(() => {
  for (const d of dirs.splice(0)) rmSync(d, { recursive: true, force: true });
});

/** A pid no process has: above every platform's pid range. */
const DEAD_PID = 0x7ffffff0;

describe("canonicalFolder", () => {
  it("gives one spelling for a Windows folder however its drive letter was typed", () => {
    expect(canonicalFolder("c:\\Work\\App\\", "win32")).toBe("C:\\Work\\App");
    expect(canonicalFolder("C:/Work/App", "win32")).toBe("C:\\Work\\App");
  });

  it("resolves symlinks and relative parts on this machine", () => {
    const d = tmp();
    mkdirSync(path.join(d, "real"));
    expect(canonicalFolder(path.join(d, "real", "..", "real"))).toBe(canonicalFolder(path.join(d, "real")));
  });
});

describe("FolderLocks", () => {
  const lockFile = (dir: string) => path.join(dir, readdirSync(dir)[0]!);

  it("lets one live process hold a folder, case-insensitively on Windows", () => {
    const dir = tmp();
    const mine = new FolderLocks(dir, "win32");
    expect(mine.acquire("C:\\Work\\App")).toBeNull();
    expect(mine.acquire("C:\\Work\\App")).toBeNull();
    // Pretend another live copy of the IDE (our parent process) holds it.
    writeFileSync(lockFile(dir), JSON.stringify({ pid: process.ppid, root: "C:\\Work\\App" }));
    expect(new FolderLocks(dir, "win32").acquire("c:\\work\\app")).toBe(process.ppid);
  });

  it("keeps case-sensitive folders apart on Linux", () => {
    const dir = tmp();
    expect(new FolderLocks(dir, "linux").acquire("/work/App")).toBeNull();
    writeFileSync(lockFile(dir), JSON.stringify({ pid: process.ppid, root: "/work/App" }));
    expect(new FolderLocks(dir, "linux").acquire("/work/app")).toBeNull();
  });

  it("takes over a lock left by a process that has exited, or a corrupt one", () => {
    const dir = tmp();
    new FolderLocks(dir, "linux").acquire("/w");
    writeFileSync(lockFile(dir), JSON.stringify({ pid: DEAD_PID, root: "/w" }));
    expect(new FolderLocks(dir, "linux").acquire("/w")).toBeNull();
    writeFileSync(lockFile(dir), "{not json");
    expect(new FolderLocks(dir, "linux").acquire("/w")).toBeNull();
  });

  it("frees the previous folder when another is opened, and everything on release", () => {
    const dir = tmp();
    const locks = new FolderLocks(dir, "linux");
    locks.acquire("/a");
    locks.acquire("/b");
    expect(readdirSync(dir)).toHaveLength(1);
    expect(JSON.parse(readFileSync(lockFile(dir), "utf8")).root).toBe("/b");
    locks.releaseAll();
    expect(readdirSync(dir)).toHaveLength(0);
  });
});

describe("process helpers", () => {
  it("knows which processes are alive", () => {
    expect(isAlive(process.pid)).toBe(true);
    expect(isAlive(DEAD_PID)).toBe(false);
    expect(isAlive(0)).toBe(false);
  });

  it("writes files atomically, replacing what was there", () => {
    const f = path.join(tmp(), "sub", "settings.json");
    writeFileAtomic(f, "one");
    writeFileAtomic(f, "two");
    expect(readFileSync(f, "utf8")).toBe("two");
    expect(readdirSync(path.dirname(f))).toEqual(["settings.json"]);
  });

  it("gives each later copy its own session folder and removes those of exited copies", () => {
    const user = tmp();
    mkdirSync(path.join(user, "instances", String(DEAD_PID)), { recursive: true });
    const mine = secondarySessionDir(user);
    expect(mine).toBe(path.join(user, "instances", String(process.pid)));
    expect(readdirSync(path.join(user, "instances"))).toEqual([String(process.pid)]);
  });

  it("finds a folder passed on the command line, skipping flags and the app itself", () => {
    const d = tmp();
    mkdirSync(path.join(d, "proj"));
    const app = path.join(d, "app");
    mkdirSync(app);
    expect(folderFromArgv(["electron", "--inspect", app, "proj"], d, app)).toBe(path.join(d, "proj"));
    expect(folderFromArgv(["RLM IDE.exe", "--remote-debugging-port=9222"], d, app)).toBeNull();
    expect(folderFromArgv(["electron", "missing"], d, app)).toBeNull();
  });
});
