import { promises as fs, watch, type FSWatcher } from "node:fs";
import path from "node:path";
import { IGNORED_DIRS, isProbablyBinary, relPath, resolveInside, walk } from "@rlm/adapters";
import type { DirEntry, SearchHit } from "@rlm/contracts";

/** File access for the renderer, always confined to the open workspace. */
export class FsService {
  private root: string | null = null;
  private watcher: FSWatcher | null = null;
  private debounce = new Map<string, NodeJS.Timeout>();

  constructor(private readonly onChanged: (p: string) => void) {}

  open(root: string): void {
    this.root = path.resolve(root);
    this.watcher?.close();
    try {
      this.watcher = watch(this.root, { recursive: true }, (_ev, file) => {
        if (!file) return;
        const rel = file.toString().split(path.sep).join("/");
        if (rel.split("/").some((seg) => IGNORED_DIRS.has(seg))) return;
        const dir = path.posix.dirname(rel);
        clearTimeout(this.debounce.get(dir));
        this.debounce.set(
          dir,
          setTimeout(() => this.onChanged(rel), 150),
        );
      });
    } catch {
      this.watcher = null;
    }
  }

  current(): string | null {
    return this.root;
  }

  private abs(p: string): string {
    if (!this.root) throw new Error("no workspace open");
    return resolveInside(this.root, p);
  }

  async readDir(p: string): Promise<DirEntry[]> {
    const abs = this.abs(p);
    const entries = await fs.readdir(abs, { withFileTypes: true });
    return entries
      .filter((e) => e.name !== ".git")
      .map((e) => ({ name: e.name, path: relPath(this.root!, path.join(abs, e.name)), kind: e.isDirectory() ? ("dir" as const) : ("file" as const) }))
      .sort((a, b) => (a.kind === b.kind ? a.name.localeCompare(b.name) : a.kind === "dir" ? -1 : 1));
  }

  async readFile(p: string): Promise<{ content: string; binary: boolean; size: number }> {
    const abs = this.abs(p);
    const st = await fs.stat(abs);
    if (st.size > 20_000_000 || (await isProbablyBinary(abs))) return { content: "", binary: true, size: st.size };
    return { content: await fs.readFile(abs, "utf8"), binary: false, size: st.size };
  }

  async writeFile(p: string, content: string): Promise<void> {
    const abs = this.abs(p);
    await fs.mkdir(path.dirname(abs), { recursive: true });
    await fs.writeFile(abs, content, "utf8");
  }

  async create(p: string, kind: "file" | "dir"): Promise<void> {
    const abs = this.abs(p);
    if (kind === "dir") await fs.mkdir(abs, { recursive: true });
    else {
      await fs.mkdir(path.dirname(abs), { recursive: true });
      await fs.writeFile(abs, "", { flag: "wx" });
    }
  }

  async rename(from: string, to: string): Promise<void> {
    await fs.rename(this.abs(from), this.abs(to));
  }

  async delete(p: string): Promise<void> {
    const abs = this.abs(p);
    if (abs === this.root) throw new Error("refusing to delete the workspace root");
    await fs.rm(abs, { recursive: true });
  }

  async listFiles(limit: number): Promise<string[]> {
    if (!this.root) return [];
    const out: string[] = [];
    for await (const f of walk(this.root, this.root, { n: limit })) out.push(relPath(this.root, f));
    return out;
  }

  async search(query: string, maxResults: number): Promise<SearchHit[]> {
    if (!this.root) return [];
    const hits: SearchHit[] = [];
    const needle = query.toLowerCase();
    for await (const f of walk(this.root, this.root, { n: 20000 })) {
      const st = await fs.stat(f).catch(() => null);
      if (!st || st.size > 2_000_000) continue;
      if (await isProbablyBinary(f).catch(() => true)) continue;
      const lines = (await fs.readFile(f, "utf8")).split("\n");
      for (let i = 0; i < lines.length; i++) {
        if (lines[i]!.toLowerCase().includes(needle)) {
          hits.push({ path: relPath(this.root, f), line: i + 1, text: lines[i]!.trim().slice(0, 240) });
          if (hits.length >= maxResults) return hits;
        }
      }
    }
    return hits;
  }
}
