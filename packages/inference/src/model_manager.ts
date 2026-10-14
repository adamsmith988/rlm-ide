// Plain, erasable TypeScript only (no enums / parameter properties): this file is also run
// directly by `node scripts/fetch-models.mjs` through Node's type stripping.
import { createHash } from "node:crypto";
import { createReadStream, createWriteStream, existsSync, promises as fs, statSync } from "node:fs";
import path from "node:path";

export interface ManifestFile {
  path: string;
  size: number | null;
  sha256: string | null;
}

export interface ModelManifestEntry {
  repo: string;
  revision: string;
  dir: string;
  files: ManifestFile[];
}

export type ModelManifest = Record<string, ModelManifestEntry>;

export interface Progress {
  modelId: string;
  file: string;
  received: number;
  total: number;
  done: boolean;
  error?: string;
}

export const HF_BASE = "https://huggingface.co";

export async function loadManifest(file: string): Promise<ModelManifest> {
  return JSON.parse(await fs.readFile(file, "utf8")) as ModelManifest;
}

export function totalBytes(entry: ModelManifestEntry): number {
  return entry.files.reduce((n, f) => n + (f.size ?? 0), 0);
}

/** Installed = every manifest file present with the expected size. (sha256 is checked at download.) */
export function isInstalled(root: string, entry: ModelManifestEntry): boolean {
  return entry.files.every((f) => {
    const p = path.join(root, entry.dir, f.path);
    if (!existsSync(p)) return false;
    return f.size === null || statSync(p).size === f.size;
  });
}

export function bytesOnDisk(root: string, entry: ModelManifestEntry): number {
  let n = 0;
  for (const f of entry.files) {
    for (const p of [path.join(root, entry.dir, f.path), path.join(root, entry.dir, `${f.path}.part`)]) {
      if (existsSync(p)) n += statSync(p).size;
    }
  }
  return n;
}

async function sha256File(p: string): Promise<string> {
  const h = createHash("sha256");
  await new Promise<void>((resolve, reject) => {
    createReadStream(p)
      .on("data", (d) => h.update(d))
      .on("end", () => resolve())
      .on("error", reject);
  });
  return h.digest("hex");
}

/** Download every file of a model with HTTP range resume, then verify sha256 for LFS files. */
export async function downloadModel(
  modelId: string,
  entry: ModelManifestEntry,
  root: string,
  onProgress: (p: Progress) => void,
  signal?: AbortSignal,
): Promise<void> {
  const total = totalBytes(entry);
  let base = 0;
  for (const f of entry.files) {
    const dest = path.join(root, entry.dir, f.path);
    await fs.mkdir(path.dirname(dest), { recursive: true });
    if (existsSync(dest) && (f.size === null || statSync(dest).size === f.size)) {
      base += f.size ?? 0;
      onProgress({ modelId, file: f.path, received: base, total, done: false });
      continue;
    }
    const part = `${dest}.part`;
    const have = existsSync(part) ? statSync(part).size : 0;
    const url = `${HF_BASE}/${entry.repo}/resolve/${entry.revision}/${f.path}`;
    const res = await fetch(url, { headers: have ? { Range: `bytes=${have}-` } : {}, signal, redirect: "follow" });
    if (!res.ok && res.status !== 206) throw new Error(`download ${f.path} failed: HTTP ${res.status}`);
    const append = res.status === 206 && have > 0;
    const out = createWriteStream(part, { flags: append ? "a" : "w" });
    let received = append ? have : 0;
    const reader = res.body!.getReader();
    let lastReport = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (!out.write(value)) await new Promise<void>((r) => out.once("drain", () => r()));
      received += value.byteLength;
      const now = Date.now();
      if (now - lastReport > 250) {
        lastReport = now;
        onProgress({ modelId, file: f.path, received: base + received, total, done: false });
      }
    }
    await new Promise<void>((resolve, reject) => out.end((e?: Error | null) => (e ? reject(e) : resolve())));
    if (f.size !== null && statSync(part).size !== f.size) throw new Error(`size mismatch for ${f.path}: got ${statSync(part).size}, expected ${f.size}`);
    if (f.sha256) {
      const digest = await sha256File(part);
      if (digest !== f.sha256) {
        await fs.rm(part, { force: true });
        throw new Error(`checksum mismatch for ${f.path}; the partial file was removed, retry the download`);
      }
    }
    await fs.rename(part, dest);
    base += f.size ?? received;
    onProgress({ modelId, file: f.path, received: base, total, done: false });
  }
  onProgress({ modelId, file: "", received: total, total, done: true });
}

/** Import a model from an offline folder (same layout as the Hugging Face repo). */
export async function importModel(entry: ModelManifestEntry, sourceDir: string, root: string): Promise<void> {
  for (const f of entry.files) {
    const src = path.join(sourceDir, f.path);
    if (!existsSync(src)) throw new Error(`missing ${f.path} in ${sourceDir}`);
    if (f.size !== null && statSync(src).size !== f.size) throw new Error(`size mismatch for ${f.path}`);
    if (f.sha256 && (await sha256File(src)) !== f.sha256) throw new Error(`checksum mismatch for ${f.path}`);
    const dest = path.join(root, entry.dir, f.path);
    await fs.mkdir(path.dirname(dest), { recursive: true });
    await fs.copyFile(src, dest);
  }
}
