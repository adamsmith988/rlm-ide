import { createHash } from "node:crypto";
import { createReadStream, createWriteStream, existsSync, promises as fs, statSync } from "node:fs";
import path from "node:path";
import type { HubJobView } from "@rlm/contracts";
import type { HfClient, RepoFile } from "./hf";

export interface JobSpec {
  id: string;
  repo: string;
  revision: string;
  sha: string;
  dest: string;
  files: RepoFile[];
}

interface Running {
  spec: JobSpec;
  view: HubJobView;
  ac: AbortController;
  /** Bytes on disk per file (finished files count fully). */
  have: Map<string, number>;
  samples: Array<{ t: number; bytes: number }>;
}

const SPEED_WINDOW_MS = 5000;

export function sha256File(p: string): Promise<string> {
  const h = createHash("sha256");
  return new Promise((resolve, reject) => {
    createReadStream(p)
      .on("data", (d) => h.update(d))
      .on("end", () => resolve(h.digest("hex")))
      .on("error", reject);
  });
}

/** A path inside the repo must stay inside the destination folder. */
export function safeJoin(root: string, rel: string): string {
  const p = path.resolve(root, rel);
  if (!p.startsWith(path.resolve(root) + path.sep)) throw new Error(`refusing file outside the model folder: ${rel}`);
  return p;
}

export class DownloadManager {
  private readonly running = new Map<string, Running>();

  constructor(
    private readonly hf: HfClient,
    private readonly opts: { concurrency: () => number; onChange: (job: HubJobView) => void; onFinished: (job: HubJobView, spec: JobSpec) => void },
  ) {}

  isRunning(id: string): boolean {
    return this.running.has(id);
  }

  /** Starts (or resumes) a job; files already complete on disk are skipped. */
  start(spec: JobSpec, previous?: HubJobView): HubJobView {
    if (this.running.has(spec.id)) return this.running.get(spec.id)!.view;
    const view: HubJobView = previous
      ? { ...previous, status: "running", error: null, speedBps: 0, etaSec: null, current: [], finishedAt: null }
      : {
          id: spec.id,
          repo: spec.repo,
          revision: spec.revision,
          sha: spec.sha,
          dest: spec.dest,
          status: "running",
          files: spec.files.length,
          filesDone: 0,
          bytesDone: 0,
          bytesTotal: spec.files.reduce((n, f) => n + f.size, 0),
          speedBps: 0,
          etaSec: null,
          current: [],
          error: null,
          startedAt: Date.now(),
          finishedAt: null,
        };
    const r: Running = { spec, view, ac: new AbortController(), have: new Map(), samples: [] };
    this.running.set(spec.id, r);
    void this.run(r);
    return view;
  }

  pause(id: string): boolean {
    const r = this.running.get(id);
    if (!r) return false;
    r.view.status = "paused";
    r.ac.abort();
    return true;
  }

  cancel(id: string): boolean {
    const r = this.running.get(id);
    if (!r) return false;
    r.view.status = "cancelled";
    r.ac.abort();
    return true;
  }

  stopAll(): void {
    for (const r of this.running.values()) {
      r.view.status = "paused";
      r.ac.abort();
    }
  }

  private progress(r: Running, force = false): void {
    const now = Date.now();
    const bytes = [...r.have.values()].reduce((a, b) => a + b, 0);
    r.view.bytesDone = Math.min(bytes, r.view.bytesTotal || bytes);
    r.samples.push({ t: now, bytes });
    while (r.samples.length > 2 && now - r.samples[0]!.t > SPEED_WINDOW_MS) r.samples.shift();
    const first = r.samples[0]!;
    const dt = (now - first.t) / 1000;
    r.view.speedBps = dt > 0.5 ? Math.max(0, (bytes - first.bytes) / dt) : r.view.speedBps;
    r.view.etaSec = r.view.speedBps > 0 ? Math.round((r.view.bytesTotal - r.view.bytesDone) / r.view.speedBps) : null;
    const last = (r as Running & { lastEmit?: number }).lastEmit ?? 0;
    if (force || now - last > 300) {
      (r as Running & { lastEmit?: number }).lastEmit = now;
      this.opts.onChange({ ...r.view, current: [...r.view.current] });
    }
  }

  private async run(r: Running): Promise<void> {
    const queue = [...r.spec.files];
    const worker = async () => {
      for (let f = queue.shift(); f; f = queue.shift()) {
        if (r.ac.signal.aborted) return;
        const dest = safeJoin(r.spec.dest, f.path);
        if (existsSync(dest) && statSync(dest).size === f.size) continue;
        r.view.current.push(f.path);
        try {
          await this.file(r, f, dest);
          r.view.filesDone++;
        } finally {
          r.view.current = r.view.current.filter((c) => c !== f!.path);
        }
        this.progress(r, true);
      }
    };
    try {
      r.view.filesDone = 0;
      for (const f of r.spec.files) {
        const dest = safeJoin(r.spec.dest, f.path);
        if (existsSync(dest) && statSync(dest).size === f.size) {
          r.have.set(f.path, f.size);
          r.view.filesDone++;
        } else if (existsSync(`${dest}.part`)) r.have.set(f.path, statSync(`${dest}.part`).size);
      }
      this.progress(r, true);
      await Promise.all(Array.from({ length: Math.max(1, Math.min(this.opts.concurrency(), queue.length || 1)) }, worker));
      if (r.ac.signal.aborted) throw new DOMException("aborted", "AbortError");
      r.view.status = "done";
      r.view.speedBps = 0;
      r.view.etaSec = 0;
    } catch (e) {
      if (r.ac.signal.aborted) {
        if (r.view.status !== "paused" && r.view.status !== "cancelled") r.view.status = "paused";
      } else {
        r.view.status = "error";
        r.view.error = e instanceof Error ? e.message : String(e);
        r.ac.abort();
      }
      r.view.speedBps = 0;
      r.view.etaSec = null;
    }
    r.view.finishedAt = Date.now();
    r.view.current = [];
    this.running.delete(r.spec.id);
    this.progress(r, true);
    this.opts.onFinished({ ...r.view }, r.spec);
  }

  private async file(r: Running, f: RepoFile, dest: string): Promise<void> {
    await fs.mkdir(path.dirname(dest), { recursive: true });
    const part = `${dest}.part`;
    let have = existsSync(part) ? statSync(part).size : 0;
    if (have > f.size) {
      await fs.rm(part, { force: true });
      have = 0;
    }
    if (have < f.size || f.size === 0) {
      const res = await fetch(this.hf.resolveUrl(r.spec.repo, r.spec.sha, f.path), {
        headers: this.hf.headers(have ? { Range: `bytes=${have}-` } : {}),
        signal: r.ac.signal,
        redirect: "follow",
      });
      if (res.status === 401 || res.status === 403)
        throw new Error(`${f.path}: HTTP ${res.status}. ${this.hf.hasToken() ? "The token has no access; for gated repos accept the license on huggingface.co first." : "This repo needs a Hugging Face token (Models hub → Tools)."}`);
      if (!res.ok && res.status !== 206) throw new Error(`${f.path}: HTTP ${res.status}`);
      const append = res.status === 206 && have > 0;
      if (!append) have = 0;
      const out = createWriteStream(part, { flags: append ? "a" : "w" });
      const reader = res.body!.getReader();
      try {
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          if (!out.write(value)) await new Promise<void>((ok) => out.once("drain", () => ok()));
          have += value.byteLength;
          r.have.set(f.path, have);
          this.progress(r);
        }
      } finally {
        await new Promise<void>((ok) => out.end(() => ok()));
      }
    }
    const size = statSync(part).size;
    if (size !== f.size) throw new Error(`${f.path}: got ${size} bytes, expected ${f.size}`);
    if (f.sha256) {
      const digest = await sha256File(part);
      if (digest !== f.sha256) {
        await fs.rm(part, { force: true });
        r.have.set(f.path, 0);
        throw new Error(`${f.path}: checksum mismatch; the partial file was removed, resume to fetch it again`);
      }
    }
    await fs.rename(part, dest);
    r.have.set(f.path, f.size);
  }
}
