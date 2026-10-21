import { cpSync, createWriteStream, existsSync, mkdirSync, readdirSync, renameSync, rmSync, statSync } from "node:fs";
import path from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { run, lastLines } from "./proc";

/** Prebuilt llama.cpp from its GitHub releases: no git, compiler or CMake needed (most Windows machines have none).
 * Every build is published as a pre-release named bNNNN with archives per system and GPU runtime, for example
 * llama-b11796-bin-win-cuda-12.4-x64.zip plus cudart-llama-bin-win-cuda-12.4-x64.zip for the CUDA runtime. */

export interface ReleaseAsset {
  name: string;
  url: string;
  size: number;
}
export interface LlamaRelease {
  tag: string;
  assets: ReleaseAsset[];
}
export type LlamaFlavor = "cpu" | "cuda" | "vulkan" | "metal";
export interface LlamaPick {
  tag: string;
  flavor: LlamaFlavor;
  main: ReleaseAsset;
  /** CUDA runtime libraries the CUDA build needs when the system has none. */
  runtime: ReleaseAsset | null;
  cudaVersion: string | null;
}
export type LlamaWant = "auto" | "cpu" | "gpu";

const OS: Partial<Record<NodeJS.Platform, string>> = { win32: "win", linux: "ubuntu", darwin: "macos" };
const ARCH: Record<string, string> = { x64: "x64", arm64: "arm64" };

/** "12.4" < "12.10" < "13.0" */
export function compareVersions(a: string, b: string): number {
  const pa = a.split(".").map(Number);
  const pb = b.split(".").map(Number);
  for (let i = 0; i < Math.max(pa.length, pb.length); i++) {
    const d = (pa[i] ?? 0) - (pb[i] ?? 0);
    if (d) return d;
  }
  return 0;
}

interface Parsed {
  asset: ReleaseAsset;
  flavor: LlamaFlavor;
  cuda: string | null;
}

function parse(a: ReleaseAsset, os: string, arch: string): Parsed | null {
  const m = new RegExp(`^llama-b\\d+-bin-${os}-(?:(.+)-)?${arch}\\.(?:zip|tar\\.gz|tgz)$`).exec(a.name);
  if (!m) return null;
  const variant = m[1] ?? "";
  if (variant === "" || variant === "cpu") return { asset: a, flavor: os === "macos" ? "metal" : "cpu", cuda: null };
  const cuda = /^cuda-([\d.]+)$/.exec(variant);
  if (cuda) return { asset: a, flavor: "cuda", cuda: cuda[1]! };
  if (variant === "vulkan") return { asset: a, flavor: "vulkan", cuda: null };
  return null;
}

/** The archive for this computer. `cudaMax`: highest CUDA version the NVIDIA driver supports (null: no NVIDIA GPU). */
export function pickLlamaAssets(rel: LlamaRelease, o: { platform: NodeJS.Platform; arch: string; want: LlamaWant; cudaMax: string | null }): LlamaPick | null {
  const os = OS[o.platform];
  const arch = ARCH[o.arch];
  if (!os || !arch) return null;
  const all = rel.assets.map((a) => parse(a, os, arch)).filter((x): x is Parsed => !!x);
  const of = (f: LlamaFlavor) => all.filter((x) => x.flavor === f);
  const done = (p: Parsed, runtime: ReleaseAsset | null = null): LlamaPick => ({ tag: rel.tag, flavor: p.flavor, main: p.asset, runtime, cudaVersion: p.cuda });
  if (os === "macos") return of("metal")[0] ? done(of("metal")[0]!) : null;
  if (o.want === "cpu") return of("cpu")[0] ? done(of("cpu")[0]!) : null;
  if (o.cudaMax) {
    const fits = of("cuda")
      .filter((x) => compareVersions(x.cuda!, o.cudaMax!) <= 0)
      .sort((a, b) => compareVersions(b.cuda!, a.cuda!));
    const best = fits[0];
    if (best) {
      const runtime = rel.assets.find((a) => new RegExp(`^cudart-llama(?:-b\\d+)?-bin-${os}-cuda-${best.cuda!.replace(/\./g, "\\.")}-${arch}\\.(?:zip|tar\\.gz|tgz)$`).test(a.name)) ?? null;
      return done(best, runtime);
    }
  }
  // Vulkan reaches AMD, Intel and NVIDIA GPUs and falls back to the CPU on its own.
  if (of("vulkan")[0]) return done(of("vulkan")[0]!);
  return of("cpu")[0] ? done(of("cpu")[0]!) : null;
}

/** Highest CUDA version the installed NVIDIA driver supports ("CUDA Version: 12.8" in nvidia-smi), or null. */
export async function driverCudaVersion(): Promise<string | null> {
  const r = await run("nvidia-smi", []);
  return r.code === 0 ? (/CUDA Version:\s*([\d.]+)/.exec(r.tail)?.[1] ?? null) : null;
}

export async function listReleases(repo: string, tag: string, fetchImpl: typeof fetch = fetch): Promise<LlamaRelease[]> {
  const url = tag ? `https://api.github.com/repos/${repo}/releases/tags/${encodeURIComponent(tag)}` : `https://api.github.com/repos/${repo}/releases?per_page=20`;
  const res = await fetchImpl(url, { headers: { Accept: "application/vnd.github+json", "User-Agent": "rlm-ide" }, signal: AbortSignal.timeout(30_000) });
  if (!res.ok) throw new Error(`GitHub releases of ${repo}: HTTP ${res.status} ${(await res.text().catch(() => "")).slice(0, 200)}`);
  const j = (await res.json()) as unknown;
  const list = (Array.isArray(j) ? j : [j]) as Array<{ tag_name: string; draft?: boolean; assets?: Array<{ name: string; browser_download_url: string; size: number }> }>;
  return list.filter((r) => !r.draft).map((r) => ({ tag: r.tag_name, assets: (r.assets ?? []).map((a) => ({ name: a.name, url: a.browser_download_url, size: a.size })) }));
}

export async function download(url: string, file: string, log: (l: string) => void, fetchImpl: typeof fetch = fetch): Promise<void> {
  mkdirSync(path.dirname(file), { recursive: true });
  const res = await fetchImpl(url, { headers: { "User-Agent": "rlm-ide" }, redirect: "follow" });
  if (!res.ok || !res.body) throw new Error(`download ${url}: HTTP ${res.status}`);
  const total = Number(res.headers.get("content-length") ?? 0);
  let got = 0;
  let shown = 0;
  const body = Readable.fromWeb(res.body as never);
  body.on("data", (c: Buffer) => {
    got += c.length;
    if (total) {
      const pct = Math.floor((got / total) * 100);
      if (pct >= shown + 10) {
        shown = pct - (pct % 10);
        log(`  ${path.basename(file)}: ${pct}% of ${Math.round(total / 2 ** 20)} MB`);
      }
    } else {
      const mb = Math.floor(got / 2 ** 20);
      if (mb >= shown + 5) {
        shown = mb - (mb % 5);
        log(`  ${path.basename(file)}: ${mb} MB`);
      }
    }
  });
  await pipeline(body, createWriteStream(file));
}

/** Unpacks .zip and .tar.gz with the system's own tools. Windows' tar.exe (bsdtar, in System32 since Windows 10)
 * reads zip; a GNU tar from Git for Windows earlier on PATH does not, so System32's is named explicitly. */
export async function extract(archive: string, dest: string): Promise<void> {
  mkdirSync(dest, { recursive: true });
  const zip = /\.zip$/i.test(archive);
  let r;
  if (process.platform === "win32") {
    const tar = path.join(process.env.SystemRoot ?? "C:\\Windows", "System32", "tar.exe");
    r = await run(existsSync(tar) ? tar : "tar", ["-xf", archive, "-C", dest]);
    if (r.code !== 0 && zip) r = await run("powershell.exe", ["-NoProfile", "-NonInteractive", "-Command", `Expand-Archive -LiteralPath '${archive.replace(/'/g, "''")}' -DestinationPath '${dest.replace(/'/g, "''")}' -Force`]);
  } else if (zip && process.platform === "linux") {
    r = await run("unzip", ["-q", "-o", archive, "-d", dest]);
    if (r.code !== 0) r = await run("python3", ["-m", "zipfile", "-e", archive, dest]);
  } else {
    r = await run("tar", ["-xf", archive, "-C", dest]);
  }
  if (r.code !== 0) throw new Error(`could not unpack ${path.basename(archive)}: ${lastLines(r.tail, 4)}`);
}

/** Copies keeping relative symlinks as they are: Linux and macOS archives link libllama.so.0 -> libllama.so.0.7.0, and
 * cpSync's default rewrites such links to absolute paths into the folder being copied from. */
export function copyTree(from: string, to: string): void {
  cpSync(from, to, { recursive: true, verbatimSymlinks: true });
}

/** Moves an unpacked folder into place; copies when a rename is refused (another drive, or Windows antivirus holding a file). */
export function moveTree(from: string, to: string): void {
  rmSync(to, { recursive: true, force: true });
  mkdirSync(path.dirname(to), { recursive: true });
  try {
    renameSync(from, to);
  } catch {
    copyTree(from, to);
  }
}

/** The folder in an unpacked archive that holds `file` (archives nest their binaries differently: build/bin, a
 * versioned folder, or the top level). */
export function findDirWith(root: string, file: string, depth = 5): string | null {
  if (existsSync(path.join(root, file))) return root;
  if (depth <= 0) return null;
  for (const e of readdirSync(root)) {
    const p = path.join(root, e);
    if (statSync(p).isDirectory()) {
      const hit = findDirWith(p, file, depth - 1);
      if (hit) return hit;
    }
  }
  return null;
}
