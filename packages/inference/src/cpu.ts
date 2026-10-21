import { execFile } from "node:child_process";
import { readFile } from "node:fs/promises";
import os from "node:os";

/** Physical cores in /proc/cpuinfo: distinct (physical id, core id) pairs. */
export function coresFromCpuinfo(text: string): number | null {
  const seen = new Set<string>();
  for (const block of text.split(/\n\s*\n/)) {
    const phys = /^physical id\s*:\s*(\d+)/m.exec(block)?.[1];
    const core = /^core id\s*:\s*(\d+)/m.exec(block)?.[1];
    if (core !== undefined) seen.add(`${phys ?? 0}:${core}`);
  }
  return seen.size || null;
}

function cmd(file: string, args: string[]): Promise<string> {
  return new Promise((resolve) => execFile(file, args, { timeout: 10_000, windowsHide: true }, (err, out) => resolve(err ? "" : String(out))));
}

/** Physical CPU cores; falls back to half the logical CPUs (two hardware threads per core is the common case). */
export async function physicalCores(platform: NodeJS.Platform = process.platform): Promise<number> {
  const logical = os.availableParallelism();
  let n: number | null = null;
  if (platform === "linux") n = coresFromCpuinfo(await readFile("/proc/cpuinfo", "utf8").catch(() => ""));
  else if (platform === "darwin") n = Number((await cmd("sysctl", ["-n", "hw.physicalcpu"])).trim()) || null;
  else if (platform === "win32") {
    const out = await cmd("powershell.exe", ["-NoProfile", "-NonInteractive", "-Command", "(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum"]);
    n = Number(out.trim()) || null;
  }
  return Math.max(1, Math.min(logical, n ?? Math.ceil(logical / 2)));
}

let detected: Promise<number> | null = null;

/** Threads for onnxruntime on the CPU: the setting, or one per physical core. */
export async function cpuThreads(configured: number): Promise<number> {
  if (configured > 0) return configured;
  detected ??= physicalCores();
  return detected;
}
