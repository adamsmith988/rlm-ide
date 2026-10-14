#!/usr/bin/env node
// Downloads a relocatable CPython (python-build-standalone, install_only) into
// resources/python/<platform>-<arch>, which electron-builder bundles as the kernel runtime.
// Usage: node scripts/fetch-python.mjs [--platform linux|win32] [--arch x64|arm64] [--release 20250612] [--minor 3.12]
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, renameSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const arg = (name, fallback) => (process.argv.includes(`--${name}`) ? process.argv[process.argv.indexOf(`--${name}`) + 1] : fallback);
const platform = arg("platform", process.platform);
const arch = arg("arch", process.arch);
const minor = arg("minor", "3.12");
const release = arg("release", process.env.PBS_RELEASE ?? "latest");

const TRIPLES = {
  "linux-x64": "x86_64-unknown-linux-gnu",
  "linux-arm64": "aarch64-unknown-linux-gnu",
  "win32-x64": "x86_64-pc-windows-msvc",
  "win32-arm64": "aarch64-pc-windows-msvc",
};
const triple = TRIPLES[`${platform}-${arch}`];
if (!triple) throw new Error(`no python-build-standalone build for ${platform}-${arch}`);

const dest = path.join(root, "resources", "python", `${platform}-${arch}`);
const exe = platform === "win32" ? path.join(dest, "python.exe") : path.join(dest, "bin", "python3");
if (existsSync(exe)) {
  console.log(`python already present: ${exe}`);
  process.exit(0);
}

const api = `https://api.github.com/repos/astral-sh/python-build-standalone/releases/${release === "latest" ? "latest" : `tags/${release}`}`;
const headers = { Accept: "application/vnd.github+json", ...(process.env.GITHUB_TOKEN ? { Authorization: `Bearer ${process.env.GITHUB_TOKEN}` } : {}) };
const rel = await (await fetch(api, { headers })).json();
const pattern = new RegExp(`^cpython-${minor.replace(".", "\\.")}\\.\\d+\\+\\d+-${triple}-install_only\\.tar\\.gz$`);
const asset = (rel.assets ?? []).find((a) => pattern.test(a.name));
if (!asset) throw new Error(`no ${minor} install_only asset for ${triple} in release ${rel.tag_name ?? release}`);

console.log(`downloading ${asset.name}`);
const res = await fetch(asset.browser_download_url);
if (!res.ok) throw new Error(`download failed: ${res.status}`);
const tmp = path.join(os.tmpdir(), asset.name);
writeFileSync(tmp, Buffer.from(await res.arrayBuffer()));

const work = path.join(os.tmpdir(), `rlm-python-${Date.now()}`);
mkdirSync(work, { recursive: true });
execFileSync("tar", ["-xzf", tmp, "-C", work], { stdio: "inherit" });
rmSync(dest, { recursive: true, force: true });
mkdirSync(path.dirname(dest), { recursive: true });
renameSync(path.join(work, "python"), dest);
rmSync(work, { recursive: true, force: true });
rmSync(tmp, { force: true });
console.log(`python ${rel.tag_name}: ${exe}`);
