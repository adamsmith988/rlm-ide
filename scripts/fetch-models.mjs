#!/usr/bin/env node
// Downloads the built-in models into resources/models (bundled at build time) or a given dir.
// Usage: node scripts/fetch-models.mjs [--only builtin:bge-m3] [--dest resources/models]
import path from "node:path";
import { fileURLToPath } from "node:url";
import { downloadModel, loadManifest, isInstalled } from "../packages/inference/src/model_manager.ts";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2);
const only = args.includes("--only") ? args[args.indexOf("--only") + 1] : null;
const dest = path.resolve(root, args.includes("--dest") ? args[args.indexOf("--dest") + 1] : "resources/models");

const manifest = await loadManifest(path.join(root, "resources/models.manifest.json"));
for (const [id, entry] of Object.entries(manifest)) {
  if (only && id !== only) continue;
  if (isInstalled(dest, entry)) {
    console.log(`${id}: already installed in ${path.join(dest, entry.dir)}`);
    continue;
  }
  console.log(`${id}: downloading ${entry.repo}@${entry.revision.slice(0, 8)} -> ${path.join(dest, entry.dir)}`);
  let last = 0;
  await downloadModel(id, entry, dest, (p) => {
    const pct = p.total ? Math.floor((p.received / p.total) * 100) : 0;
    if (pct !== last || p.done) {
      last = pct;
      process.stdout.write(`\r${id}: ${pct}% ${p.file}            `);
    }
  });
  console.log(`\n${id}: done (sha256 verified)`);
}
