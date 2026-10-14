import { spawn } from "node:child_process";
import { writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { electronBinary } from "./app.mjs";

const here = path.dirname(fileURLToPath(import.meta.url));

/** Render a static page headlessly; returns console errors, DOM summary and the probe result. */
export function renderPage(htmlFile, screenshotFile, probeJs) {
  let probeFile = "-";
  if (probeJs) {
    probeFile = `${screenshotFile}.probe.js`;
    writeFileSync(probeFile, probeJs);
  }
  return new Promise((resolve) => {
    const args = [path.join(here, "../render/main.cjs"), htmlFile, screenshotFile, probeFile];
    if (process.platform === "linux") args.unshift("--no-sandbox");
    const p = spawn(electronBinary(), args, { stdio: ["ignore", "pipe", "pipe"] });
    let out = "";
    p.stdout.on("data", (d) => (out += d));
    const kill = setTimeout(() => p.kill("SIGKILL"), 40000);
    p.on("exit", () => {
      clearTimeout(kill);
      const line = out.split("\n").find((l) => l.startsWith("{"));
      try {
        resolve(JSON.parse(line));
      } catch {
        resolve({ loaded: false, errors: ["renderer produced no result"], console: [], probe: null });
      }
    });
  });
}
