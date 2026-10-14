import { spawnSync } from "node:child_process";
import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import path from "node:path";

export const check = (name, pass, detail = "") => ({ name, pass: !!pass, detail: String(detail).slice(0, 400) });

export function read(ws, rel) {
  const p = path.join(ws, rel);
  return existsSync(p) ? readFileSync(p, "utf8") : null;
}

export function run(ws, cmd, args, timeoutMs = 60000) {
  const r = spawnSync(cmd, args, { cwd: ws, encoding: "utf8", timeout: timeoutMs });
  return { code: r.status, out: `${r.stdout ?? ""}${r.stderr ?? ""}`.trim() };
}

/** All files under a folder (relative paths), skipping .git and node_modules. */
export function listFiles(ws, rel = ".") {
  const out = [];
  const walk = (d) => {
    for (const name of readdirSync(path.join(ws, d))) {
      if (name === ".git" || name === "node_modules") continue;
      const r = path.join(d, name);
      if (statSync(path.join(ws, r)).isDirectory()) walk(r);
      else out.push(path.normalize(r));
    }
  };
  if (existsSync(path.join(ws, rel))) walk(rel);
  return out;
}

export function filesFor(ws, prefix) {
  return listFiles(ws, prefix)
    .filter((f) => /\.(html|css|js|mjs|ts|json|md)$/.test(f))
    .map((f) => ({ path: f, content: readFileSync(path.join(ws, f), "utf8") }));
}

/** Checks common to static web apps: files exist, JS parses, HTML wires CSS/JS, page renders cleanly. */
export function staticAppChecks(ws, dir, render) {
  const html = read(ws, `${dir}/index.html`);
  const css = read(ws, `${dir}/styles.css`);
  const js = read(ws, `${dir}/app.js`);
  const syntax = js === null ? { code: 1, out: "missing" } : run(ws, process.execPath, ["--check", `${dir}/app.js`]);
  return [
    check("index.html exists", html !== null),
    check("styles.css exists", css !== null),
    check("app.js exists", js !== null),
    check("app.js parses", syntax.code === 0, syntax.out),
    check("index.html links styles.css", html && /href=["']\.?\/?styles\.css["']/.test(html)),
    check("index.html loads app.js", html && /src=["']\.?\/?app\.js["']/.test(html)),
    check("page renders", render?.loaded && !(render.errors ?? []).length, (render?.errors ?? []).join("; ")),
    check("no console errors", render?.loaded && !appConsole(render).some((c) => /error|exception/i.test(c)), appConsole(render ?? {}).join("; ")),
  ];
}

/** Console messages from the page itself (the headless renderer adds Electron's own warnings). */
const appConsole = (render) => (render.console ?? []).filter((c) => !/Electron Security Warning/.test(c));

/** True when every hex/rgb color in the CSS is a gray (r≈g≈b). */
export function isMonochrome(css) {
  const colors = [];
  for (const m of css.matchAll(/#([0-9a-f]{3}|[0-9a-f]{6})\b/gi)) {
    const h = m[1].length === 3 ? [...m[1]].map((c) => c + c).join("") : m[1];
    colors.push([0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16)));
  }
  for (const m of css.matchAll(/rgba?\(\s*(\d+)[ ,]+(\d+)[ ,]+(\d+)/gi)) colors.push([+m[1], +m[2], +m[3]]);
  const off = colors.filter(([r, g, b]) => Math.max(r, g, b) - Math.min(r, g, b) > 24);
  return { ok: off.length === 0, colors: colors.length, off: off.slice(0, 5) };
}
