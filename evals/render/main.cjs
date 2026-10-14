// Headless page renderer for evals: node evals/lib/render.mjs spawns this with Electron.
// argv: <html file> <screenshot png> <probe js file or "-">. Prints one JSON line to stdout.
const { app, BrowserWindow } = require("electron");
const fs = require("node:fs");

const [file, shot, probeFile] = process.argv.slice(-3);
app.disableHardwareAcceleration();

app.whenReady().then(async () => {
  const out = { loaded: false, console: [], errors: [], probe: null, title: "", text: "", counts: {} };
  const win = new BrowserWindow({ show: false, width: 1280, height: 860, webPreferences: { sandbox: true, contextIsolation: true } });
  win.webContents.on("console-message", (e) => {
    const level = e.level ?? "";
    if (level === "error" || level === "warning" || level === 3 || level === 2) out.console.push(`${level}: ${e.message}`);
  });
  win.webContents.on("did-fail-load", (_e, code, desc, url) => out.errors.push(`load failed ${code} ${desc} ${url}`));
  const timer = setTimeout(() => finish(), 20000);
  async function finish() {
    clearTimeout(timer);
    try {
      const img = await win.webContents.capturePage();
      fs.writeFileSync(shot, img.toPNG());
    } catch (e) {
      out.errors.push(`screenshot: ${e}`);
    }
    process.stdout.write(JSON.stringify(out) + "\n");
    app.exit(0);
  }
  try {
    await win.loadFile(file);
    out.loaded = true;
    await new Promise((r) => setTimeout(r, 1200));
    const summary = await win.webContents.executeJavaScript(`(() => {
      const vis = (el) => { const r = el.getBoundingClientRect(); const s = getComputedStyle(el); return r.width > 0 && r.height > 0 && s.visibility !== "hidden" && s.display !== "none"; };
      return {
        title: document.title,
        text: document.body ? document.body.innerText.slice(0, 3000) : "",
        counts: {
          buttons: [...document.querySelectorAll("button, [role=button]")].filter(vis).length,
          images: document.querySelectorAll("img, svg").length,
          inputs: document.querySelectorAll("input, textarea, select").length,
          links: document.querySelectorAll("a").length,
          elements: document.querySelectorAll("*").length,
        },
        stylesheets: document.styleSheets.length,
        scripts: document.scripts.length,
        bodyBg: document.body ? getComputedStyle(document.body).backgroundColor : "",
        fonts: [...new Set([...document.querySelectorAll("body *")].slice(0, 400).map((e) => getComputedStyle(e).fontFamily.split(",")[0].trim()))].slice(0, 8),
      };
    })()`);
    Object.assign(out, summary);
    if (probeFile && probeFile !== "-") {
      const code = fs.readFileSync(probeFile, "utf8");
      out.probe = await win.webContents.executeJavaScript(`(async () => { ${code} })()`).catch((e) => ({ error: String(e) }));
      await new Promise((r) => setTimeout(r, 300));
    }
  } catch (e) {
    out.errors.push(String(e));
  }
  await finish();
});
