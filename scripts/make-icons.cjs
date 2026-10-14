// Renders the RLM logo mark into resources/icons (PNG sizes for Linux, icon.ico for Windows).
// Run with Electron: npx electron scripts/make-icons.cjs   (add --no-sandbox on Ubuntu)
const { app, BrowserWindow } = require("electron");
const fs = require("node:fs");
const path = require("node:path");

const OUT = path.join(__dirname, "..", "resources", "icons");
const SIZES = [16, 24, 32, 48, 64, 128, 256, 512];
const ICO_SIZES = [16, 24, 32, 48, 64, 128, 256];

const html = `<!doctype html><html style="overflow:hidden;background:transparent"><body style="margin:0;overflow:hidden;background:transparent">
<svg width="512" height="512" viewBox="0 0 512 512" xmlns="http://www.w3.org/2000/svg">
  <rect x="16" y="16" width="480" height="480" rx="112" fill="#0b0b0c"/>
  <g transform="translate(64 64) scale(16)" fill="none" stroke="#f4f4f2" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">
    <circle cx="12" cy="12" r="9.5"/>
    <path d="m8.5 9 3 3-3 3M13 15h3"/>
  </g>
</svg></body></html>`;

/** ICO with PNG-compressed entries (supported since Windows Vista). */
function ico(images) {
  const header = Buffer.alloc(6);
  header.writeUInt16LE(0, 0);
  header.writeUInt16LE(1, 2);
  header.writeUInt16LE(images.length, 4);
  const dir = Buffer.alloc(16 * images.length);
  let offset = 6 + dir.length;
  images.forEach(({ size, png }, i) => {
    const o = i * 16;
    dir.writeUInt8(size >= 256 ? 0 : size, o);
    dir.writeUInt8(size >= 256 ? 0 : size, o + 1);
    dir.writeUInt8(0, o + 2);
    dir.writeUInt8(0, o + 3);
    dir.writeUInt16LE(1, o + 4);
    dir.writeUInt16LE(32, o + 6);
    dir.writeUInt32LE(png.length, o + 8);
    dir.writeUInt32LE(offset, o + 12);
    offset += png.length;
  });
  return Buffer.concat([header, dir, ...images.map((x) => x.png)]);
}

app.disableHardwareAcceleration();

app.whenReady().then(async () => {
  const win = new BrowserWindow({ width: 512, height: 512, show: false, frame: false, transparent: true, backgroundColor: "#00000000", useContentSize: true });
  await win.loadURL(`data:text/html;charset=utf-8,${encodeURIComponent(html)}`);
  await new Promise((r) => setTimeout(r, 300));
  const full = (await win.webContents.capturePage()).resize({ width: 512, height: 512, quality: "best" });
  fs.mkdirSync(OUT, { recursive: true });
  const pngs = {};
  for (const size of SIZES) {
    pngs[size] = (size === 512 ? full : full.resize({ width: size, height: size, quality: "best" })).toPNG();
    fs.writeFileSync(path.join(OUT, `${size}x${size}.png`), pngs[size]);
  }
  fs.writeFileSync(path.join(OUT, "icon.png"), pngs[512]);
  fs.writeFileSync(path.join(OUT, "icon.ico"), ico(ICO_SIZES.map((size) => ({ size, png: pngs[size] }))));
  console.log(`icons written to ${OUT}`);
  app.quit();
});
