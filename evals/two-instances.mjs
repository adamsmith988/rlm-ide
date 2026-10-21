// Runs two copies of the IDE on one profile, each with its own project, and checks they work side by side.
import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, realpathSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { IdeApp } from "./lib/app.mjs";
import { sleep } from "./lib/cdp.mjs";

const out = process.argv[2] ?? mkdtempSync(path.join(tmpdir(), "rlm-two-"));
mkdirSync(out, { recursive: true });
const userData = path.join(out, "profile");
const projects = ["alpha", "beta"].map((n) => {
  const dir = path.join(out, n);
  mkdirSync(dir, { recursive: true });
  writeFileSync(path.join(dir, `${n}.txt`), `${n}\n`);
  return dir;
});

const results = [];
const check = (name, ok, detail = "") => {
  results.push({ name, ok, detail });
  console.log(`${ok ? "PASS" : "FAIL"} ${name}${detail ? ` — ${detail}` : ""}`);
};

const apps = projects.map((workspace, i) => new IdeApp({ userData, workspace, logFile: path.join(out, `app${i}.log`), settings: { autoUpdate: false } }));
try {
  await apps[0].start();
  check("first instance opens its project", true);
  await apps[1].start();
  check("second instance opens on the same profile", true);
  for (const [i, app] of apps.entries()) {
    const info = await app.cdp.invoke("app:info");
    const tree = await app.cdp.invoke("fs:readDir", { path: "" });
    check(`instance ${i + 1} shows its own folder`, tree.some((e) => e.name === `${path.basename(projects[i])}.txt`), tree.map((e) => e.name).join(", "));
    const chat = await app.cdp.invoke("chat:create", { title: `chat ${i}` });
    check(`instance ${i + 1} creates a chat in its project`, chat.workspaceId === projects[i], chat.workspaceId);
    void info;
  }
  const [l0, l1] = await Promise.all(apps.map((a) => a.cdp.invoke("chat:list")));
  check("each instance lists only its own project's chats", l0.length === 1 && l1.length === 1 && l0[0].id !== l1[0].id, `${l0.map((c) => c.title)} | ${l1.map((c) => c.title)}`);
  await apps[0].cdp.invoke("settings:update", { patch: { fontSize: 15 } });
  await apps[1].cdp.invoke("settings:update", { patch: { theme: "light" } });
  await sleep(2500);
  const [s0, s1] = await Promise.all(apps.map((a) => a.cdp.invoke("settings:get")));
  check("a setting changed in one instance is not lost by the other", s0.fontSize === 15 && s1.fontSize === 15 && s0.theme === "light" && s1.theme === "light", `fontSize ${s0.fontSize}/${s1.fontSize}, theme ${s0.theme}/${s1.theme}`);
  const opened = await apps[1].cdp.invoke("workspace:open", { root: projects[0] }).then(
    () => "opened",
    (e) => String(e),
  );
  check("a folder open in one instance is not opened again in another", /already open/i.test(opened), opened);
  const buttons = await apps[1].cdp.eval(`["open-folder","new-window"].filter((id) => document.querySelector('[data-testid="' + id + '"]')).join(",")`);
  check("the title bar offers Open folder and New window", buttons === "open-folder,new-window", buttons);

  // "New window" starts a third copy on its own folder; it shows up as that folder's lock owner.
  const gamma = path.join(out, "gamma");
  mkdirSync(gamma, { recursive: true });
  await apps[1].cdp.invoke("app:newWindow", { root: gamma });
  const locksDir = path.join(userData, "locks");
  let third = null;
  for (let i = 0; i < 60 && !third; i++) {
    await sleep(1000);
    for (const f of existsSync(locksDir) ? readdirSync(locksDir) : []) {
      try {
        const l = JSON.parse(readFileSync(path.join(locksDir, f), "utf8"));
        if (l.root === realpathSync(gamma)) third = l;
      } catch {
        /* being written */
      }
    }
  }
  check("New window opens another copy of the IDE on the chosen folder", !!third, third ? `pid ${third.pid}` : "no lock for gamma after 60 s");
  if (third) {
    const viaOther = await apps[0].cdp.invoke("workspace:open", { root: gamma }).then(() => "opened", (e) => String(e));
    check("the new window's folder is not opened again elsewhere", /already open/i.test(viaOther), viaOther);
    process.kill(third.pid, "SIGTERM");
    for (let i = 0; i < 20; i++) {
      await sleep(500);
      try {
        process.kill(third.pid, 0);
      } catch {
        break;
      }
    }
    const freed = await apps[0].cdp.invoke("workspace:open", { root: gamma }).then(() => "opened", (e) => String(e));
    check("a closed window frees its folder", freed === "opened", freed);
  }
  for (const [i, app] of apps.entries()) {
    const errs = app.cdp.console.filter((m) => /error/i.test(m.type ?? "") || /Uncaught/.test(m.text ?? ""));
    check(`instance ${i + 1} has no renderer errors`, errs.length === 0, errs.slice(0, 3).map((m) => m.text).join(" | "));
  }
} catch (e) {
  check("run", false, String(e?.stack ?? e));
} finally {
  for (const a of apps) await a.stop().catch(() => undefined);
}
writeFileSync(path.join(out, "results.json"), JSON.stringify(results, null, 2));
const failed = results.filter((r) => !r.ok).length;
console.log(`${results.length - failed}/${results.length} passed; logs in ${out}`);
process.exit(failed ? 1 : 0);
