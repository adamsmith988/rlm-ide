// End-to-end check of the team hub with two real IDE instances.
// Alice hosts a hub and shares her folder; Bob joins over SSH with an invite code, clones,
// pushes a branch; Alice reviews and merges it in the UI, then reports a bug that is routed to Bob,
// who answers with a fix status. Run after `pnpm build`.
import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { IdeApp, REPO } from "./lib/app.mjs";
import { sleep } from "./lib/cdp.mjs";

const out = path.join(REPO, "evals/results/team");
mkdirSync(out, { recursive: true });
const tmp = mkdtempSync(path.join(os.tmpdir(), "rlm-team-e2e-"));
const results = [];
const check = (name, ok, detail = "") => {
  results.push({ name, ok: !!ok, detail });
  console.log(`${ok ? "PASS" : "FAIL"} ${name}${detail ? ` - ${detail}` : ""}`);
};

const httpPort = 20000 + Math.floor(Math.random() * 20000);
const sshPort = httpPort + 1;

// Alice's project: a small library with tests.
const aliceWs = path.join(tmp, "cart");
mkdirSync(aliceWs);
writeFileSync(path.join(aliceWs, "cart.js"), ["function total(items) {", "  return items.reduce((sum, i) => sum + i.price * i.qty, 0);", "}", "", "module.exports = { total };", ""].join("\n"));
writeFileSync(path.join(aliceWs, "README.md"), "# Cart\n\nTotals for the checkout page.\n");
const bobHome = path.join(tmp, "bob-home");
mkdirSync(bobHome);

const tid = (id) => `[data-testid="${id}"]`;
async function ui(app, js) {
  return app.cdp.eval(`(async () => { ${js} })()`);
}
async function click(app, testid, timeoutMs = 10000) {
  const until = Date.now() + timeoutMs;
  for (;;) {
    const ok = await app.cdp.eval(`(() => { const el = document.querySelector('${tid(testid)}'); if (!el || el.disabled) return false; el.click(); return true; })()`);
    if (ok) return;
    if (Date.now() > until) throw new Error(`could not click ${testid}`);
    await sleep(200);
  }
}
async function type(app, testid, text) {
  await app.cdp.eval(`(() => { const el = document.querySelector('${tid(testid)}'); el.focus(); el.select?.(); })()`);
  await app.cdp.send("Input.insertText", { text });
}
async function waitFor(app, js, timeoutMs = 15000) {
  const until = Date.now() + timeoutMs;
  for (;;) {
    const v = await app.cdp.eval(js).catch(() => null);
    if (v) return v;
    if (Date.now() > until) return null;
    await sleep(250);
  }
}
const text = (app, testid) => app.cdp.eval(`document.querySelector('${tid(testid)}')?.innerText ?? ""`);
async function pickSelect(app, testid, matcher) {
  return app.cdp.eval(`(() => {
    const el = document.querySelector('${tid(testid)}');
    const opt = [...el.options].find((o) => o.value.includes(${JSON.stringify(matcher)}) || o.text.includes(${JSON.stringify(matcher)}));
    if (!opt) return false;
    Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, "value").set.call(el, opt.value);
    el.dispatchEvent(new Event("change", { bubbles: true }));
    return true;
  })()`);
}

const alice = new IdeApp({ userData: path.join(tmp, "alice-data"), workspace: aliceWs, logFile: path.join(out, "alice.log") });
let bob = new IdeApp({ userData: path.join(tmp, "bob-data"), workspace: bobHome, logFile: path.join(out, "bob.log") });

try {
  await alice.start();
  await bob.start();

  // ---- Alice hosts -------------------------------------------------------------------------------
  await click(alice, "activity-team");
  await type(alice, "team-name", "Alice");
  await type(alice, "team-email", "alice@team.dev");
  await click(alice, "team-save-profile");
  await sleep(300);
  await alice.cdp.invoke("team:host", { enabled: true, name: "Alice's hub", lan: true, httpPort, sshPort, discovery: true });
  const hosting = await waitFor(alice, `document.querySelector('${tid("team-host-info")}')?.innerText`);
  check("host: hub runs and shows how to connect", hosting?.includes(`:${httpPort}`) && hosting?.includes(`${sshPort}`), hosting?.split("\n").slice(0, 3).join(" | "));
  await alice.screenshot(path.join(out, "1-alice-hosting.png"));

  await type(alice, "team-group-name", "Checkout");
  await click(alice, "team-create-group");
  await click(alice, "group-tab-invites");
  await click(alice, "group-create-invite");
  const code = await waitFor(alice, `document.querySelector('${tid("group-invite-code")}')?.innerText`);
  check("host: invite code created in the UI", /^[A-Z2-9]{4}-[A-Z2-9]{4}-[A-Z2-9]{4}$/.test(code ?? ""), code);
  await click(alice, "group-tab-projects");
  await click(alice, "group-share-workspace", 15000);
  const project = await waitFor(alice, `document.querySelector('${tid("team-project")}')?.innerText`, 30000);
  check("host: open folder shared as a project and pushed", project?.includes("cart") && !project?.includes("This project is empty"));
  const aliceRemote = execFileSync("git", ["remote", "-v"], { cwd: aliceWs, encoding: "utf8" });
  check("host: workspace got a hub remote", aliceRemote.includes(`/git/checkout/cart.git`), aliceRemote.split("\n")[0]);

  // ---- Bob joins over SSH ------------------------------------------------------------------------
  await bob.cdp.invoke("team:profile", { name: "Bob", email: "bob@team.dev" });
  const seen = await waitFor(bob, `window.rlm.invoke("team:state", {}).then((s) => s.discovered.find((d) => d.name === "Alice's hub"))`, 15000);
  check("join: Bob discovers Alice's hub on the LAN", !!seen, seen ? `${seen.address}:${seen.httpPort}` : "not seen");
  // Typed the way a person would after hearing it: lower case, spaces.
  const spoken = code.toLowerCase().replace(/-/g, " ");
  const st = await bob.cdp.invoke("team:connect", { transport: { kind: "ssh", host: "127.0.0.1", port: sshPort }, code: spoken });
  const conn = st.connections[0];
  check("join: Bob joined over SSH with the invite code", conn?.hubName === "Alice's hub" && conn.transport.hostFingerprint?.startsWith("SHA256:"), conn?.transport.hostFingerprint);
  const online = await waitFor(bob, `window.rlm.invoke("team:state", {}).then((s) => s.connections[0]?.status === "online")`, 15000);
  check("join: Bob's tunnel is online", online);
  const dest = await bob.cdp.invoke("team:clone", { connId: conn.id, group: "checkout", project: "cart", parent: bobHome });
  const cloned = readFileSync(path.join(dest, "cart.js"), "utf8");
  check("join: Bob cloned the project over git+ssh", cloned.includes("i.price * i.qty"));
  const ident = execFileSync("git", ["config", "--local", "user.email"], { cwd: dest, encoding: "utf8" }).trim();
  check("join: the clone commits as Bob's hub identity", ident === "bob@team.dev", ident);

  // Bob reopens the IDE on the clone; his hub connection must survive the restart.
  await bob.stop();
  bob = new IdeApp({ userData: path.join(tmp, "bob-data"), workspace: dest, logFile: path.join(out, "bob2.log") });
  await bob.start();
  const linked = await waitFor(bob, `window.rlm.invoke("team:workspaceProject", {})`, 15000);
  check("join: after restart the clone is linked to the hub project", linked?.group === "checkout" && linked?.project === "cart", JSON.stringify(linked));

  // ---- Bob branches, commits, pushes -------------------------------------------------------------
  await bob.cdp.invoke("git:checkout", { branch: "bob/discounts", create: true });
  await bob.cdp.invoke("fs:writeFile", {
    path: "cart.js",
    content: ["function total(items, discount = 0) {", "  const sum = items.reduce((s, i) => s + i.price * i.qty, 0);", "  return sum - sum * discount * 100;", "}", "", "module.exports = { total };", ""].join("\n"),
  });
  await bob.cdp.invoke("git:commit", { message: "Support discounts", paths: [] });
  await click(bob, "activity-git");
  await click(bob, "git-push");
  const pushed = await waitFor(bob, `[...document.querySelectorAll("body *")].some((e) => e.children.length === 0 && e.textContent === "Pushed")`, 20000);
  check("git: Bob pushed his branch from the Source control panel", pushed);
  const denied = await bob.cdp
    .eval(`window.rlm.invoke("team:git", { op: "push", branch: "main" }).then(() => "allowed", (e) => String(e.message))`)
    .catch((e) => String(e));
  check("git: pushing to the protected main branch is refused", /protected/.test(denied), denied.slice(-90));
  const pr = await bob.cdp.invoke("team:call", { connId: conn.id, method: "POST", path: "/api/projects/checkout/cart/pulls", body: { title: "Support discounts", body: "Adds an optional discount to `total`.", source: "bob/discounts" } });
  check("pr: Bob opened a pull request", pr?.number > 0, `!${pr?.number}`);

  // ---- Alice reviews and merges ------------------------------------------------------------------
  const badge = await waitFor(alice, `document.querySelector('${tid("activity-team")} span')?.innerText`, 10000);
  check("pr: Alice gets a live notification badge", Number(badge) >= 1, `badge ${badge}`);
  await click(alice, "team-inbox");
  await waitFor(alice, `document.querySelector('${tid("team-inbox-list")}')?.innerText.includes("Support discounts")`);
  await ui(alice, `[...document.querySelectorAll('${tid("team-inbox-list")} button')].find((b) => b.innerText.includes("Support discounts")).click()`);
  const diff = await waitFor(alice, `document.querySelector('${tid("pull-detail")}') && (document.querySelector('${tid("pull-tab-files")}').click(), document.querySelector('${tid("diff-view")}')?.innerText)`, 15000);
  check("pr: Alice sees the diff in the IDE", diff?.includes("discount * 100"));
  await alice.screenshot(path.join(out, "2-alice-pull-request.png"));
  await click(alice, "pull-tab-conversation");
  await click(alice, "pull-merge");
  const merged = await waitFor(alice, `document.querySelector('${tid("pull-detail")}')?.innerText.includes("merged by Alice")`, 15000);
  check("pr: Alice merged it on the hub", merged);

  // ---- Alice pulls, finds the bug, reports it ----------------------------------------------------
  await alice.cdp.invoke("team:git", { op: "pull" });
  const pulled = readFileSync(path.join(aliceWs, "cart.js"), "utf8");
  check("git: Alice pulled the merge", pulled.includes("discount * 100"));
  await click(alice, "activity-team");
  await ui(alice, `document.querySelector('${tid("team-main")}') && null`);
  await alice.cdp.eval(`document.querySelector('${tid("team-home")}').click()`);
  await sleep(300);
  // Open the project and use its Report a bug button.
  await ui(alice, `[...document.querySelectorAll("aside button")].find((b) => b.innerText.trim().startsWith("cart"))?.click()`);
  await click(alice, "project-report-bug");
  await waitFor(alice, `document.querySelector('${tid("bug-dialog")}') && document.querySelector('${tid("bug-target")}').value.includes("cart")`);
  await type(alice, "bug-title", "Discounted totals are negative");
  await type(alice, "bug-file", "cart.js");
  await type(alice, "bug-line", "3");
  await type(alice, "bug-error", "AssertionError: total([{price: 10, qty: 2}], 0.1) expected 18, got -180");
  await alice.screenshot(path.join(out, "3-alice-bug-dialog.png"));
  await click(alice, "bug-submit");
  const sent = await waitFor(alice, `document.querySelector('${tid("bug-sent")}')?.innerText`, 20000);
  check("bug: report routed to Bob, who wrote the line", sent?.includes("Sent to Bob") && /wrote commit/.test(sent ?? ""), sent?.replace(/\s+/g, " ").slice(0, 120));
  await alice.screenshot(path.join(out, "4-alice-bug-sent.png"));

  // ---- Bob gets it and answers -------------------------------------------------------------------
  const bobBadge = await waitFor(bob, `document.querySelector('${tid("activity-team")} span')?.innerText`, 15000);
  check("bug: Bob is notified live", Number(bobBadge) >= 1, `badge ${bobBadge}`);
  await click(bob, "activity-team");
  await click(bob, "team-inbox");
  await waitFor(bob, `document.querySelector('${tid("team-inbox-list")}')?.innerText.includes("Discounted totals")`);
  await ui(bob, `[...document.querySelectorAll('${tid("team-inbox-list")} button')].find((b) => b.innerText.includes("Discounted totals")).click()`);
  const report = await waitFor(bob, `document.querySelector('${tid("issue-body")}')?.innerText`, 15000);
  const sections = ["Discounted totals are negative", "Error", "AssertionError", "Code at", "sum * discount * 100", "Origin", "Support discounts", "Merged through pull request", "Pushed to the hub by", "For the developer"];
  const missing = sections.filter((s) => !report?.includes(s));
  check("bug: Bob reads the full markdown report with blame, PR and pusher", !missing.length, missing.length ? `missing ${missing.join(", ")}` : `${report?.length} chars`);
  await bob.screenshot(path.join(out, "5-bob-bug-report.png"));
  await click(bob, "bug-status-fixed");
  await type(bob, "bug-comment", "discount is a fraction already; removed the * 100. Thanks!");
  await click(bob, "bug-send-status");
  await sleep(1000);
  const aliceInbox = await alice.cdp.invoke("team:call", { connId: "local", method: "GET", path: "/api/notifications" });
  const fb = aliceInbox.find((n) => n.kind === "bug_status");
  check("feedback: Alice receives Bob's fix status and comment", fb && /fixed/.test(fb.title) && fb.body.includes("fraction"), fb ? `${fb.title}: ${fb.body}` : "none");
  const errs = [...alice.cdp.console, ...bob.cdp.console].filter((c) => c.type === "exception");
  check("no renderer exceptions", !errs.length, errs.map((e) => e.text).join(" | ").slice(0, 200));
} catch (e) {
  check("run completed", false, e instanceof Error ? e.stack?.split("\n").slice(0, 3).join(" ") : String(e));
  await alice.screenshot(path.join(out, "error-alice.png")).catch(() => undefined);
  await bob.screenshot(path.join(out, "error-bob.png")).catch(() => undefined);
} finally {
  await bob.stop();
  await alice.stop();
  writeFileSync(path.join(out, "results.json"), JSON.stringify(results, null, 2));
  rmSync(tmp, { recursive: true, force: true });
  const failed = results.filter((r) => !r.ok).length;
  console.log(`\n${results.length - failed}/${results.length} checks passed`);
  process.exitCode = failed ? 1 : 0;
}
