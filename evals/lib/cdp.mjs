/** Minimal Chrome DevTools Protocol client for driving the IDE renderer. */
export class Cdp {
  static async connect(port, urlPart, timeoutMs = 30000) {
    const until = Date.now() + timeoutMs;
    for (;;) {
      try {
        const list = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
        const page = list.find((p) => p.type === "page" && p.url.includes(urlPart));
        if (page) {
          const c = new Cdp(page.webSocketDebuggerUrl);
          await c.open();
          return c;
        }
      } catch {
        /* not up yet */
      }
      if (Date.now() > until) throw new Error(`no CDP page matching ${urlPart} on :${port}`);
      await sleep(300);
    }
  }

  constructor(wsUrl) {
    this.wsUrl = wsUrl;
    this.id = 0;
    this.pending = new Map();
    this.console = [];
  }

  open() {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(this.wsUrl);
      this.ws.onopen = resolve;
      this.ws.onerror = reject;
      this.ws.onmessage = (m) => {
        const msg = JSON.parse(m.data);
        if (msg.id && this.pending.has(msg.id)) {
          this.pending.get(msg.id)(msg);
          this.pending.delete(msg.id);
        } else if (msg.method === "Runtime.exceptionThrown") {
          this.console.push({ type: "exception", text: msg.params.exceptionDetails?.exception?.description ?? msg.params.exceptionDetails?.text });
        } else if (msg.method === "Runtime.consoleAPICalled" && (msg.params.type === "error" || msg.params.type === "warning")) {
          this.console.push({ type: msg.params.type, text: msg.params.args.map((a) => a.value ?? a.description).join(" ") });
        }
      };
    });
  }

  send(method, params = {}) {
    return new Promise((resolve, reject) => {
      const id = ++this.id;
      this.pending.set(id, (msg) => (msg.error ? reject(new Error(`${method}: ${msg.error.message}`)) : resolve(msg.result)));
      this.ws.send(JSON.stringify({ id, method, params }));
    });
  }

  async eval(expression) {
    const r = await this.send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
    if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description ?? r.exceptionDetails.text);
    return r.result.value;
  }

  /** Call the app's typed IPC bridge (`window.rlm.invoke`). */
  invoke(channel, payload = {}) {
    return this.eval(`window.rlm.invoke(${JSON.stringify(channel)}, ${JSON.stringify(payload)})`);
  }

  async click(selector) {
    const ok = await this.eval(`(() => { const el = document.querySelector(${JSON.stringify(selector)}); if (!el) return false; el.click(); return true; })()`);
    if (!ok) throw new Error(`no element ${selector}`);
  }

  async typeInto(selector, text, { enter = false } = {}) {
    const ok = await this.eval(`(() => { const el = document.querySelector(${JSON.stringify(selector)}); if (!el) return false; el.focus(); return true; })()`);
    if (!ok) throw new Error(`no element ${selector}`);
    await this.send("Input.insertText", { text });
    if (enter) {
      await this.send("Input.dispatchKeyEvent", { type: "keyDown", key: "Enter", code: "Enter", windowsVirtualKeyCode: 13, text: "\r" });
      await this.send("Input.dispatchKeyEvent", { type: "keyUp", key: "Enter", code: "Enter", windowsVirtualKeyCode: 13 });
    }
  }

  async screenshot(file) {
    const { writeFileSync } = await import("node:fs");
    const r = await this.send("Page.captureScreenshot", { format: "png" });
    writeFileSync(file, Buffer.from(r.data, "base64"));
  }

  close() {
    this.ws?.close();
  }
}

export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
