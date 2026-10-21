import { BrowserWindow, ipcMain, MessageChannelMain, utilityProcess, type UtilityProcess } from "electron";
import { execFile } from "node:child_process";
import { readdirSync, statSync } from "node:fs";
import path from "node:path";
import type { Settings } from "@rlm/contracts";
import { BUILTIN } from "@rlm/contracts";
import type { OrchestratorHost } from "./orchestrator_host";
import { paths } from "./paths";

export interface InferenceState {
  device: "webgpu" | "cpu" | "none";
  status: string;
  detail: string;
}

/** Hosts the built-in agent model. WebGPU runs in a hidden sandboxed window (utilityProcess has
 * no WebGPU); without a usable GPU it falls back to a CPU utilityProcess (onnxruntime-node). */
export class InferenceHost {
  private win: BrowserWindow | null = null;
  private cpu: UtilityProcess | null = null;
  private starting: Promise<void> | null = null;
  private gpuBlocked: string | null = null;
  state: InferenceState = { device: "none", status: "not started", detail: "" };

  constructor(
    private readonly orch: OrchestratorHost,
    private readonly settings: () => Settings,
    private readonly onState: (s: InferenceState) => void,
    private readonly loadInferencePage: (win: BrowserWindow) => void,
  ) {
    ipcMain.on("inference:status", (_e, s: string) => {
      // A WebGPU runtime that ran out of GPU memory takes Chromium's GPU process (which also draws the UI) down with
      // it, and after a few of those Chromium turns WebGPU off. In auto mode, stay on the CPU until the device setting changes.
      if (this.state.device === "webgpu" && /out of memory/i.test(s)) this.gpuBlocked = "the GPU ran out of memory while running it; another app is using most of the GPU memory";
      this.set({ status: s });
    });
  }

  private set(p: Partial<InferenceState>): void {
    this.state = { ...this.state, ...p };
    this.onState(this.state);
  }

  /** Called when the orchestrator has no usable port: (re)start the runtime and hand over a new one. */
  ensure(): Promise<void> {
    if (!this.starting && (this.win || this.cpu)) this.stop();
    if (!this.starting) {
      this.starting = this.start().finally(() => (this.starting = null));
    }
    return this.starting;
  }

  private async start(): Promise<void> {
    const modelRoot = paths.findModelDir(BUILTIN.agentDir);
    if (!modelRoot) {
      this.set({ device: "none", status: "not installed", detail: "Install FrogNano in Settings → Models." });
      this.orch.post({ kind: "inference-unavailable", reason: "The built-in model is not installed. Open Settings → Models and install FrogNano 4B, or pick an API model." });
      return;
    }
    const pref = this.settings().local.preferredDevice;
    if (pref !== "cpu" && !(pref === "auto" && this.gpuBlocked)) {
      const ok = await this.startWebGpu(modelRoot);
      if (ok) return;
      if (pref === "webgpu") {
        this.orch.post({ kind: "inference-unavailable", reason: `WebGPU with shader-f16 is not available: ${this.state.detail}` });
        return;
      }
    }
    this.startCpu(modelRoot, pref === "cpu" ? "" : (this.gpuBlocked ?? this.state.detail));
  }

  /** The device settings changed: drop the runtime (the next request starts a fresh one) and try the GPU again. */
  devicesChanged(): void {
    this.gpuBlocked = null;
    this.stop();
  }

  /** Free memory of the emptiest NVIDIA GPU (nvidia-smi ships with the driver on Windows and Linux), or null. */
  private nvidiaFreeMB(): Promise<number | null> {
    return new Promise((resolve) =>
      execFile("nvidia-smi", ["--query-gpu=memory.free", "--format=csv,noheader,nounits"], { timeout: 5000, windowsHide: true }, (err, out) => {
        const free = err ? [] : String(out).split("\n").map(Number).filter((n) => Number.isFinite(n) && n > 0);
        resolve(free.length ? Math.max(...free) : null);
      }),
    );
  }

  private weightsMB(modelRoot: string): number {
    const dir = path.join(modelRoot, BUILTIN.agentDir, "onnx");
    try {
      return Math.round(readdirSync(dir).reduce((sum, f) => sum + statSync(path.join(dir, f)).size, 0) / 2 ** 20);
    } catch {
      return 0;
    }
  }

  private startWebGpu(modelRoot: string): Promise<boolean> {
    return new Promise((resolve) => {
      const win = new BrowserWindow({
        show: false,
        webPreferences: {
          preload: path.join(__dirname, "../preload/inference.cjs"),
          sandbox: true,
          contextIsolation: true,
          nodeIntegration: false,
          backgroundThrottling: false,
        },
      });
      this.win = win;
      const giveUp = (detail: string) => {
        this.set({ detail });
        win.destroy();
        this.win = null;
        resolve(false);
      };
      const onHello = async (_e: Electron.IpcMainEvent, info: { webgpu: boolean; f16: boolean; adapter: string; powerPreference?: string; error?: string }) => {
        if (_e.sender !== win.webContents) return;
        ipcMain.removeListener("inference:hello", onHello);
        console.log(`[inference] WebGPU probe: ${JSON.stringify(info)}`);
        if (!info.webgpu || !info.f16) return giveUp(info.error ?? (info.webgpu ? `${info.adapter || "the GPU"} offers no shader-f16 to WebGPU` : "no WebGPU adapter"));
        if (/nvidia/i.test(info.adapter) && this.settings().local.preferredDevice === "auto") {
          const free = await this.nvidiaFreeMB();
          const need = this.weightsMB(modelRoot) + this.settings().local.gpuHeadroomMB;
          if (free !== null && free < need) return giveUp(`the GPU has ${(free / 1024).toFixed(1)} GB free and FrogNano needs about ${(need / 1024).toFixed(1)} GB; another app is using most of the GPU memory`);
        }
        const { port1, port2 } = new MessageChannelMain();
        win.webContents.postMessage("inference-port", { modelRoot: "rlm-model://models/", modelDir: BUILTIN.agentDir, wasmPaths: "rlm-model://ort/", powerPreference: info.powerPreference }, [port2]);
        this.orch.post({ kind: "inference-port", device: "webgpu" }, [port1]);
        this.set({ device: "webgpu", status: "loading", detail: info.adapter });
        resolve(true);
      };
      ipcMain.on("inference:hello", onHello);
      win.on("closed", () => {
        if (this.win === win) this.win = null;
      });
      win.webContents.on("render-process-gone", (_e, d) => {
        this.set({ status: "crashed", detail: d.reason });
        this.win = null;
        this.orch.post({ kind: "inference-unavailable", reason: `local model process crashed (${d.reason})` });
      });
      setTimeout(() => {
        if (this.state.device === "none" && this.win === win) {
          ipcMain.removeListener("inference:hello", onHello);
          win.destroy();
          this.win = null;
          this.set({ detail: "inference window did not respond" });
          resolve(false);
        }
      }, 20000);
      this.loadInferencePage(win);
    });
  }

  private startCpu(modelRoot: string, gpuProblem: string): void {
    const child = utilityProcess.fork(path.join(__dirname, "inference-cpu.js"), [], { serviceName: "rlm-inference-cpu", stdio: "pipe" });
    this.cpu = child;
    child.stderr?.on("data", (d: Buffer) => process.stderr.write(`[infer-cpu] ${d}`));
    child.on("message", (m: { kind: string; status?: string }) => {
      if (m.kind === "status" && m.status) {
        console.log(`[infer-cpu] ${m.status}`);
        this.set({ status: m.status });
      }
    });
    child.on("exit", (code) => {
      if (this.cpu !== child) return;
      this.cpu = null;
      const reason = `the local model's CPU process stopped unexpectedly (exit code ${code})`;
      console.warn(`[inference] ${reason}`);
      this.set({ device: "none", status: "crashed", detail: reason });
      this.orch.post({ kind: "inference-unavailable", reason });
    });
    const { port1, port2 } = new MessageChannelMain();
    child.postMessage({ kind: "port", modelRoot, modelDir: BUILTIN.agentDir, cpuThreads: this.settings().local.cpuThreads }, [port2]);
    this.orch.post({ kind: "inference-port", device: "cpu" }, [port1]);
    const remedy = /memory/i.test(gpuProblem) ? "Close the app using the GPU and change Settings → Local inference → Device to retry, or pick an API model." : "Update the graphics driver, or pick an API model.";
    const why = gpuProblem ? `GPU not usable (${gpuProblem}), so FrogNano runs on the CPU and generates slowly. ${remedy}` : "CPU (onnxruntime-node). Expect slow generation.";
    console.warn(`[inference] ${why}`);
    this.set({ device: "cpu", status: "loading", detail: why });
  }

  stop(): void {
    this.win?.destroy();
    this.win = null;
    this.cpu?.kill();
    this.cpu = null;
    this.set({ device: "none", status: "stopped", detail: "" });
  }
}
