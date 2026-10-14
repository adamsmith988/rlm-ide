import { BrowserWindow, ipcMain, MessageChannelMain, utilityProcess, type UtilityProcess } from "electron";
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
  state: InferenceState = { device: "none", status: "not started", detail: "" };

  constructor(
    private readonly orch: OrchestratorHost,
    private readonly settings: () => Settings,
    private readonly onState: (s: InferenceState) => void,
    private readonly loadInferencePage: (win: BrowserWindow) => void,
  ) {
    ipcMain.on("inference:status", (_e, s: string) => this.set({ status: s }));
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
    if (pref !== "cpu") {
      const ok = await this.startWebGpu();
      if (ok) return;
      if (pref === "webgpu") {
        this.orch.post({ kind: "inference-unavailable", reason: `WebGPU with shader-f16 is not available: ${this.state.detail}` });
        return;
      }
    }
    this.startCpu(modelRoot);
  }

  private startWebGpu(): Promise<boolean> {
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
      const onHello = (_e: Electron.IpcMainEvent, info: { webgpu: boolean; f16: boolean; adapter: string; powerPreference?: string; error?: string }) => {
        if (_e.sender !== win.webContents) return;
        ipcMain.removeListener("inference:hello", onHello);
        if (!info.webgpu || !info.f16) {
          this.set({ detail: info.error ?? (info.webgpu ? "GPU lacks shader-f16" : "no WebGPU adapter") });
          win.destroy();
          this.win = null;
          resolve(false);
          return;
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

  private startCpu(modelRoot: string): void {
    const child = utilityProcess.fork(path.join(__dirname, "inference-cpu.js"), [], { serviceName: "rlm-inference-cpu", stdio: "pipe" });
    this.cpu = child;
    child.stderr?.on("data", (d: Buffer) => process.stderr.write(`[infer-cpu] ${d}`));
    child.on("message", (m: { kind: string; status?: string }) => {
      if (m.kind === "status" && m.status) this.set({ status: m.status });
    });
    child.on("exit", () => {
      if (this.cpu === child) this.cpu = null;
    });
    const { port1, port2 } = new MessageChannelMain();
    child.postMessage({ kind: "port", modelRoot, modelDir: BUILTIN.agentDir }, [port2]);
    this.orch.post({ kind: "inference-port", device: "cpu" }, [port1]);
    this.set({ device: "cpu", status: "loading", detail: "CPU (onnxruntime-node). Expect slow generation." });
  }

  stop(): void {
    this.win?.destroy();
    this.win = null;
    this.cpu?.kill();
    this.cpu = null;
    this.set({ device: "none", status: "stopped", detail: "" });
  }
}
