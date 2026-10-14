import { LocalGenerator, type InferenceRequest } from "@rlm/inference/src/generate";

declare global {
  interface Window {
    inferenceHost: { hello(info: unknown): void; status(s: string): void };
  }
}

async function probe(): Promise<{ webgpu: boolean; f16: boolean; adapter: string; powerPreference?: string; error?: string }> {
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter(o?: unknown): Promise<{ features: Set<string>; info?: { vendor?: string; architecture?: string; description?: string } } | null> } }).gpu;
  if (!gpu) return { webgpu: false, f16: false, adapter: "", error: "navigator.gpu missing" };
  try {
    let first: { f16: boolean; adapter: string } | null = null;
    for (const powerPreference of ["high-performance", "low-power"]) {
      const a = await gpu.requestAdapter({ powerPreference });
      if (!a) continue;
      const info = a.info ? `${a.info.vendor ?? ""} ${a.info.architecture ?? ""} ${a.info.description ?? ""}`.trim() : "";
      const f16 = a.features.has("shader-f16");
      if (f16) return { webgpu: true, f16, adapter: info, powerPreference };
      first ??= { f16, adapter: info };
    }
    if (!first) return { webgpu: false, f16: false, adapter: "", error: "no adapter" };
    return { webgpu: true, ...first };
  } catch (e) {
    return { webgpu: false, f16: false, adapter: "", error: String(e) };
  }
}

window.addEventListener("message", (ev) => {
  if (ev.data?.type !== "inference-port" || !ev.ports[0]) return;
  const port = ev.ports[0];
  const cfg = ev.data.config as { modelRoot: string; modelDir: string; wasmPaths: string; powerPreference?: "high-performance" | "low-power" };
  const gen = new LocalGenerator({
    modelRoot: cfg.modelRoot,
    modelDir: cfg.modelDir,
    device: "webgpu",
    wasmPaths: cfg.wasmPaths,
    powerPreference: cfg.powerPreference,
    onStatus: (s) => window.inferenceHost.status(s),
  });
  port.onmessage = (m) => gen.handle(m.data as InferenceRequest, (r) => port.postMessage(r));
  gen
    .load()
    .then(() => port.postMessage({ type: "ready" }))
    .catch((e: unknown) => {
      window.inferenceHost.status(`failed: ${String(e)}`);
      port.postMessage({ type: "error", id: "*", message: `local model failed to load: ${String(e)}` });
    });
});

void probe().then((info) => window.inferenceHost.hello(info));
