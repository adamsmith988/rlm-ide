/** CPU fallback for the built-in agent model (onnxruntime-node via Transformers.js). */
import { LocalGenerator, type InferenceRequest } from "@rlm/inference/src/generate";

interface PortLike {
  on(ev: "message", cb: (e: { data: unknown }) => void): void;
  postMessage(m: unknown): void;
  start(): void;
}
const parent = (process as unknown as { parentPort: { on(ev: "message", cb: (e: { data: Record<string, unknown>; ports: PortLike[] }) => void): void; postMessage(m: unknown): void } }).parentPort;

parent.on("message", (e) => {
  if (e.data.kind !== "port") return;
  const port = e.ports[0];
  if (!port) return;
  const gen = new LocalGenerator({
    modelRoot: String(e.data.modelRoot),
    modelDir: String(e.data.modelDir),
    device: "cpu",
    onStatus: (s) => parent.postMessage({ kind: "status", status: s }),
  });
  port.on("message", (m) => gen.handle(m.data as InferenceRequest, (r) => port.postMessage(r)));
  port.start();
  gen
    .load()
    .then(() => port.postMessage({ type: "ready" }))
    .catch((err: unknown) => {
      parent.postMessage({ kind: "status", status: `failed: ${String(err)}` });
      port.postMessage({ type: "error", id: "*", message: `local model failed to load on CPU: ${String(err)}` });
    });
});
