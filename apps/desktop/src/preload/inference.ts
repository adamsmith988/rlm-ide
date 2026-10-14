import { contextBridge, ipcRenderer } from "electron";

/** Preload for the hidden inference window: hands the MessagePort to the page and reports status. */
ipcRenderer.on("inference-port", (e, config) => {
  window.postMessage({ type: "inference-port", config }, "*", e.ports);
});

contextBridge.exposeInMainWorld("inferenceHost", {
  hello: (info: unknown) => ipcRenderer.send("inference:hello", info),
  status: (s: string) => ipcRenderer.send("inference:status", s),
});
