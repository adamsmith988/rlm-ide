import { contextBridge, ipcRenderer } from "electron";
import type { RlmApi } from "@rlm/contracts";

function on<T>(channel: string) {
  return (cb: (v: T) => void) => {
    const h = (_e: unknown, v: T) => cb(v);
    ipcRenderer.on(channel, h);
    return () => ipcRenderer.removeListener(channel, h);
  };
}

const api: RlmApi & { pushEditorState(state: unknown): void } = {
  invoke: (channel, payload) => ipcRenderer.invoke("rpc", channel, payload),
  onTurnEvent: on("turn:event"),
  onPtyData: on("pty:data"),
  onPtyExit: on("pty:exit"),
  onFsChanged: on("fs:changed"),
  onStatus: on("status"),
  onModelProgress: on("models:progress"),
  onSettings: on("settings"),
  onBrowserState: on("browser:state"),
  onTeamState: on("team:state"),
  onHubState: on("hub:state"),
  onTaskBoard: on("task:board"),
  onTeamEvent: on("team:event"),
  browserBounds: (bounds, visible) => ipcRenderer.send("browser:bounds", bounds, visible),
  pushEditorState: (state) => ipcRenderer.send("editor:state", state),
  window: {
    minimize: () => ipcRenderer.send("window:minimize"),
    maximize: () => ipcRenderer.send("window:maximize"),
    close: () => ipcRenderer.send("window:close"),
    platform: process.platform,
  },
};

contextBridge.exposeInMainWorld("rlm", api);
