import type { IpcChannel, IpcRequest, RlmApi } from "@rlm/contracts";

type Api = RlmApi & { pushEditorState(state: unknown): void };

declare global {
  interface Window {
    rlm: Api;
  }
}

export const api = window.rlm;

export function call<T = unknown>(channel: IpcChannel, payload: IpcRequest<typeof channel> = {} as never): Promise<T> {
  return api.invoke(channel, payload as never) as Promise<T>;
}

export function errText(e: unknown): string {
  const m = e instanceof Error ? e.message : String(e);
  return m.replace(/^Error invoking remote method 'rpc': (Error: )?/, "");
}
