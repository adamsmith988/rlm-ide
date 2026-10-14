import { create } from "zustand";
import type { TeamStateView } from "@rlm/contracts";
import { api, call } from "../../api";

export type TeamView =
  | { kind: "home" }
  | { kind: "inbox" }
  | { kind: "group"; connId: string; group: string }
  | { kind: "project"; connId: string; group: string; project: string; tab?: ProjectTab; item?: number };

export type ProjectTab = "code" | "issues" | "bugs" | "pulls" | "activity";

export interface BugDraft {
  connId?: string;
  group?: string;
  project?: string;
  file?: string;
  line?: number;
  error?: string;
}

interface TeamStore {
  state: TeamStateView | null;
  view: TeamView;
  /** Bumped by live hub events so open views refetch. */
  tick: number;
  unread: number;
  bugDraft: BugDraft | null;
  load(): Promise<void>;
  go(v: TeamView): void;
  reportBug(d: BugDraft): void;
  refreshUnread(): Promise<void>;
}

/** Calls a hub API endpoint through main for one connection. */
export function hub<T>(connId: string, method: "GET" | "POST" | "PATCH" | "DELETE", path: string, body?: unknown): Promise<T> {
  return call<T>("team:call", { connId, method, path, body });
}

export const useTeam = create<TeamStore>((set, get) => ({
  state: null,
  view: { kind: "home" },
  tick: 0,
  unread: 0,
  bugDraft: null,
  async load() {
    set({ state: await call<TeamStateView>("team:state", {}) });
    void get().refreshUnread();
  },
  go(view) {
    set({ view });
  },
  reportBug(d) {
    set({ bugDraft: d });
  },
  async refreshUnread() {
    const conns = get().state?.connections.filter((c) => c.status === "online") ?? [];
    let n = 0;
    for (const c of conns) n += (await hub<unknown[]>(c.id, "GET", "/api/notifications?unread=1").catch(() => [])).length;
    set({ unread: n });
  },
}));

let wired = false;
/** Subscribes the store to main's team events once. */
export function wireTeam(onNotify: (title: string, body: string) => void): () => void {
  if (wired) return () => undefined;
  wired = true;
  const offs = [
    api.onTeamState((state) => {
      useTeam.setState({ state });
      void useTeam.getState().refreshUnread();
    }),
    api.onTeamEvent((e) => {
      useTeam.setState({ tick: useTeam.getState().tick + 1 });
      if (e.type === "notification") {
        const n = (e.data as { notification?: { title: string; body: string } }).notification;
        if (n) onNotify(n.title, n.body);
        void useTeam.getState().refreshUnread();
      }
    }),
  ];
  void useTeam.getState().load();
  return () => {
    wired = false;
    offs.forEach((f) => f());
  };
}

export const ROLES = ["reporter", "developer", "maintainer", "owner"] as const;
export const atLeast = (role: string | undefined, min: (typeof ROLES)[number]) => ROLES.indexOf((role ?? "reporter") as never) >= ROLES.indexOf(min);

export function ago(t: number): string {
  const s = Math.max(1, Math.round((Date.now() - t) / 1000));
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  if (s < 86400) return `${Math.round(s / 3600)}h ago`;
  return new Date(t).toLocaleDateString();
}
