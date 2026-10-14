import { useEffect, useState, type ReactNode } from "react";
import type { TeamStateView } from "@rlm/contracts";
import { call, errText } from "../../api";
import { useApp } from "../../store";
import { Circles } from "../../ui/LineArt";
import { ICopy, IPlus, IRefresh, ITrash } from "../../ui/icons";
import { ago, atLeast, hub, ROLES, useTeam } from "./teamStore";
import { ProjectView } from "./ProjectView";

type Conn = TeamStateView["connections"][number];

interface GroupSummary {
  id: string;
  slug: string;
  name: string;
  role: string;
  projects: number;
  members: number;
}

/** The team hub screen: hosting, connections, groups, projects, inbox. */
export function TeamScreen() {
  const { state, view } = useTeam();
  if (!state) return <div className="h-full bg-bg" />;
  return (
    <div className="flex h-full min-h-0">
      <TeamRail state={state} />
      <div className="min-w-0 flex-1 overflow-y-auto" data-testid="team-main">
        {view.kind === "home" && <Home state={state} />}
        {view.kind === "inbox" && <Inbox state={state} />}
        {view.kind === "group" && <GroupPage connId={view.connId} slug={view.group} />}
        {view.kind === "project" && <ProjectView key={`${view.connId}/${view.group}/${view.project}`} {...view} />}
      </div>
    </div>
  );
}

function StatusDot({ status }: { status: Conn["status"] }) {
  const color = status === "online" ? "var(--ok)" : status === "connecting" ? "var(--warn)" : status === "error" ? "var(--err)" : "var(--text-faint)";
  return <span className="inline-block h-2 w-2 shrink-0 rounded-full" style={{ background: color }} title={status} />;
}

function TeamRail({ state }: { state: TeamStateView }) {
  const { view, go, unread, tick } = useTeam();
  const [groups, setGroups] = useState<Record<string, GroupSummary[]>>({});
  useEffect(() => {
    for (const c of state.connections)
      if (c.status === "online")
        void hub<GroupSummary[]>(c.id, "GET", "/api/groups").then(
          (g) => setGroups((s) => ({ ...s, [c.id]: g })),
          () => undefined,
        );
  }, [state.connections.map((c) => `${c.id}:${c.status}`).join(","), tick]);

  const item = (active: boolean, onClick: () => void, children: ReactNode, testid?: string) => (
    <button
      data-testid={testid}
      onClick={onClick}
      className="flex w-full items-center gap-2 rounded-[10px] px-2.5 py-[6px] text-left text-[12.5px] transition-colors hover:bg-surface-2"
      style={active ? { background: "var(--accent)", color: "var(--accent-ink)" } : undefined}
    >
      {children}
    </button>
  );

  return (
    <aside className="flex w-[250px] shrink-0 flex-col gap-1 overflow-y-auto border-r border-line bg-surface px-2 py-3">
      {item(view.kind === "home", () => go({ kind: "home" }), <span className="font-semibold">Team hub</span>, "team-home")}
      {item(
        view.kind === "inbox",
        () => go({ kind: "inbox" }),
        <>
          <span>Inbox</span>
          {unread > 0 && <span className="ml-auto rounded-full bg-ink px-1.5 text-[10px] font-semibold leading-[16px] text-bg">{unread}</span>}
        </>,
        "team-inbox",
      )}
      {state.connections.map((c) => (
        <div key={c.id} className="mt-3">
          <div className="flex items-center gap-2 px-2.5 pb-1">
            <StatusDot status={c.status} />
            <span className="smallcaps truncate" title={c.label}>
              {c.hubName}
            </span>
          </div>
          {c.status !== "online" && <div className="px-2.5 text-[11px] text-faint">{c.error ?? c.status}</div>}
          {(groups[c.id] ?? []).map((g) => (
            <div key={g.id}>
              {item(view.kind === "group" && view.connId === c.id && view.group === g.slug, () => go({ kind: "group", connId: c.id, group: g.slug }), <span className="truncate">{g.name}</span>)}
              <GroupProjects connId={c.id} group={g.slug} />
            </div>
          ))}
          {c.status === "online" && !(groups[c.id] ?? []).length && <div className="px-2.5 text-[11px] text-faint">No groups yet</div>}
        </div>
      ))}
    </aside>
  );
}

function GroupProjects({ connId, group }: { connId: string; group: string }) {
  const { view, go, tick } = useTeam();
  const [projects, setProjects] = useState<Array<{ slug: string; name: string; openBugs: number }>>([]);
  useEffect(() => {
    void hub<{ projects: Array<{ slug: string; name: string; openBugs: number }> }>(connId, "GET", `/api/groups/${group}`).then(
      (g) => setProjects(g.projects),
      () => undefined,
    );
  }, [connId, group, tick]);
  return (
    <div className="ml-3 border-l border-line pl-1">
      {projects.map((p) => {
        const on = view.kind === "project" && view.connId === connId && view.group === group && view.project === p.slug;
        return (
          <button
            key={p.slug}
            onClick={() => go({ kind: "project", connId, group, project: p.slug })}
            className="flex w-full items-center gap-2 rounded-[8px] px-2 py-[4px] text-left text-[12px] hover:bg-surface-2"
            style={on ? { background: "var(--surface-2)", fontWeight: 600 } : undefined}
          >
            <span className="truncate">{p.name}</span>
            {p.openBugs > 0 && <span className="mono ml-auto text-[10.5px] text-[var(--err)]">{p.openBugs} bug{p.openBugs === 1 ? "" : "s"}</span>}
          </button>
        );
      })}
    </div>
  );
}

// ---- home ---------------------------------------------------------------------------------------

function Home({ state }: { state: TeamStateView }) {
  return (
    <div>
      <div className="relative overflow-hidden border-b border-line bg-bg">
        <Circles count={3} size={700} />
        <div className="relative mx-auto max-w-[900px] px-10 pb-8 pt-12">
          <div className="smallcaps">Work together without leaving the IDE</div>
          <h1 className="heading m-0 mt-1 text-[44px] leading-none">Team hub</h1>
          <p className="mt-3 max-w-[620px] text-[13px] text-dim">
            Host a hub on this machine or join a teammate's over the LAN, SSH, or an SSH jump host. Groups own projects with git push and pull, issues, pull requests with reviews, and bug reports
            that go straight to whoever wrote or pushed the code.
          </p>
        </div>
      </div>
      <div className="mx-auto grid max-w-[900px] gap-5 px-10 py-6">
        <ProfileCard state={state} />
        <HostCard state={state} />
        <ConnectCard state={state} />
        <ConnectionsCard state={state} />
      </div>
    </div>
  );
}

function Card({ title, sub, children, right }: { title: string; sub?: string; children: ReactNode; right?: ReactNode }) {
  return (
    <section className="card p-5">
      <div className="mb-3 flex items-start gap-3">
        <div className="flex-1">
          <h2 className="heading m-0 text-[20px]">{title}</h2>
          {sub && <div className="mt-0.5 text-[12px] text-dim">{sub}</div>}
        </div>
        {right}
      </div>
      {children}
    </section>
  );
}

function Field({ label, children, hint }: { label: string; children: ReactNode; hint?: string }) {
  return (
    <label className="flex flex-col gap-1">
      <span className="smallcaps">{label}</span>
      {children}
      {hint && <span className="text-[11px] text-faint">{hint}</span>}
    </label>
  );
}

function useRun() {
  const toast = useApp((s) => s.toast);
  const [busy, setBusy] = useState(false);
  const run = async <T,>(fn: () => Promise<T>, ok?: string): Promise<T | undefined> => {
    setBusy(true);
    try {
      const v = await fn();
      if (ok) toast(ok);
      return v;
    } catch (e) {
      toast(errText(e), "error");
      return undefined;
    } finally {
      setBusy(false);
    }
  };
  return { busy, run };
}

function copy(text: string) {
  void navigator.clipboard.writeText(text);
  useApp.getState().toast("Copied");
}

function ProfileCard({ state }: { state: TeamStateView }) {
  const [name, setName] = useState(state.profile.name);
  const [email, setEmail] = useState(state.profile.email);
  const { busy, run } = useRun();
  const save = () => run(async () => useTeam.setState({ state: await call<TeamStateView>("team:profile", { name, email }) }), "Profile saved");
  return (
    <Card title="You" sub="Shown on your commits, reviews and bug reports.">
      <div className="grid grid-cols-2 gap-3">
        <Field label="Name">
          <input className="input" value={name} onChange={(e) => setName(e.target.value)} data-testid="team-name" />
        </Field>
        <Field label="Email">
          <input className="input" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@company.dev" data-testid="team-email" />
        </Field>
      </div>
      <div className="mt-3 flex items-center gap-3">
        <button className="btn btn-sm btn-primary" disabled={busy || (name === state.profile.name && email === state.profile.email)} onClick={() => void save()} data-testid="team-save-profile">
          Save
        </button>
        <span className="mono min-w-0 flex-1 truncate text-[11px] text-faint" title={state.publicKey}>
          {state.publicKey}
        </span>
        <button className="iconbtn" title="Copy SSH public key" onClick={() => copy(state.publicKey)}>
          <ICopy size={13} />
        </button>
      </div>
    </Card>
  );
}

function HostCard({ state }: { state: TeamStateView }) {
  const [cfg, setCfg] = useState(state.host);
  const { busy, run } = useRun();
  useEffect(() => setCfg(state.host), [state.host]);
  const apply = (next: typeof cfg) =>
    run(async () => {
      const s = await call<TeamStateView>("team:host", next);
      useTeam.setState({ state: s });
      if (next.enabled && s.hostError) throw new Error(s.hostError);
    }, next.enabled ? "Hub is running" : "Hub stopped");
  const h = state.hosting;
  const addrs = h ? (h.addresses.length ? h.addresses : ["127.0.0.1"]) : [];
  return (
    <Card
      title="Host a hub on this machine"
      sub="Your IDE serves the API and git. Teammates connect to it; you are the owner."
      right={
        <button className="btn btn-sm" style={h ? undefined : { background: "var(--accent)", color: "var(--accent-ink)" }} disabled={busy} onClick={() => void apply({ ...cfg, enabled: !h })} data-testid="team-host-toggle">
          {h ? "Stop hosting" : "Start hosting"}
        </button>
      }
    >
      <div className="grid grid-cols-4 gap-3">
        <Field label="Hub name">
          <input className="input" value={cfg.name} placeholder={`${state.profile.name}'s hub`} onChange={(e) => setCfg({ ...cfg, name: e.target.value })} />
        </Field>
        <Field label="Who can reach it">
          <select className="input" value={cfg.lan ? "lan" : "local"} onChange={(e) => setCfg({ ...cfg, lan: e.target.value === "lan" })}>
            <option value="lan">Local network</option>
            <option value="local">This machine only</option>
          </select>
        </Field>
        <Field label="HTTP port">
          <input className="input" type="number" value={cfg.httpPort} onChange={(e) => setCfg({ ...cfg, httpPort: Number(e.target.value) })} />
        </Field>
        <Field label="SSH port">
          <input className="input" type="number" value={cfg.sshPort ?? ""} placeholder="off" onChange={(e) => setCfg({ ...cfg, sshPort: e.target.value ? Number(e.target.value) : null })} />
        </Field>
      </div>
      <label className="mt-3 flex items-center gap-2 text-[12px]">
        <input type="checkbox" checked={cfg.discovery} onChange={(e) => setCfg({ ...cfg, discovery: e.target.checked })} /> Announce on the local network so teammates see it
      </label>
      {state.hostError && <div className="mt-3 text-[12px] text-[var(--err)]">{state.hostError}</div>}
      {h && (
        <div className="mt-4 rounded-[12px] border border-line p-3 text-[12px]" data-testid="team-host-info">
          <div className="smallcaps mb-2">Teammates connect with</div>
          {addrs.map((a) => (
            <div key={a} className="mono flex items-center gap-2 py-0.5">
              <span className="w-[70px] text-faint">LAN</span>
              <span className="selectable">http://{a}:{h.httpPort}</span>
              <button className="iconbtn" onClick={() => copy(`http://${a}:${h.httpPort}`)}>
                <ICopy size={12} />
              </button>
            </div>
          ))}
          {h.sshPort &&
            addrs.map((a) => (
              <div key={`s${a}`} className="mono flex items-center gap-2 py-0.5">
                <span className="w-[70px] text-faint">SSH</span>
                <span className="selectable">
                  {a}:{h.sshPort}
                </span>
                <button className="iconbtn" onClick={() => copy(`${a}:${h.sshPort}`)}>
                  <ICopy size={12} />
                </button>
              </div>
            ))}
          {h.sshFingerprint && <div className="mono mt-1 text-[11px] text-faint">host key {h.sshFingerprint}</div>}
          <div className="mt-2 text-[11.5px] text-dim">
            From outside the network, teammates use the SSH port (forward it on your router or VPN) or an SSH jump host that can reach this machine. Create a group below, then share an invite code.
          </div>
        </div>
      )}
    </Card>
  );
}

type Kind = "http" | "ssh" | "jump";

function ConnectCard({ state }: { state: TeamStateView }) {
  const [kind, setKind] = useState<Kind>("http");
  const [url, setUrl] = useState("http://");
  const [host, setHost] = useState("");
  const [port, setPort] = useState(7422);
  const [user, setUser] = useState("");
  const [password, setPassword] = useState("");
  const [hubHost, setHubHost] = useState("127.0.0.1");
  const [hubPort, setHubPort] = useState(7420);
  const [code, setCode] = useState("");
  const [token, setToken] = useState("");
  const [useToken, setUseToken] = useState(false);
  const { busy, run } = useRun();

  const transport =
    kind === "http" ? { kind, url: url.trim() } : kind === "ssh" ? { kind, host: host.trim(), port } : { kind, host: host.trim(), port, username: user.trim(), hubHost: hubHost.trim(), hubPort };
  const connect = () =>
    run(async () => {
      const s = await call<TeamStateView>("team:connect", { transport, code: useToken ? undefined : code.trim(), token: useToken ? token.trim() : undefined, password: kind === "jump" && password ? password : undefined });
      useTeam.setState({ state: s });
      setCode("");
      setToken("");
    }, "Connected");

  const tabs: Array<[Kind, string]> = [
    ["http", "LAN / URL"],
    ["ssh", "SSH"],
    ["jump", "SSH jump host"],
  ];
  return (
    <Card title="Join a hub" sub="Use the invite code a teammate gave you. Your SSH key is registered automatically for git over SSH.">
      {state.discovered.length > 0 && (
        <div className="mb-4">
          <div className="smallcaps mb-1">On your network</div>
          <div className="flex flex-wrap gap-2">
            {state.discovered.map((d) => (
              <button
                key={d.hubId}
                className="btn btn-sm"
                data-testid="team-discovered"
                onClick={() => {
                  setKind("http");
                  setUrl(`http://${d.address}:${d.httpPort}`);
                }}
              >
                {d.name} · {d.address}
              </button>
            ))}
          </div>
        </div>
      )}
      <div className="seg mb-3">
        {tabs.map(([k, l]) => (
          <button key={k} className={kind === k ? "on" : ""} onClick={() => setKind(k)} data-testid={`team-kind-${k}`}>
            {l}
          </button>
        ))}
      </div>
      <div className="grid grid-cols-2 gap-3">
        {kind === "http" && (
          <Field label="Hub address" hint="As shown on the host, e.g. http://192.168.1.20:7420">
            <input className="input" value={url} onChange={(e) => setUrl(e.target.value)} data-testid="team-url" />
          </Field>
        )}
        {kind !== "http" && (
          <>
            <Field label={kind === "ssh" ? "Hub host" : "SSH server"} hint={kind === "jump" ? "Any machine you can ssh into that reaches the hub" : "Hub address and its SSH port"}>
              <input className="input" value={host} onChange={(e) => setHost(e.target.value)} placeholder={kind === "ssh" ? "hub.example.org" : "bastion.example.org"} />
            </Field>
            <Field label="SSH port">
              <input className="input" type="number" value={port} onChange={(e) => setPort(Number(e.target.value))} />
            </Field>
          </>
        )}
        {kind === "jump" && (
          <>
            <Field label="SSH user">
              <input className="input" value={user} onChange={(e) => setUser(e.target.value)} />
            </Field>
            <Field label="SSH password" hint="Leave empty to use your IDE key (add it to authorized_keys)">
              <input className="input" type="password" value={password} onChange={(e) => setPassword(e.target.value)} />
            </Field>
            <Field label="Hub as seen from that server">
              <input className="input" value={hubHost} onChange={(e) => setHubHost(e.target.value)} />
            </Field>
            <Field label="Hub HTTP port">
              <input className="input" type="number" value={hubPort} onChange={(e) => setHubPort(Number(e.target.value))} />
            </Field>
          </>
        )}
        <Field label={useToken ? "Access token" : "Invite code"}>
          {useToken ? (
            <input className="input mono" value={token} onChange={(e) => setToken(e.target.value)} placeholder="rlmt_…" />
          ) : (
            <input className="input mono" value={code} onChange={(e) => setCode(e.target.value)} placeholder="ABCD-EFGH-JKLM" data-testid="team-code" />
          )}
        </Field>
      </div>
      <div className="mt-3 flex items-center gap-3">
        <button className="btn btn-primary btn-sm" disabled={busy || !(useToken ? token.trim() : code.trim())} onClick={() => void connect()} data-testid="team-connect">
          {busy ? "Connecting…" : "Connect"}
        </button>
        <button className="btn btn-ghost btn-sm" onClick={() => setUseToken(!useToken)}>
          {useToken ? "I have an invite code" : "I already have an account token"}
        </button>
      </div>
    </Card>
  );
}

function ConnectionsCard({ state }: { state: TeamStateView }) {
  const { busy, run } = useRun();
  const { go } = useTeam();
  const [groupName, setGroupName] = useState("");
  const online = state.connections.filter((c) => c.status === "online");
  const [target, setTarget] = useState<string>("");
  const connId = target || online[0]?.id || "";
  if (!state.connections.length) return null;
  return (
    <Card title="Hubs" sub="Hubs you belong to. Create a group to start a team.">
      <div className="flex flex-col gap-2">
        {state.connections.map((c) => (
          <div key={c.id} className="flex items-center gap-3 rounded-[12px] border border-line px-3 py-2 text-[12.5px]">
            <StatusDot status={c.status} />
            <div className="min-w-0 flex-1">
              <div className="font-semibold">{c.hubName}</div>
              <div className="truncate text-[11px] text-faint">
                {c.transport.kind === "http" ? c.transport.url : c.transport.kind === "ssh" ? `ssh ${c.transport.host}:${c.transport.port}` : `via ${c.transport.username}@${c.transport.host}`} · signed in as {c.userName}
                {c.error && c.status !== "online" ? ` · ${c.error}` : ""}
              </div>
            </div>
            <button className="iconbtn" title="Reconnect" onClick={() => void run(async () => useTeam.setState({ state: await call<TeamStateView>("team:reconnect", { connId: c.id }) }))}>
              <IRefresh size={13} />
            </button>
            {c.id !== "local" && (
              <button
                className="iconbtn"
                title="Leave this hub on this machine"
                onClick={() => {
                  if (confirm(`Forget ${c.hubName} on this machine? Your account on the hub stays.`)) void run(async () => useTeam.setState({ state: await call<TeamStateView>("team:disconnect", { connId: c.id }) }));
                }}
              >
                <ITrash size={13} />
              </button>
            )}
          </div>
        ))}
      </div>
      {online.length > 0 && (
        <div className="mt-4 flex items-end gap-2">
          <Field label="New group">
            <input className="input" value={groupName} onChange={(e) => setGroupName(e.target.value)} placeholder="e.g. Payments team" data-testid="team-group-name" />
          </Field>
          {online.length > 1 && (
            <select className="input shrink-0" style={{ width: 200 }} value={connId} onChange={(e) => setTarget(e.target.value)}>
              {online.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.hubName}
                </option>
              ))}
            </select>
          )}
          <button
            className="btn btn-primary btn-sm mb-[1px]"
            disabled={busy || !groupName.trim()}
            data-testid="team-create-group"
            onClick={() =>
              void run(async () => {
                const g = await hub<{ slug: string }>(connId, "POST", "/api/groups", { name: groupName.trim() });
                setGroupName("");
                useTeam.setState({ tick: useTeam.getState().tick + 1 });
                go({ kind: "group", connId, group: g.slug });
              }, "Group created")
            }
          >
            <IPlus size={13} /> Create
          </button>
        </div>
      )}
    </Card>
  );
}

// ---- inbox --------------------------------------------------------------------------------------

interface Notice {
  id: string;
  kind: string;
  title: string;
  body: string;
  link: string;
  read: boolean;
  createdAt: number;
}

/** "group/project#12", "group/project!3", "group/project@branch" -> a view. */
export function linkView(connId: string, link: string) {
  const m = /^([^/]+)\/([^#!@]+)(?:([#!@])(.+))?$/.exec(link);
  if (!m) return null;
  const [, group, project, sep, rest] = m;
  if (sep === "#") return { kind: "project" as const, connId, group: group!, project: project!, tab: "issues" as const, item: Number(rest) };
  if (sep === "!") return { kind: "project" as const, connId, group: group!, project: project!, tab: "pulls" as const, item: Number(rest) };
  return { kind: "project" as const, connId, group: group!, project: project!, tab: "code" as const };
}

function Inbox({ state }: { state: TeamStateView }) {
  const { go, tick, refreshUnread } = useTeam();
  const [items, setItems] = useState<Array<Notice & { connId: string; hubName: string }>>([]);
  const load = async () => {
    const all: Array<Notice & { connId: string; hubName: string }> = [];
    for (const c of state.connections.filter((x) => x.status === "online"))
      for (const n of await hub<Notice[]>(c.id, "GET", "/api/notifications").catch(() => [] as Notice[])) all.push({ ...n, connId: c.id, hubName: c.hubName });
    setItems(all.sort((a, b) => b.createdAt - a.createdAt));
  };
  useEffect(() => {
    void load();
  }, [tick, state.connections.map((c) => c.status).join()]);
  const open = async (n: Notice & { connId: string }) => {
    if (!n.read) await hub(n.connId, "POST", "/api/notifications/read", { ids: [n.id] }).catch(() => undefined);
    void refreshUnread();
    const v = linkView(n.connId, n.link);
    if (v) go(n.kind.startsWith("bug") ? { ...v, tab: v.tab === "issues" ? "bugs" : v.tab } : v);
  };
  const readAll = async () => {
    for (const c of new Set(items.map((i) => i.connId))) await hub(c, "POST", "/api/notifications/read", {}).catch(() => undefined);
    await load();
    void refreshUnread();
  };
  return (
    <div className="mx-auto max-w-[860px] px-10 py-8">
      <div className="mb-4 flex items-end">
        <h1 className="heading m-0 flex-1 text-[34px]">Inbox</h1>
        <button className="btn btn-sm" onClick={() => void readAll()}>
          Mark all read
        </button>
      </div>
      <div className="flex flex-col gap-2" data-testid="team-inbox-list">
        {items.map((n) => (
          <button key={n.id} className="card flex items-start gap-3 px-4 py-3 text-left" onClick={() => void open(n)}>
            <span className="mt-1.5 h-2 w-2 shrink-0 rounded-full" style={{ background: n.read ? "transparent" : "var(--text)" }} />
            <div className="min-w-0 flex-1">
              <div className="flex gap-2 text-[13px]">
                <span className="font-semibold">{n.title}</span>
                <span className="smallcaps ml-auto shrink-0">{n.kind.replace(/_/g, " ")}</span>
              </div>
              <div className="mt-0.5 text-[12.5px] text-dim">{n.body}</div>
              <div className="mt-1 text-[11px] text-faint">
                {n.hubName} · {n.link} · {ago(n.createdAt)}
              </div>
            </div>
          </button>
        ))}
        {!items.length && <div className="py-10 text-center text-dim">Nothing here. Bug reports routed to you, reviews and merges show up here.</div>}
      </div>
    </div>
  );
}

// ---- group --------------------------------------------------------------------------------------

interface GroupDetail {
  id: string;
  slug: string;
  name: string;
  description: string;
  role: string;
  members: Array<{ id: string; name: string; email: string; role: string; joinedAt: number }>;
  projects: Array<{ slug: string; name: string; description: string; openIssues: number; openBugs: number; openPulls: number; defaultBranch: string }>;
  invites: Array<{ code: string; role: string; usesLeft: number; expiresAt: number }>;
}

function GroupPage({ connId, slug }: { connId: string; slug: string }) {
  const { go, tick, state } = useTeam();
  const { workspace, toast } = useApp();
  const [g, setG] = useState<GroupDetail | null>(null);
  const [tab, setTab] = useState<"projects" | "members" | "invites">("projects");
  const [newProject, setNewProject] = useState("");
  const [inviteRole, setInviteRole] = useState("developer");
  const [shared, setShared] = useState<{ group: string; project: string } | null>(null);
  const { busy, run } = useRun();
  const load = () => hub<GroupDetail>(connId, "GET", `/api/groups/${slug}`).then(setG, (e) => toast(errText(e), "error"));
  useEffect(() => {
    void load();
    void call<{ group: string; project: string } | null>("team:workspaceProject", {}).then(setShared, () => undefined);
  }, [connId, slug, tick]);
  if (!g) return null;
  const canManage = atLeast(g.role, "maintainer");
  const conn = state?.connections.find((c) => c.id === connId);
  const hosting = state?.hosting;
  const joinHint = (code: string) => {
    const lines = [`Join "${g.name}" on ${conn?.hubName ?? "our team hub"} from the RLM IDE (Team → Join a hub).`, `Invite code: ${code}`];
    if (connId === "local" && hosting) {
      for (const a of hosting.addresses) lines.push(`LAN: http://${a}:${hosting.httpPort}`);
      if (hosting.sshPort) for (const a of hosting.addresses) lines.push(`SSH: ${a} port ${hosting.sshPort} (host key ${hosting.sshFingerprint})`);
    } else if (conn) lines.push(conn.transport.kind === "http" ? `Address: ${conn.transport.url}` : conn.transport.kind === "ssh" ? `SSH: ${conn.transport.host} port ${conn.transport.port}` : `SSH jump: ${conn.transport.username}@${conn.transport.host}`);
    return lines.join("\n");
  };

  return (
    <div className="mx-auto max-w-[900px] px-10 py-8">
      <div className="smallcaps">Group · you are {g.role}</div>
      <h1 className="heading m-0 mt-1 text-[38px] leading-none">{g.name}</h1>
      <div className="seg mt-5">
        {(["projects", "members", "invites"] as const).map((t) => (
          <button key={t} className={tab === t ? "on" : ""} onClick={() => setTab(t)} data-testid={`group-tab-${t}`}>
            {t[0]!.toUpperCase() + t.slice(1)} {t === "members" ? `(${g.members.length})` : t === "projects" ? `(${g.projects.length})` : ""}
          </button>
        ))}
      </div>

      {tab === "projects" && (
        <div className="mt-5">
          <div className="grid grid-cols-2 gap-3">
            {g.projects.map((p) => (
              <button key={p.slug} className="card p-4 text-left" onClick={() => go({ kind: "project", connId, group: slug, project: p.slug })} data-testid="group-project">
                <div className="text-[15px] font-semibold">{p.name}</div>
                <div className="mt-0.5 line-clamp-2 text-[12px] text-dim">{p.description || `${slug}/${p.slug}`}</div>
                <div className="mt-3 flex gap-4 text-[11.5px] text-dim">
                  <span>{p.openIssues} issues</span>
                  <span style={{ color: p.openBugs ? "var(--err)" : undefined }}>{p.openBugs} bugs</span>
                  <span>{p.openPulls} pull requests</span>
                </div>
              </button>
            ))}
          </div>
          {atLeast(g.role, "developer") && (
            <div className="card mt-4 p-4">
              <div className="flex items-end gap-2">
                <Field label="New empty project">
                  <input className="input" value={newProject} onChange={(e) => setNewProject(e.target.value)} placeholder="e.g. checkout-service" data-testid="group-new-project" />
                </Field>
                <button
                  className="btn btn-sm mb-[1px]"
                  disabled={busy || !newProject.trim()}
                  onClick={() =>
                    void run(async () => {
                      const p = await hub<{ slug: string }>(connId, "POST", `/api/groups/${slug}/projects`, { name: newProject.trim() });
                      setNewProject("");
                      go({ kind: "project", connId, group: slug, project: p.slug });
                    }, "Project created")
                  }
                >
                  Create
                </button>
              </div>
              {workspace && !shared && (
                <div className="mt-4 flex items-center gap-3 border-t border-line pt-4">
                  <div className="flex-1 text-[12.5px]">
                    Add the open folder <b>{workspace.name}</b> to this group. It becomes a project here and gets pushed (a git repository is created if needed).
                  </div>
                  <button
                    className="btn btn-primary btn-sm"
                    disabled={busy}
                    data-testid="group-share-workspace"
                    onClick={() =>
                      void run(async () => {
                        const r = await call<{ group: string; project: string }>("team:share", { connId, group: slug, name: workspace.name });
                        await useApp.getState().refreshGit();
                        go({ kind: "project", connId, group: r.group, project: r.project });
                      }, `Shared ${workspace.name}`)
                    }
                  >
                    {busy ? "Pushing…" : "Share open folder"}
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {tab === "members" && (
        <div className="mt-5 flex flex-col gap-2">
          {g.members.map((m) => (
            <div key={m.id} className="card flex items-center gap-3 px-4 py-3 text-[13px]">
              <div className="flex h-8 w-8 items-center justify-center rounded-full border border-ink text-[12px] font-semibold">{m.name.slice(0, 1).toUpperCase()}</div>
              <div className="min-w-0 flex-1">
                <div className="font-semibold">{m.name}</div>
                <div className="text-[11.5px] text-faint">
                  {m.email} · joined {ago(m.joinedAt)}
                </div>
              </div>
              {canManage && m.role !== "owner" ? (
                <>
                  <select className="input shrink-0" style={{ width: 130 }} value={m.role} onChange={(e) => void run(() => hub(connId, "PATCH", `/api/groups/${slug}/members/${m.id}`, { role: e.target.value }).then(load))}>
                    {ROLES.filter((r) => r !== "owner").map((r) => (
                      <option key={r}>{r}</option>
                    ))}
                  </select>
                  <button className="iconbtn" title="Remove from group" onClick={() => confirm(`Remove ${m.name} from ${g.name}?`) && void run(() => hub(connId, "DELETE", `/api/groups/${slug}/members/${m.id}`).then(load))}>
                    <ITrash size={13} />
                  </button>
                </>
              ) : (
                <span className="smallcaps">{m.role}</span>
              )}
            </div>
          ))}
        </div>
      )}

      {tab === "invites" && (
        <div className="mt-5">
          {canManage ? (
            <>
              <div className="flex items-end gap-2">
                <Field label="Role for new members">
                  <select className="input shrink-0" style={{ width: 180 }} value={inviteRole} onChange={(e) => setInviteRole(e.target.value)}>
                    {ROLES.filter((r) => r !== "owner" && atLeast(g.role, r)).map((r) => (
                      <option key={r}>{r}</option>
                    ))}
                  </select>
                </Field>
                <button className="btn btn-primary btn-sm mb-[1px]" disabled={busy} data-testid="group-create-invite" onClick={() => void run(() => hub(connId, "POST", `/api/groups/${slug}/invites`, { role: inviteRole }).then(load), "Invite created")}>
                  <IPlus size={13} /> New invite code
                </button>
              </div>
              <div className="mt-4 flex flex-col gap-2">
                {g.invites.map((i) => (
                  <div key={i.code} className="card px-4 py-3">
                    <div className="flex items-center gap-3">
                      <span className="mono selectable text-[16px] font-semibold tracking-wider" data-testid="group-invite-code">
                        {i.code}
                      </span>
                      <span className="smallcaps">{i.role}</span>
                      <span className="text-[11.5px] text-faint">
                        {i.usesLeft} uses left · expires {new Date(i.expiresAt).toLocaleString()}
                      </span>
                      <div className="flex-1" />
                      <button className="btn btn-sm" onClick={() => copy(joinHint(i.code))}>
                        <ICopy size={12} /> Copy instructions
                      </button>
                      <button className="iconbtn" title="Revoke" onClick={() => void run(() => hub(connId, "DELETE", `/api/groups/${slug}/invites/${i.code}`).then(load))}>
                        <ITrash size={13} />
                      </button>
                    </div>
                  </div>
                ))}
                {!g.invites.length && <div className="text-[12.5px] text-dim">No active invites.</div>}
              </div>
            </>
          ) : (
            <div className="text-[12.5px] text-dim">Only maintainers and owners can invite people.</div>
          )}
        </div>
      )}
    </div>
  );
}
