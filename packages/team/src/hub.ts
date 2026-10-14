import path from "node:path";
import { CliGitHost } from "./adapters/git_host";
import { BeaconSender, DISCOVERY_PORT, lanAddresses } from "./adapters/discovery";
import { HttpHub } from "./adapters/http_server";
import { SqliteTeamStore } from "./adapters/sqlite_store";
import { SshHub } from "./adapters/ssh_server";
import { TeamService } from "./service";

export interface HubConfig {
  dataDir: string;
  name: string;
  owner: { name: string; email: string };
  /** "0.0.0.0" to serve the LAN, "127.0.0.1" for this machine only (remote people then come in over SSH). */
  bindHost: string;
  httpPort: number;
  /** null disables the built-in SSH server. */
  sshPort: number | null;
  discovery: boolean;
  discoveryPort?: number;
  log?: (msg: string) => void;
}

export interface RunningHub {
  service: TeamService;
  hubId: string;
  name: string;
  httpPort: number;
  sshPort: number | null;
  sshFingerprint: string | null;
  ownerToken: string;
  ownerId: string;
  addresses: string[];
  close(): Promise<void>;
}

/** Starts a hub: database, repositories, HTTP API + git, SSH server and LAN beacon. */
export async function startHub(cfg: HubConfig): Promise<RunningHub> {
  const store = new SqliteTeamStore(path.join(cfg.dataDir, "hub.db"));
  const git = new CliGitHost(path.join(cfg.dataDir, "repos"));
  const service = new TeamService(store, git);
  const boot = service.bootstrap(cfg.name, cfg.owner);

  let ssh: SshHub | null = null;
  let sshPort: number | null = null;
  let httpPort = 0;
  const http = new HttpHub(service, {
    host: cfg.bindHost,
    port: cfg.httpPort,
    reposRoot: path.join(cfg.dataDir, "repos"),
    git,
    log: cfg.log,
    publicInfo: () => ({ httpPort, sshPort, sshFingerprint: ssh?.fingerprint ?? null }),
  });
  try {
    httpPort = (await http.listen()).port;
    if (cfg.sshPort !== null) {
      ssh = new SshHub(service, {
        host: cfg.bindHost,
        port: cfg.sshPort,
        hostKeyFile: path.join(cfg.dataDir, "ssh_host_ed25519"),
        httpPort: () => httpPort,
        git,
        isValidInvite: (c) => service.inviteValid(c),
        log: cfg.log,
      });
      sshPort = (await ssh.listen()).port;
    }
  } catch (e) {
    await http.close().catch(() => undefined);
    await ssh?.close().catch(() => undefined);
    store.close();
    throw e;
  }

  const beacon = cfg.discovery && cfg.bindHost !== "127.0.0.1"
    ? new BeaconSender(() => ({ rlmHub: 1, hubId: boot.hubId, name: service.info().name, httpPort, sshPort, sshFingerprint: ssh?.fingerprint ?? null }), cfg.discoveryPort ?? DISCOVERY_PORT)
    : null;
  beacon?.start();

  return {
    service,
    hubId: boot.hubId,
    name: cfg.name,
    httpPort,
    sshPort,
    sshFingerprint: ssh?.fingerprint ?? null,
    ownerToken: boot.token,
    ownerId: boot.user.id,
    addresses: cfg.bindHost === "127.0.0.1" ? ["127.0.0.1"] : lanAddresses(),
    async close() {
      beacon?.stop();
      await http.close();
      await ssh?.close();
      store.close();
    },
  };
}
