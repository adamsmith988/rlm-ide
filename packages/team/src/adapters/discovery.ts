import dgram from "node:dgram";
import os from "node:os";

export const DISCOVERY_PORT = 47421;

export interface Beacon {
  rlmHub: 1;
  hubId: string;
  name: string;
  httpPort: number;
  sshPort: number | null;
  sshFingerprint: string | null;
}

export interface DiscoveredHub extends Beacon {
  /** Address the beacon came from, and every address the hub listed. */
  address: string;
  lastSeen: number;
}

/** IPv4 broadcast addresses of every non-internal interface. */
function broadcastAddresses(): string[] {
  const out = new Set<string>();
  for (const list of Object.values(os.networkInterfaces())) {
    for (const i of list ?? []) {
      if (i.family !== "IPv4" || i.internal) continue;
      const ip = i.address.split(".").map(Number);
      const mask = i.netmask.split(".").map(Number);
      out.add(ip.map((b, k) => (b | (~mask[k]! & 255)) & 255).join("."));
    }
  }
  out.add("255.255.255.255");
  return [...out];
}

/** Announces a hub on the LAN every few seconds. */
export class BeaconSender {
  private sock = dgram.createSocket({ type: "udp4", reuseAddr: true });
  private timer: ReturnType<typeof setInterval> | null = null;

  constructor(
    private beacon: () => Beacon,
    private port = DISCOVERY_PORT,
    private everyMs = 3000,
  ) {}

  start(): void {
    this.sock.bind(() => {
      this.sock.setBroadcast(true);
      const send = () => {
        const msg = Buffer.from(JSON.stringify(this.beacon()));
        for (const addr of [...broadcastAddresses(), "127.0.0.1"]) this.sock.send(msg, this.port, addr, () => undefined);
      };
      send();
      this.timer = setInterval(send, this.everyMs);
    });
    this.sock.on("error", () => undefined);
  }

  stop(): void {
    if (this.timer) clearInterval(this.timer);
    try {
      this.sock.close();
    } catch {
      /* already closed */
    }
  }
}

/** Listens for hub beacons; `list()` returns hubs heard within the last `ttlMs`. */
export class BeaconListener {
  private sock = dgram.createSocket({ type: "udp4", reuseAddr: true });
  private hubs = new Map<string, DiscoveredHub>();

  constructor(
    private onChange?: (hubs: DiscoveredHub[]) => void,
    private port = DISCOVERY_PORT,
    private ttlMs = 12000,
  ) {}

  start(): Promise<void> {
    return new Promise((resolve) => {
      this.sock.on("message", (msg, rinfo) => {
        try {
          const b = JSON.parse(msg.toString("utf8")) as Beacon;
          if (b.rlmHub !== 1 || !b.hubId || !Number.isInteger(b.httpPort)) return;
          const key = `${b.hubId}@${rinfo.address}`;
          const known = this.hubs.has(key);
          this.hubs.set(key, { ...b, name: String(b.name).slice(0, 80), address: rinfo.address, lastSeen: Date.now() });
          if (!known) this.onChange?.(this.list());
        } catch {
          /* not ours */
        }
      });
      this.sock.on("error", () => resolve());
      this.sock.bind(this.port, () => resolve());
    });
  }

  list(): DiscoveredHub[] {
    const cutoff = Date.now() - this.ttlMs;
    for (const [k, h] of this.hubs) if (h.lastSeen < cutoff) this.hubs.delete(k);
    // One entry per hub, preferring a non-loopback address other machines can also use.
    const byHub = new Map<string, DiscoveredHub>();
    for (const h of this.hubs.values()) {
      const cur = byHub.get(h.hubId);
      if (!cur || (cur.address.startsWith("127.") && !h.address.startsWith("127."))) byHub.set(h.hubId, h);
    }
    return [...byHub.values()].sort((a, b) => a.name.localeCompare(b.name));
  }

  stop(): void {
    try {
      this.sock.close();
    } catch {
      /* already closed */
    }
  }
}

/** Container and VM bridges that teammates on other machines cannot reach. VPN interfaces stay. */
const VIRTUAL_IF = /^(docker|br-|veth|virbr|vmnet|vboxnet|lxc|lxd|cni|flannel|kube|podman|cali|vEthernet \((WSL|Default Switch))/i;

/** Addresses other machines can use to reach this one, private LAN ranges first. */
export function lanAddresses(): string[] {
  const out: string[] = [];
  for (const [name, list] of Object.entries(os.networkInterfaces())) {
    if (VIRTUAL_IF.test(name)) continue;
    for (const i of list ?? []) if (i.family === "IPv4" && !i.internal) out.push(i.address);
  }
  const rank = (a: string) => (a.startsWith("192.168.") ? 0 : a.startsWith("10.") ? 1 : /^172\.(1[6-9]|2\d|3[01])\./.test(a) ? 2 : 3);
  return out.sort((a, b) => rank(a) - rank(b));
}
