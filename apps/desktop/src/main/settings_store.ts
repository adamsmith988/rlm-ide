import { safeStorage } from "electron";
import { existsSync, readFileSync, statSync, unwatchFile, watchFile } from "node:fs";
import { Settings, defaultSettings, migrateSettings, repairRoles } from "@rlm/contracts";
import { writeFileAtomic } from "./instance";

function deepMerge<T>(base: T, patch: unknown): T {
  if (patch === null || typeof patch !== "object" || Array.isArray(patch)) return patch as T;
  const out: Record<string, unknown> = { ...(base as Record<string, unknown>) };
  for (const [k, v] of Object.entries(patch as Record<string, unknown>)) {
    const cur = out[k];
    out[k] = cur && typeof cur === "object" && !Array.isArray(cur) && v && typeof v === "object" && !Array.isArray(v) ? deepMerge(cur, v) : v;
  }
  return out as T;
}

function readText(file: string): string | null {
  try {
    return readFileSync(file, "utf8");
  } catch {
    return null;
  }
}

/** settings.json in userData. `RLM_SETTINGS_JSON` (a JSON patch) overrides values for tests/evals.
 * Other copies of the IDE share the file: changes they write are picked up, and an update starts from the file. */
export class SettingsStore {
  private value: Settings;
  private disk: string | null = null;
  private readonly listeners = new Set<(s: Settings) => void>();

  constructor(private readonly file: string) {
    this.value = this.load();
  }

  private load(): Settings {
    this.disk = readText(this.file);
    let raw: unknown = {};
    if (this.disk !== null) {
      try {
        raw = JSON.parse(this.disk);
      } catch {
        raw = {};
      }
    }
    raw = migrateSettings(raw);
    if (process.env.RLM_SETTINGS_JSON) raw = deepMerge(raw, JSON.parse(process.env.RLM_SETTINGS_JSON));
    const parsed = Settings.safeParse(raw);
    return repairRoles(parsed.success ? parsed.data : defaultSettings());
  }

  /** True when another copy of the IDE wrote the file since this one last read or wrote it. */
  private reloadIfChanged(): boolean {
    const now = readText(this.file);
    if (now === null || now === this.disk) return false;
    try {
      JSON.parse(now);
    } catch {
      return false;
    }
    this.value = this.load();
    return true;
  }

  get(): Settings {
    return this.value;
  }

  update(patch: Record<string, unknown>): Settings {
    this.reloadIfChanged();
    const merged = deepMerge(this.value, patch);
    this.value = repairRoles(Settings.parse(merged));
    const text = JSON.stringify(this.value, null, 2);
    writeFileAtomic(this.file, text);
    this.disk = text;
    for (const l of this.listeners) l(this.value);
    return this.value;
  }

  /** Follows changes other copies make (polling: file watching is unreliable across Windows processes). */
  watch(intervalMs = 1000): () => void {
    const onChange = () => {
      if (this.reloadIfChanged()) for (const l of this.listeners) l(this.value);
    };
    watchFile(this.file, { interval: intervalMs }, onChange);
    return () => unwatchFile(this.file, onChange);
  }

  onChange(cb: (s: Settings) => void): () => void {
    this.listeners.add(cb);
    return () => this.listeners.delete(cb);
  }
}

/** API keys encrypted with Electron safeStorage. Never written to settings.json, never logged. */
export class SecretStore {
  private data: Record<string, string> = {};
  private stamp = "";

  constructor(private readonly file: string) {
    this.refresh();
  }

  /** Another copy of the IDE may have saved a key since this one read the file. */
  private refresh(): void {
    let st = "";
    try {
      const s = statSync(this.file);
      st = `${s.mtimeMs}:${s.size}`;
    } catch {
      st = "";
    }
    if (st === this.stamp) return;
    this.stamp = st;
    if (!st || !existsSync(this.file)) {
      this.data = {};
      return;
    }
    try {
      this.data = JSON.parse(readFileSync(this.file, "utf8")) as Record<string, string>;
    } catch {
      /* a write in progress; keep what we had */
    }
  }

  backend(): string {
    if (!safeStorage.isEncryptionAvailable()) return "unavailable";
    const b = (safeStorage as unknown as { getSelectedStorageBackend?: () => string }).getSelectedStorageBackend?.();
    return b ?? "os";
  }

  set(name: string, value: string): void {
    this.refresh();
    if (!value) delete this.data[name];
    else {
      if (!safeStorage.isEncryptionAvailable()) throw new Error("Secure storage is unavailable on this system; the key was not saved.");
      this.data[name] = safeStorage.encryptString(value).toString("base64");
    }
    writeFileAtomic(this.file, JSON.stringify(this.data), 0o600);
    this.stamp = "";
    this.refresh();
  }

  get(name: string): string | null {
    this.refresh();
    const enc = this.data[name];
    if (!enc) return null;
    try {
      return safeStorage.decryptString(Buffer.from(enc, "base64"));
    } catch {
      return null;
    }
  }

  /** Deletes provider keys whose provider no longer exists. Returns how many were removed. */
  pruneProviders(liveIds: string[]): number {
    this.refresh();
    const live = new Set(liveIds.map((id) => `provider:${id}`));
    const dead = Object.keys(this.data).filter((k) => k.startsWith("provider:") && !live.has(k));
    for (const k of dead) this.set(k, "");
    return dead.length;
  }

  /** Model provider keys only; other secrets (team tokens) never leave the main process. */
  all(): Record<string, string> {
    this.refresh();
    const out: Record<string, string> = {};
    for (const k of Object.keys(this.data)) {
      if (!k.startsWith("provider:")) continue;
      const v = this.get(k);
      if (v) out[k] = v;
    }
    if (process.env.RLM_API_KEY && process.env.RLM_API_PROVIDER) out[`provider:${process.env.RLM_API_PROVIDER}`] = process.env.RLM_API_KEY;
    return out;
  }
}
