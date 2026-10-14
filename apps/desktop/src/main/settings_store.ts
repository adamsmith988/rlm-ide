import { safeStorage } from "electron";
import { existsSync, mkdirSync, readFileSync, writeFileSync, renameSync } from "node:fs";
import path from "node:path";
import { Settings, defaultSettings, repairRoles } from "@rlm/contracts";

function deepMerge<T>(base: T, patch: unknown): T {
  if (patch === null || typeof patch !== "object" || Array.isArray(patch)) return patch as T;
  const out: Record<string, unknown> = { ...(base as Record<string, unknown>) };
  for (const [k, v] of Object.entries(patch as Record<string, unknown>)) {
    const cur = out[k];
    out[k] = cur && typeof cur === "object" && !Array.isArray(cur) && v && typeof v === "object" && !Array.isArray(v) ? deepMerge(cur, v) : v;
  }
  return out as T;
}

/** settings.json in userData. `RLM_SETTINGS_JSON` (a JSON patch) overrides values for tests/evals. */
export class SettingsStore {
  private value: Settings;
  private readonly listeners = new Set<(s: Settings) => void>();

  constructor(private readonly file: string) {
    let raw: unknown = {};
    if (existsSync(file)) {
      try {
        raw = JSON.parse(readFileSync(file, "utf8"));
      } catch {
        raw = {};
      }
    }
    if (process.env.RLM_SETTINGS_JSON) raw = deepMerge(raw, JSON.parse(process.env.RLM_SETTINGS_JSON));
    const parsed = Settings.safeParse(raw);
    this.value = repairRoles(parsed.success ? parsed.data : defaultSettings());
  }

  get(): Settings {
    return this.value;
  }

  update(patch: Record<string, unknown>): Settings {
    const merged = deepMerge(this.value, patch);
    this.value = repairRoles(Settings.parse(merged));
    mkdirSync(path.dirname(this.file), { recursive: true });
    const tmp = `${this.file}.tmp`;
    writeFileSync(tmp, JSON.stringify(this.value, null, 2));
    renameSync(tmp, this.file);
    for (const l of this.listeners) l(this.value);
    return this.value;
  }

  onChange(cb: (s: Settings) => void): () => void {
    this.listeners.add(cb);
    return () => this.listeners.delete(cb);
  }
}

/** API keys encrypted with Electron safeStorage. Never written to settings.json, never logged. */
export class SecretStore {
  private data: Record<string, string> = {};
  constructor(private readonly file: string) {
    if (existsSync(file)) {
      try {
        this.data = JSON.parse(readFileSync(file, "utf8")) as Record<string, string>;
      } catch {
        this.data = {};
      }
    }
  }

  backend(): string {
    if (!safeStorage.isEncryptionAvailable()) return "unavailable";
    const b = (safeStorage as unknown as { getSelectedStorageBackend?: () => string }).getSelectedStorageBackend?.();
    return b ?? "os";
  }

  set(name: string, value: string): void {
    if (!value) delete this.data[name];
    else {
      if (!safeStorage.isEncryptionAvailable()) throw new Error("Secure storage is unavailable on this system; the key was not saved.");
      this.data[name] = safeStorage.encryptString(value).toString("base64");
    }
    mkdirSync(path.dirname(this.file), { recursive: true });
    writeFileSync(this.file, JSON.stringify(this.data), { mode: 0o600 });
  }

  get(name: string): string | null {
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
    const live = new Set(liveIds.map((id) => `provider:${id}`));
    const dead = Object.keys(this.data).filter((k) => k.startsWith("provider:") && !live.has(k));
    for (const k of dead) this.set(k, "");
    return dead.length;
  }

  /** Model provider keys only; other secrets (team tokens) never leave the main process. */
  all(): Record<string, string> {
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
