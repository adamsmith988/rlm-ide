import { DatabaseSync } from "node:sqlite";
import { mkdirSync } from "node:fs";
import path from "node:path";
import m001 from "./migrations/001_init.sql?raw";

const MIGRATIONS: Array<{ version: number; sql: string }> = [{ version: 1, sql: m001 }];

export interface Db {
  raw: DatabaseSync;
  vecLoaded: boolean;
}

/** Opens the app database, runs migrations and tries to load sqlite-vec. When the extension is
 * unavailable, vector math falls back to JavaScript (same results, slower on huge stores). */
export function openDb(file: string, vecExtensionPath: string | null): Db {
  if (file !== ":memory:") mkdirSync(path.dirname(file), { recursive: true });
  const raw = new DatabaseSync(file, { allowExtension: true } as ConstructorParameters<typeof DatabaseSync>[1]);
  raw.exec("PRAGMA journal_mode = WAL; PRAGMA foreign_keys = ON; PRAGMA busy_timeout = 5000;");
  let vecLoaded = false;
  if (vecExtensionPath) {
    try {
      (raw as unknown as { loadExtension(p: string): void }).loadExtension(vecExtensionPath);
      vecLoaded = true;
    } catch {
      vecLoaded = false;
    }
  }
  raw.exec("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)");
  const row = raw.prepare("SELECT MAX(version) AS v FROM schema_version").get() as { v: number | null } | undefined;
  const current = row?.v ?? 0;
  for (const m of MIGRATIONS) {
    if (m.version <= current) continue;
    raw.exec("BEGIN");
    try {
      raw.exec(m.sql);
      raw.prepare("INSERT INTO schema_version (version) VALUES (?)").run(m.version);
      raw.exec("COMMIT");
    } catch (e) {
      raw.exec("ROLLBACK");
      throw e;
    }
  }
  return { raw, vecLoaded };
}
