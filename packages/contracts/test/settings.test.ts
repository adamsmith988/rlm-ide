import { describe, expect, it } from "vitest";
import { defaultSettings, migrateSettings, Settings, SETTINGS_VERSION } from "../src";

describe("settings migration", () => {
  it("moves files from before per-project memory to it, once", () => {
    const old = { memory: { longTermScope: "user", minScore: 0.45, sessionLimit: 3 } };
    const s = Settings.parse(migrateSettings(old));
    expect(s.settingsVersion).toBe(SETTINGS_VERSION);
    expect(s.memory).toMatchObject({ longTermScope: "workspace", minScore: 0.55, sessionLimit: 3 });
    // A user who picks "all projects" again keeps it.
    const later = { ...s, memory: { ...s.memory, longTermScope: "user" } };
    expect(Settings.parse(migrateSettings(later)).memory.longTermScope).toBe("user");
  });

  it("leaves a tuned recall threshold alone and starts new installs on the current defaults", () => {
    expect(Settings.parse(migrateSettings({ memory: { minScore: 0.7 } })).memory.minScore).toBe(0.7);
    expect(Settings.parse(migrateSettings({}))).toEqual(defaultSettings());
  });
});
