import { describe, expect, it } from "vitest";
import { windowsShell } from "../src/main/pty_service";

describe("windowsShell", () => {
  const env = { SystemRoot: "C:\\Windows", PATH: "C:\\Tools;C:\\Program Files\\PowerShell\\7", ComSpec: "C:\\Windows\\system32\\cmd.exe" };

  it("uses Windows PowerShell by its full path, so a trimmed PATH does not matter", () => {
    const have = new Set(["C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe", "C:\\Windows\\system32\\cmd.exe"]);
    expect(windowsShell({ ...env, PATH: "" }, (p) => have.has(p))).toBe("C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe");
  });

  it("falls back to PowerShell 7 on PATH, then the command processor", () => {
    expect(windowsShell(env, (p) => p === "C:\\Program Files\\PowerShell\\7\\pwsh.exe" || p === env.ComSpec)).toBe("C:\\Program Files\\PowerShell\\7\\pwsh.exe");
    expect(windowsShell(env, (p) => p === env.ComSpec)).toBe(env.ComSpec);
    expect(windowsShell({}, () => false)).toBe("cmd.exe");
  });
});
