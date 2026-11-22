import { promises as fs } from "node:fs";
import path from "node:path";
import type { SkillConfig } from "@rlm/contracts";
import { relPath, resolveInside } from "./fs/confine";

export type SkillSource = "user" | "project";

export interface LoadedSkill {
  name: string;
  description: string;
  body: string;
  source: SkillSource;
  enabled: boolean;
  /** Absolute skill folder for project skills; null for settings-only skills. */
  dir: string | null;
  files: string[];
}

const NAME = /^[\w.-]+$/;
const BODY_MAX = 20_000;
const FILE_MAX = 20_000;

/** Reads YAML-ish front matter from a SKILL.md. */
export function parseSkillMarkdown(raw: string, fallbackName: string): { name: string; description: string; body: string; enabled: boolean } {
  const trimmed = raw.replace(/^\uFEFF/, "");
  const m = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?/.exec(trimmed);
  const meta: Record<string, string> = {};
  let body = trimmed;
  if (m) {
    body = trimmed.slice(m[0].length);
    for (const line of m[1]!.split(/\r?\n/)) {
      const i = line.indexOf(":");
      if (i <= 0) continue;
      const key = line.slice(0, i).trim().toLowerCase();
      let val = line.slice(i + 1).trim();
      if ((val.startsWith('"') && val.endsWith('"')) || (val.startsWith("'") && val.endsWith("'"))) val = val.slice(1, -1);
      meta[key] = val;
    }
  }
  const name = sanitizeName(meta.name || fallbackName) || sanitizeName(fallbackName) || "skill";
  const enabled = meta.enabled === undefined ? true : !/^(false|0|no|off)$/i.test(meta.enabled);
  return { name, description: (meta.description ?? "").slice(0, 400), body: body.trim(), enabled };
}

export function sanitizeName(name: string): string {
  return name.trim().replace(/[^\w.-]/g, "").slice(0, 40);
}

export function userSkills(list: SkillConfig[]): LoadedSkill[] {
  return list
    .filter((s) => s.enabled && NAME.test(s.name) && s.body.trim())
    .map((s) => ({
      name: s.name,
      description: s.description,
      body: s.body.trim(),
      source: "user" as const,
      enabled: true,
      dir: null,
      files: [],
    }));
}

/** Project skills live in `.rlm/skills/<name>/SKILL.md`, with optional extra files and scripts. */
export async function loadProjectSkills(root: string): Promise<LoadedSkill[]> {
  const base = path.join(root, ".rlm", "skills");
  let dirs: string[] = [];
  try {
    dirs = (await fs.readdir(base, { withFileTypes: true })).filter((e) => e.isDirectory()).map((e) => e.name);
  } catch {
    return [];
  }
  const out: LoadedSkill[] = [];
  for (const folder of dirs) {
    const dir = path.join(base, folder);
    let raw = "";
    try {
      raw = await fs.readFile(path.join(dir, "SKILL.md"), "utf8");
    } catch {
      continue;
    }
    const parsed = parseSkillMarkdown(raw, folder);
    if (!parsed.enabled || !parsed.body) continue;
    out.push({
      ...parsed,
      body: parsed.body.slice(0, BODY_MAX),
      source: "project",
      dir,
      files: await listExtraFiles(dir),
    });
  }
  return out;
}

/** Project skills win when a user skill reuses the same name. */
export function mergeSkills(user: LoadedSkill[], project: LoadedSkill[]): LoadedSkill[] {
  const byName = new Map<string, LoadedSkill>();
  for (const s of user) byName.set(s.name, s);
  for (const s of project) byName.set(s.name, s);
  return [...byName.values()].sort((a, b) => a.name.localeCompare(b.name));
}

export function catalogBlock(skills: LoadedSkill[]): string {
  if (!skills.length) return "";
  const lines = skills.map((s) => `- /${s.name} (${s.source})${s.description ? ` — ${s.description}` : ""}${s.files.length ? `; extra files: ${s.files.join(", ")}` : ""}`);
  return [
    "When a request matches a skill, call skills.read(name=\"...\") and follow every step. Load extra files with skills.read_file(name=..., path=...). Run scripts the skill names with shell.run from the project root. Do not skip steps or invent a shorter procedure.",
    "",
    ...lines,
  ].join("\n");
}

export async function readSkillFile(skill: LoadedSkill, rel: string): Promise<string> {
  if (!skill.dir) throw new Error(`/${skill.name} has no extra files`);
  const abs = resolveInside(skill.dir, rel);
  const st = await fs.stat(abs);
  if (!st.isFile()) throw new Error(`${rel} is not a file`);
  if (st.size > FILE_MAX) throw new Error(`${rel} is larger than ${FILE_MAX} characters; page it with fs.read_file from the project path`);
  return (await fs.readFile(abs, "utf8")).slice(0, FILE_MAX);
}

async function listExtraFiles(dir: string): Promise<string[]> {
  const out: string[] = [];
  const stack = [dir];
  while (stack.length && out.length < 40) {
    const cur = stack.pop()!;
    let entries: import("node:fs").Dirent[] = [];
    try {
      entries = await fs.readdir(cur, { withFileTypes: true });
    } catch {
      continue;
    }
    for (const e of entries) {
      const abs = path.join(cur, e.name);
      if (e.isDirectory()) {
        if (e.name.startsWith(".")) continue;
        stack.push(abs);
        continue;
      }
      if (e.name === "SKILL.md") continue;
      out.push(relPath(dir, abs));
    }
  }
  return out.sort();
}
