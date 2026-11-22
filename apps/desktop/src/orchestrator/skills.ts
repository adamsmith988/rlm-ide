import type { SkillConfig } from "@rlm/contracts";
import { catalogBlock, loadProjectSkills, mergeSkills, readSkillFile, userSkills, type LoadedSkill } from "@rlm/adapters";

/** Merges Settings skills with `.rlm/skills` in the open project. */
export class SkillHub {
  private items: LoadedSkill[] = [];

  async sync(configured: SkillConfig[], workspaceRoot: string | null): Promise<void> {
    const user = userSkills(configured);
    const project = workspaceRoot ? await loadProjectSkills(workspaceRoot) : [];
    this.items = mergeSkills(user, project);
  }

  list(): Array<{ name: string; description: string; source: LoadedSkill["source"]; files: string[] }> {
    return this.items.map((s) => ({ name: s.name, description: s.description, source: s.source, files: s.files }));
  }

  catalog(): string {
    return catalogBlock(this.items);
  }

  read(name: string): { name: string; description: string; source: LoadedSkill["source"]; body: string; files: string[] } {
    const s = this.find(name);
    return { name: s.name, description: s.description, source: s.source, body: s.body, files: s.files };
  }

  readFile(name: string, rel: string): Promise<string> {
    return readSkillFile(this.find(name), rel);
  }

  private find(name: string): LoadedSkill {
    const s = this.items.find((x) => x.name === name);
    if (!s) throw new Error(`no skill "/${name}". Add one in Settings → Skills & MCP or as .rlm/skills/${name}/SKILL.md.`);
    return s;
  }
}
