import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import { catalogBlock, loadProjectSkills, mergeSkills, parseSkillMarkdown, readSkillFile, userSkills } from "../src/skills";

describe("skill files", () => {
  const root = mkdtempSync(path.join(tmpdir(), "rlm-skills-"));
  const dir = path.join(root, ".rlm", "skills", "receipt");
  mkdirSync(path.join(dir, "scripts"), { recursive: true });
  writeFileSync(
    path.join(dir, "SKILL.md"),
    `---
name: receipt
description: Write a dated receipt file
---
# Receipt

1. Read format.md
2. Run scripts/stamp.py
3. Write receipts/today.txt
`,
  );
  writeFileSync(path.join(dir, "format.md"), "SKILL_RECEIPT_OK\n");
  writeFileSync(path.join(dir, "scripts", "stamp.py"), "print('stamp')\n");

  it("reads front matter, extra files, and merged catalogs", async () => {
    const parsed = parseSkillMarkdown("---\nname: house-style\ndescription: Notes\nenabled: true\n---\nUse ATX headings.\n", "x");
    expect(parsed).toMatchObject({ name: "house-style", description: "Notes", body: "Use ATX headings." });
    const project = await loadProjectSkills(root);
    expect(project).toHaveLength(1);
    expect(project[0]).toMatchObject({ name: "receipt", source: "project", files: ["format.md", "scripts/stamp.py"] });
    expect(await readSkillFile(project[0]!, "format.md")).toContain("SKILL_RECEIPT_OK");
    const user = userSkills([{ id: "1", name: "receipt", description: "old", body: "user copy", enabled: true }]);
    const merged = mergeSkills(user, project);
    expect(merged).toHaveLength(1);
    expect(merged[0]?.source).toBe("project");
    expect(catalogBlock(merged)).toContain("/receipt (project)");
    expect(catalogBlock(merged)).toContain("format.md");
  });

  afterAll(() => rmSync(root, { recursive: true, force: true }));
});
