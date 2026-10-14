import { pythonSignature, type ToolDescriptor } from "../domain/tools";

/** Renders the prompt section documenting every kernel callable. Prompts reference tools only
 * through this section, so adding a tool never requires editing a prompt. */
export function renderToolSurface(tools: ToolDescriptor[]): string {
  const byNs = new Map<string, ToolDescriptor[]>();
  for (const t of tools) {
    const ns = t.id.split(".")[0] ?? t.id;
    byNs.set(ns, [...(byNs.get(ns) ?? []), t]);
  }
  const lines: string[] = [];
  for (const [ns, list] of [...byNs.entries()].sort()) {
    lines.push(`### ${ns}`);
    for (const t of list.sort((a, b) => a.id.localeCompare(b.id))) {
      const flag = t.sideEffects ? " [side effects: needs user approval]" : "";
      lines.push(`- \`${pythonSignature(t)}\` -> ${t.returns}${flag}`);
      lines.push(`  ${t.description}`);
      for (const [name, p] of Object.entries(t.parameters.properties ?? {})) {
        if (p.description) lines.push(`  - ${name}: ${p.description}`);
      }
    }
    lines.push("");
  }
  return lines.join("\n").trim();
}
