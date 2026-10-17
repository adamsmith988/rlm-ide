import { pythonSignature, type ToolDescriptor } from "../domain/tools";

function head(t: ToolDescriptor): string {
  const flag = t.sideEffects ? " [side effects: needs user approval]" : "";
  return `- \`${pythonSignature(t)}\` -> ${t.returns}${flag}`;
}

/** Full documentation of one tool: what the prompt shows normally and what `help(tool)` shows in the kernel. */
export function toolDoc(t: ToolDescriptor): string {
  const lines = [head(t), `  ${t.description}`];
  for (const [name, p] of Object.entries(t.parameters.properties ?? {})) {
    if (p.description) lines.push(`  - ${name}: ${p.description}`);
  }
  return lines.join("\n");
}

export function firstSentence(text: string): string {
  const masked = text.trim().replace(/\b(e\.g|i\.e|etc|vs)\./gi, (m) => m.replace(/\./g, "\u0000"));
  const m = /^[\s\S]*?[.!?](?=\s|$)/.exec(masked);
  return (m ? m[0] : masked).replace(/\u0000/g, ".").trim();
}

/** Renders the prompt section documenting every kernel callable. Prompts reference tools only
 * through this section, so adding a tool never requires editing a prompt. `compact` keeps one line per tool for
 * small context windows; the full text stays reachable through `help(...)`. */
export function renderToolSurface(tools: ToolDescriptor[], compact = false): string {
  const byNs = new Map<string, ToolDescriptor[]>();
  for (const t of tools) {
    const ns = t.id.split(".")[0] ?? t.id;
    byNs.set(ns, [...(byNs.get(ns) ?? []), t]);
  }
  const lines: string[] = compact ? ["Short form for a small context window. Before using a tool for the first time, run `help(namespace.tool)` in a cell to print its full documentation.", ""] : [];
  for (const [ns, list] of [...byNs.entries()].sort()) {
    lines.push(`### ${ns}`);
    for (const t of list.sort((a, b) => a.id.localeCompare(b.id))) {
      const summary = firstSentence(t.description);
      lines.push(!compact ? toolDoc(t) : summary ? `${head(t)}: ${summary}` : head(t));
    }
    lines.push("");
  }
  return lines.join("\n").trim();
}
