import { check, listFiles, read } from "../lib/checks.mjs";

export default {
  id: "memory",
  title: "Cross-chat long-term memory",
  seed: { "README.md": "# Utils\n\nSmall JavaScript helpers.\n" },
  turns: [
    { prompt: "For all my projects: I use 2-space indentation, single quotes in JavaScript, and kebab-case file names. Please remember that. No code needed right now." },
    { newChat: true, waitBeforeMs: 8000, prompt: "Create a small JavaScript module in utils/ with a function that converts a string to title case, and export it." },
  ],
  async checks(ws, { app }) {
    const files = listFiles(ws, "utils").filter((f) => /\.(m?js|ts)$/.test(f));
    const f = files[0];
    const src = f ? read(ws, f) : "";
    const lines = src.split("\n").filter((l) => /^\s+\S/.test(l) && !/^\s+\*/.test(l));
    const indents = lines.map((l) => /^( *)/.exec(l)[1].length).filter((n) => n > 0);
    const twoSpace = indents.length > 0 && indents.every((n) => n % 2 === 0) && Math.min(...indents) === 2 && !/\t/.test(src);
    const name = f ? f.split("/").pop().replace(/\.(m?js|ts)$/, "") : "";
    const facts = (await app.cdp.invoke("memory:list", { query: "" })) ?? [];
    return [
      check("utils module created", !!f, files.join(", ")),
      check("kebab-case file name", /^[a-z0-9]+(-[a-z0-9]+)*$/.test(name), name),
      check("single quotes", src && !/"[^"\n]*"/.test(src.replace(/\/\/.*$/gm, "")) && /'[^'\n]*'/.test(src)),
      check("2-space indentation", twoSpace, `indents: ${[...new Set(indents)].join(",")}`),
      check("preference stored as long-term fact", facts.some((x) => /indent|quote|kebab/i.test(x.text)), facts.map((x) => x.text).join(" | ")),
    ];
  },
};
