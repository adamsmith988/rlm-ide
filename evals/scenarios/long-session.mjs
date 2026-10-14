import { check } from "../lib/checks.mjs";

const lorem = (topic, n) =>
  Array.from({ length: n }, (_, i) => `- ${topic} note ${i + 1}: the ${topic} module handles case ${i + 1} by validating input, logging the outcome and returning a typed result.`).join("\n");

export default {
  id: "long-session",
  title: "Long session: compaction keeps early facts",
  // A small window forces session compaction after a few turns.
  contextLength: 8192,
  seed: {
    "docs/auth.md": `# Auth\n\n${lorem("auth", 40)}\n`,
    "docs/billing.md": `# Billing\n\n${lorem("billing", 40)}\n`,
    "docs/search.md": `# Search\n\n${lorem("search", 40)}\n`,
    "docs/storage.md": `# Storage\n\n${lorem("storage", 40)}\n`,
  },
  turns: [
    { prompt: "Our project codename is BLUE-HERON-42 and the release date is 14 March. Just acknowledge, no tools needed." },
    { prompt: "Read docs/auth.md and list every note in it, one line each, paraphrased in your own words." },
    { prompt: "Read docs/billing.md and list every note in it, one line each, paraphrased in your own words." },
    { prompt: "Read docs/search.md and list every note in it, one line each, paraphrased in your own words." },
    { prompt: "Read docs/storage.md and list every note in it, one line each, paraphrased in your own words." },
    { prompt: "What is the project codename and the release date I told you at the start? Answer in one line." },
  ],
  checks(ws, { results }) {
    const last = results[results.length - 1];
    const compactions = results.reduce((n, r) => n + r.events.filter((e) => e.type === "compaction").length, 0);
    return [
      check("all turns completed", results.every((r) => r.status === "done"), results.map((r) => r.status).join(",")),
      check("session compaction happened", compactions > 0, `${compactions} compaction events`),
      check("codename recalled after compaction", /BLUE-HERON-42/i.test(last?.text ?? ""), (last?.text ?? "").slice(0, 200)),
      check("release date recalled", /14(th)?\s+march|march\s+14/i.test(last?.text ?? "")),
    ];
  },
};
