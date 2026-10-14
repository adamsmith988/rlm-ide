import { check, filesFor, isMonochrome, read, staticAppChecks } from "../lib/checks.mjs";

const TASK =
  "Build a small motivational quotes web app in a new folder quotes-app/: index.html, styles.css and app.js (no frameworks, no build step). " +
  "Design: monochrome black and white, a big italic serif quote on a card, the author below it, a round 'next quote' button, " +
  "and category pills (Love, Success, Life, Focus) that filter the quotes. Include at least 12 quotes. Keep the code clean.";

export default {
  id: "quotes-app",
  title: "Build the quotes app (mockup 1)",
  turns: [{ prompt: TASK }],
  render: {
    entry: "quotes-app/index.html",
    probe: `
      const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
      const label = (b) => [b.innerText, b.getAttribute("aria-label"), b.title, b.id, b.className].join(" ");
      const buttons = () => [...document.querySelectorAll("button, [role=button]")];
      const find = (re) => buttons().find((b) => re.test(label(b)));
      const text = () => document.body.innerText;
      const next = find(/next|new quote|another|→|›|»/i);
      let nextChangesQuote = null;
      if (next) {
        const before = text();
        nextChangesQuote = false;
        for (let i = 0; i < 4 && !nextChangesQuote; i++) { next.click(); await sleep(250); nextChangesQuote = text() !== before; }
      }
      const pills = {};
      for (const c of ["Love", "Success", "Life", "Focus"]) pills[c] = !!buttons().find((b) => new RegExp("^\\\\s*" + c + "\\\\s*$", "i").test(b.innerText));
      let filterChangesQuote = null;
      const love = buttons().find((b) => /^\\s*love\\s*$/i.test(b.innerText));
      if (love) { const before = text(); love.click(); await sleep(300); filterChangesQuote = text() !== before; }
      return { nextButton: !!next, nextChangesQuote, pills, filterChangesQuote };
    `,
  },
  checks(ws, { render }) {
    const css = read(ws, "quotes-app/styles.css") ?? "";
    const js = read(ws, "quotes-app/app.js") ?? "";
    const mono = isMonochrome(css);
    const quoteCount = Math.max((js.match(/author\s*:/g) ?? []).length, (js.match(/["'`]\s*,\s*["'`][A-Z][\w .'-]+["'`]\s*\]/g) ?? []).length);
    const p = render?.probe ?? {};
    return [
      ...staticAppChecks(ws, "quotes-app", render),
      check("at least 12 quotes", quoteCount >= 12, `${quoteCount} found`),
      check("monochrome palette", mono.ok, JSON.stringify(mono.off)),
      check("italic serif quote", /font-style\s*:\s*italic/.test(css) && /(^|[^-])serif/.test(css.replace(/sans-serif/g, ""))),
      check("round next button", /border-radius\s*:\s*(50%|9{2,}px|[4-9]\d*px)/.test(css)),
      check("next button changes the quote", p.nextChangesQuote === true, JSON.stringify(p)),
      check("all four category pills", p.pills && Object.values(p.pills).every(Boolean), JSON.stringify(p.pills)),
      check("category filter reacts", p.filterChangesQuote === true),
    ];
  },
  judge: {
    task: TASK,
    files: "quotes-app",
    rubric: [
      "All requested features are present and work (card, author, next button, 4 category filters, 12+ quotes).",
      "Visual design matches the brief: monochrome, big italic serif quote, round button, pill categories, good spacing.",
      "Code quality: clean structure, no dead code, accessible markup (buttons, labels), no obvious bugs.",
    ],
  },
};
