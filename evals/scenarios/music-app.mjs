import { check, read, staticAppChecks } from "../lib/checks.mjs";

const TASK =
  "Build a static music player web app in a new folder music-app/ (index.html, styles.css, app.js; no frameworks, no build step). " +
  "Design: light theme with black accents, rounded cards, italic serif headings, pill buttons. It has: " +
  "(1) a 'Categories' grid of tiles (Pop, Jazz, Rock, Indie, Lo-fi, Classical); clicking a tile selects it and the selected tile is filled black; " +
  "(2) a playlist of at least 8 songs (title, artist, duration) filtered by the selected category; " +
  "(3) a 'Now playing' card with a circular cover placeholder, the song title and artist, a progress bar and round prev/play/next buttons. " +
  "Play toggles to pause and the progress bar advances while playing (simulate with a timer; no audio files). Clicking a song in the playlist makes it the current song.";

export default {
  id: "music-app",
  title: "Build the music player (mockup 2)",
  turns: [{ prompt: TASK }],
  render: {
    entry: "music-app/index.html",
    probe: `
      const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
      const all = () => [...document.querySelectorAll("button, [role=button], [onclick], .tile, li, [class*=tile], [class*=song], [data-category], [data-index]")];
      const label = (b) => [b.innerText, b.getAttribute("aria-label"), b.title, b.id, typeof b.className === "string" ? b.className : ""].join(" ");
      const btn = (re) => [...document.querySelectorAll("button, [role=button]")].find((b) => re.test(label(b)));
      const out = {};
      const jazz = all().find((b) => /^\\s*jazz\\s*$/i.test(b.innerText));
      if (jazz) {
        const before = jazz.outerHTML + getComputedStyle(jazz).backgroundColor;
        jazz.click(); await sleep(250);
        out.tileSelects = before !== jazz.outerHTML + getComputedStyle(jazz).backgroundColor;
        const tile = document.querySelector("[data-category='Jazz'], [data-category='jazz']") ?? [...document.querySelectorAll("*")].filter((e) => /^\\s*jazz\\s*$/i.test(e.innerText ?? "")).pop();
        const label = [tile, ...tile.querySelectorAll("*")].filter((e) => /^\\s*jazz\\s*$/i.test(e.innerText ?? "")).pop();
        let bg = "rgba(0, 0, 0, 0)";
        for (let e = label; e && /rgba\\(0, 0, 0, 0\\)|transparent/.test(bg); e = e.parentElement) bg = getComputedStyle(e).backgroundColor;
        const lum = (c) => { const [r, g, b] = (c.match(/[\\d.]+/g) ?? [0, 0, 0]).slice(0, 3).map((v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); }); return 0.2126 * r + 0.7152 * g + 0.0722 * b; };
        const l1 = lum(getComputedStyle(label).color), l2 = lum(bg);
        out.selectedTileContrast = Math.round(((Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05)) * 10) / 10;
        out.selectedTileReadable = out.selectedTileContrast >= 3;
      }
      const progress = () => [...document.querySelectorAll("[class*=progress], [id*=progress], progress, [role=progressbar], input[type=range]")]
        .map((e) => e.outerHTML.length + "|" + (e.value ?? "") + "|" + e.style.width + "|" + [...e.querySelectorAll("*")].map((c) => c.style.width).join(",")).join(";");
      const play = btn(/play|▶|►/i);
      out.playButton = !!play;
      if (play) {
        const before = label(play) + play.innerHTML;
        const p0 = progress();
        play.click(); await sleep(1800);
        out.playToggles = before !== label(play) + play.innerHTML;
        out.progressAdvances = progress() !== p0;
      }
      const titleBefore = document.body.innerText;
      const visible = (sel) => [...document.querySelectorAll(sel)].filter((e) => e.offsetParent);
      const items = visible("[data-index], [data-song], [data-song-id], [data-id]").filter((e) => !e.closest("[class*=categor]"));
      const song = (items.length ? items : visible("li, tr")).slice(-1)[0];
      if (song) { song.click(); await sleep(250); out.songSelects = document.body.innerText !== titleBefore; }
      return out;
    `,
  },
  checks(ws, { render }) {
    const js = read(ws, "music-app/app.js") ?? "";
    const css = read(ws, "music-app/styles.css") ?? "";
    const html = read(ws, "music-app/index.html") ?? "";
    const songs = (js.match(/artist\s*:/g) ?? []).length;
    const text = render?.text ?? "";
    const p = render?.probe ?? {};
    return [
      ...staticAppChecks(ws, "music-app", render),
      check("at least 8 songs", songs >= 8, `${songs} found`),
      check("all six categories shown", ["Pop", "Jazz", "Rock", "Indie", "Lo-fi", "Classical"].every((c) => new RegExp(c, "i").test(text + html + js))),
      check("circular cover", /border-radius\s*:\s*50%/.test(css)),
      check("italic serif headings", /font-style\s*:\s*italic/.test(css) && /(^|[^-])serif/.test(css.replace(/sans-serif/g, ""))),
      check("tile click selects", p.tileSelects === true, JSON.stringify(p)),
      check("selected tile text stays readable", p.selectedTileReadable === true),
      check("play toggles to pause", p.playToggles === true),
      check("progress advances while playing", p.progressAdvances === true),
      check("clicking a song changes now playing", p.songSelects === true),
    ];
  },
  judge: {
    task: TASK,
    files: "music-app",
    rubric: [
      "All requested features are present and work (category tiles with selection, filtered playlist of 8+ songs, now-playing card, play/pause, advancing progress, prev/next, song click).",
      "Visual design matches the brief: light theme with black accents, rounded cards, italic serif headings, pill/round buttons, circular cover.",
      "Code quality: clean structure, state handled in one place, timers cleaned up, accessible buttons, no obvious bugs.",
    ],
  },
};
