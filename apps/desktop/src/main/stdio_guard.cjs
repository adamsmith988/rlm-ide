// Injected at the top of every main-process bundle (rollup `output.intro`), before any module loads.
// A Windows GUI app can inherit stdout/stderr handles it cannot open (launched by an installer, an updater or another
// process): Node's lazy `process.stderr` getter then throws EBADF, and the first module that touches it (loading
// `assert` does) crashes the app with "A JavaScript error occurred in the main process". Logging must never do that.
(() => {
  const { Writable } = require("node:stream");
  for (const name of ["stdout", "stderr"]) {
    try {
      // A reader that goes away later turns writes into async EPIPE errors; ignore them too.
      process[name].on("error", () => undefined);
    } catch {
      Object.defineProperty(process, name, {
        configurable: true,
        enumerable: true,
        writable: true,
        value: new Writable({ write: (_chunk, _enc, cb) => cb() }),
      });
    }
  }
})();
