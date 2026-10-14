import { app } from "electron";
import { existsSync } from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";

const require = createRequire(__filename);

/** Repo root in development (apps/desktop/out/main -> repo). */
const devRoot = path.resolve(__dirname, "../../../..");

function res(...p: string[]): string {
  return app.isPackaged ? path.join(process.resourcesPath, ...p) : path.join(devRoot, ...p);
}

export const paths = {
  get userData(): string {
    return app.getPath("userData");
  },
  get db(): string {
    return process.env.RLM_DB_PATH ?? path.join(app.getPath("userData"), "rlm-ide.db");
  },
  get snapshots(): string {
    return path.join(app.getPath("userData"), "rlm_snapshots");
  },
  get prompts(): string {
    return res("prompts");
  },
  get workerScript(): string {
    return app.isPackaged ? res("python", "rlm_worker", "worker.py") : res("python", "rlm_worker", "worker.py");
  },
  get hubScripts(): string {
    return res("python", "rlm_hub");
  },
  get manifest(): string {
    return app.isPackaged ? res("models.manifest.json") : res("resources", "models.manifest.json");
  },
  /** Downloaded models (writable) first, then bundled models (read-only). */
  get modelDirs(): string[] {
    return [path.join(app.getPath("userData"), "models"), app.isPackaged ? res("models") : res("resources", "models")];
  },
  get downloadDir(): string {
    return path.join(app.getPath("userData"), "models");
  },
  pythonCandidates(): string[] {
    const bundled = process.platform === "win32" ? res("python", "runtime", "python.exe") : res("python", "runtime", "bin", "python3");
    const devBundled =
      process.platform === "win32"
        ? path.join(devRoot, "resources", "python", `${process.platform}-${process.arch}`, "python.exe")
        : path.join(devRoot, "resources", "python", `${process.platform}-${process.arch}`, "bin", "python3");
    return [process.env.RLM_PYTHON ?? "", app.isPackaged ? bundled : devBundled, "python3", "python"].filter(Boolean);
  },
  vecExtension(): string | null {
    try {
      const p = (require("sqlite-vec") as { getLoadablePath(): string }).getLoadablePath();
      return p.replace(`app.asar${path.sep}`, `app.asar.unpacked${path.sep}`);
    } catch {
      return null;
    }
  },
  ortWebDist(): string | null {
    try {
      // onnxruntime-web does not export ./package.json; its main entry lives in dist/.
      const d = path.dirname(require.resolve("onnxruntime-web", { paths: [require.resolve("@huggingface/transformers")] }));
      return existsSync(d) ? d.replace(`app.asar${path.sep}`, `app.asar.unpacked${path.sep}`) : null;
    } catch {
      return null;
    }
  },
  findModelDir(dir: string): string | null {
    for (const root of this.modelDirs) {
      const p = path.join(root, dir);
      if (existsSync(path.join(p, "config.json"))) return root;
    }
    return null;
  },
};
