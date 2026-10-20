import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { defineConfig, externalizeDepsPlugin } from "electron-vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

const workspacePkgs = ["@rlm/contracts", "@rlm/core", "@rlm/adapters", "@rlm/inference", "@rlm/team", "@rlm/hub"];

export default defineConfig({
  main: {
    plugins: [externalizeDepsPlugin({ exclude: workspacePkgs })],
    build: {
      rollupOptions: {
        input: {
          index: resolve(__dirname, "src/main/index.ts"),
          orchestrator: resolve(__dirname, "src/orchestrator/index.ts"),
          "inference-cpu": resolve(__dirname, "src/inference-cpu/index.ts"),
        },
        external: ["@huggingface/transformers", "node-pty", "sqlite-vec", "onnxruntime-node", "electron-updater"],
        output: { intro: readFileSync(resolve(__dirname, "src/main/stdio_guard.cjs"), "utf8") },
      },
    },
  },
  preload: {
    plugins: [externalizeDepsPlugin({ exclude: workspacePkgs })],
    build: {
      rollupOptions: {
        input: {
          index: resolve(__dirname, "src/preload/index.ts"),
          inference: resolve(__dirname, "src/preload/inference.ts"),
        },
        output: { format: "cjs", entryFileNames: "[name].cjs" },
      },
    },
  },
  renderer: {
    root: resolve(__dirname, "src/renderer"),
    plugins: [react(), tailwindcss()],
    worker: { format: "es" },
    build: {
      rollupOptions: {
        input: {
          index: resolve(__dirname, "src/renderer/index.html"),
          inference: resolve(__dirname, "src/renderer/inference.html"),
        },
      },
    },
  },
});
