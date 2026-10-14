import * as monaco from "monaco-editor";
import EditorWorker from "monaco-editor/editor/editor.worker.js?worker";
import TsWorker from "monaco-editor/language/typescript/ts.worker.js?worker";
import JsonWorker from "monaco-editor/language/json/json.worker.js?worker";
import CssWorker from "monaco-editor/language/css/css.worker.js?worker";
import HtmlWorker from "monaco-editor/language/html/html.worker.js?worker";

self.MonacoEnvironment = {
  getWorker(_id: string, label: string) {
    if (label === "typescript" || label === "javascript") return new TsWorker();
    if (label === "json") return new JsonWorker();
    if (label === "css" || label === "scss" || label === "less") return new CssWorker();
    if (label === "html" || label === "handlebars" || label === "razor") return new HtmlWorker();
    return new EditorWorker();
  },
};

monaco.editor.defineTheme("rlm-dark", {
  base: "vs-dark",
  inherit: true,
  rules: [
    { token: "comment", foreground: "6a6a70", fontStyle: "italic" },
    { token: "keyword", foreground: "ffffff", fontStyle: "bold" },
    { token: "string", foreground: "b9c7b2" },
    { token: "number", foreground: "d6c19a" },
    { token: "type", foreground: "c9c9cf" },
    { token: "delimiter", foreground: "8a8a90" },
  ],
  colors: {
    "editor.background": "#0b0b0c",
    "editor.foreground": "#e6e6e3",
    "editorLineNumber.foreground": "#3f3f44",
    "editorLineNumber.activeForeground": "#b5b5b9",
    "editor.lineHighlightBackground": "#141415",
    "editor.selectionBackground": "#3a3a40",
    "editorCursor.foreground": "#ffffff",
    "editorIndentGuide.background1": "#1f1f22",
    "editorWidget.background": "#141415",
    "editorWidget.border": "#2a2a2d",
    "diffEditor.insertedTextBackground": "#6eb47822",
    "diffEditor.removedTextBackground": "#dc6e6e22",
    "diffEditor.insertedLineBackground": "#6eb47814",
    "diffEditor.removedLineBackground": "#dc6e6e14",
    "scrollbarSlider.background": "#25252899",
  },
});

monaco.editor.defineTheme("rlm-light", {
  base: "vs",
  inherit: true,
  rules: [
    { token: "comment", foreground: "96969b", fontStyle: "italic" },
    { token: "keyword", foreground: "111111", fontStyle: "bold" },
    { token: "string", foreground: "3f6b48" },
    { token: "number", foreground: "8a6a1f" },
  ],
  colors: {
    "editor.background": "#ffffff",
    "editor.foreground": "#1a1a1a",
    "editorLineNumber.foreground": "#c4c4bf",
    "editorLineNumber.activeForeground": "#555555",
    "editor.lineHighlightBackground": "#f5f5f3",
    "editor.selectionBackground": "#dcdcd6",
    "editorCursor.foreground": "#111111",
  },
});

monaco.typescript.typescriptDefaults.setDiagnosticsOptions({ noSemanticValidation: true, noSyntaxValidation: false });
monaco.typescript.javascriptDefaults.setDiagnosticsOptions({ noSemanticValidation: true, noSyntaxValidation: false });
monaco.typescript.typescriptDefaults.setCompilerOptions({ jsx: monaco.typescript.JsxEmit.Preserve, allowJs: true, target: monaco.typescript.ScriptTarget.ESNext });

const EXT: Record<string, string> = {
  ts: "typescript", tsx: "typescript", mts: "typescript", cts: "typescript",
  js: "javascript", jsx: "javascript", mjs: "javascript", cjs: "javascript",
  json: "json", md: "markdown", css: "css", scss: "scss", less: "less", html: "html", htm: "html",
  py: "python", rs: "rust", go: "go", java: "java", c: "c", h: "c", cpp: "cpp", hpp: "cpp", cs: "csharp",
  sh: "shell", bash: "shell", yml: "yaml", yaml: "yaml", toml: "ini", ini: "ini", sql: "sql", xml: "xml", svg: "xml",
  rb: "ruby", php: "php", kt: "kotlin", swift: "swift", lua: "lua", dockerfile: "dockerfile",
};

export function languageFor(path: string): string {
  const name = path.split("/").pop()?.toLowerCase() ?? "";
  if (name === "dockerfile") return "dockerfile";
  return EXT[name.split(".").pop() ?? ""] ?? "plaintext";
}

export { monaco };
