import { execFile } from "node:child_process";
import path from "node:path";

const JS_EXT = new Set([".js", ".mjs", ".cjs"]);

/** Quick syntax check of a file the agent just wrote. Returns a one-paragraph problem report, or "" when the
 * file parses (or its language is not checked). Uses `node --check`, which honours "type": "module". */
export async function syntaxProblem(abs: string, content: string, rel: string): Promise<string> {
  const ext = path.extname(abs).toLowerCase();
  if (ext === ".json") {
    try {
      JSON.parse(content);
      return "";
    } catch (e) {
      return `${rel} is not valid JSON: ${(e as Error).message}`;
    }
  }
  if (!JS_EXT.has(ext)) return "";
  const out = await new Promise<string>((resolve) => {
    execFile(process.execPath, ["--check", abs], { timeout: 10000, env: { ...process.env, ELECTRON_RUN_AS_NODE: "1" } }, (err, _o, stderr) =>
      resolve(err && !err.killed ? String(stderr) : ""),
    );
  });
  if (!out.trim()) return "";
  const lines = out.split("\n");
  const at = /:(\d+)\s*$/.exec(lines[0] ?? "")?.[1];
  const message = lines.find((l) => /^\w*Error:/.test(l)) ?? lines.find((l) => l.trim()) ?? "syntax error";
  const source = lines[1]?.trim();
  return `${rel} has a syntax error${at ? ` at line ${at}` : ""}: ${message.trim()}${source ? `\n    ${source}` : ""}\nThe file was saved as written. Fix it with fs.edit_file.`;
}
