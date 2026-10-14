import { describe, expect, it } from "vitest";
import { CellFenceFilter, extractCells, stripCells, syntaxHint, unclosedCellAt } from "../src/application/rlm/cells";

describe("cells", () => {
  it("extracts closed repl cells only", () => {
    const t = "Plan.\n```repl\nprint(1)\n```\nthen\n```python\nnot_run()\n```\n```repl\nprint(2)\n```";
    expect(extractCells(t)).toEqual(["print(1)", "print(2)"]);
    expect(unclosedCellAt(t)).toBe(-1);
  });

  it("does not close a cell on backticks inside a string", () => {
    const t = '```repl\nfs.write_file(path="README.md", content="```js\\nx()\\n```")\nprint("ok")\n```';
    expect(extractCells(t)).toEqual(['fs.write_file(path="README.md", content="```js\\nx()\\n```")\nprint("ok")']);
  });

  it("detects a cell cut off at the output limit", () => {
    const t = "Writing files.\n```repl\nprint(1)\n```\n```repl\nfs.write_file(path='a.html', content='''<html>";
    expect(extractCells(t)).toEqual(["print(1)"]);
    expect(unclosedCellAt(t)).toBe(t.indexOf("```repl\nfs"));
    expect(stripCells(t.slice(0, unclosedCellAt(t)))).toBe("Writing files.");
  });

  it("hints when JavaScript is run as Python", () => {
    expect(syntaxHint("const x = require('./a');", "SyntaxError: invalid syntax")).toMatch(/shell\.run/);
    expect(syntaxHint('fs.write_file(path="a.js", content="x\\n', "SyntaxError: unterminated string literal")).toMatch(/triple-quoted/);
    expect(syntaxHint("print(1)", "NameError: x")).toBe("");
  });

  it("stream filter hides cells split across chunks", () => {
    const f = new CellFenceFilter();
    const text = "Hi.\n```repl\ncode = '```python'\nprint(code)\n```\nDone.";
    let out = "";
    for (let i = 0; i < text.length; i += 3) out += f.push(text.slice(i, i + 3));
    out += f.flush();
    expect(out).toBe("Hi.\n\nDone.");
  });
});
