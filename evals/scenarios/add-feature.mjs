import { writeFileSync } from "node:fs";
import path from "node:path";
import { check, read, run } from "../lib/checks.mjs";

const TODO = `let nextId = 1;

export class TodoList {
  constructor() {
    this.todos = [];
  }

  add(title) {
    const todo = { id: nextId++, title, done: false };
    this.todos.push(todo);
    return todo;
  }

  toggle(id) {
    const t = this.todos.find((x) => x.id === id);
    if (t) t.done = !t.done;
    return t;
  }

  pending() {
    return this.todos.filter((t) => !t.done);
  }
}
`;

const TEST = `import { test } from "node:test";
import assert from "node:assert/strict";
import { TodoList } from "../src/todo.js";

test("add and toggle", () => {
  const l = new TodoList();
  const t = l.add("Write docs");
  l.toggle(t.id);
  assert.equal(l.pending().length, 0);
});
`;

const HIDDEN = `import { test } from "node:test";
import assert from "node:assert/strict";
import { TodoList } from "./src/todo.js";

test("hidden: search is case-insensitive substring match", () => {
  const l = new TodoList();
  l.add("Buy Milk");
  l.add("buy bread");
  l.add("Call mom");
  assert.deepEqual(l.search("BUY").map((t) => t.title), ["Buy Milk", "buy bread"]);
  assert.deepEqual(l.search("zzz"), []);
});

test("hidden: empty query returns all todos", () => {
  const l = new TodoList();
  l.add("a");
  l.add("b");
  assert.equal(l.search("").length, 2);
});
`;

export default {
  id: "add-feature",
  title: "Add a feature with tests",
  seed: {
    "package.json": JSON.stringify({ name: "todo", version: "1.0.0", type: "module", scripts: { test: "node --test" } }, null, 2) + "\n",
    "src/todo.js": TODO,
    "test/todo.test.js": TEST,
  },
  turns: [
    {
      prompt:
        "Add a `search(query)` method to TodoList in src/todo.js. It returns the todos whose title contains the query, case-insensitively; an empty query returns all todos. Add tests for it in test/todo.test.js and make sure all tests pass.",
    },
  ],
  checks(ws) {
    const t = run(ws, process.execPath, ["--test"]);
    writeFileSync(path.join(ws, "hidden.test.mjs"), HIDDEN);
    const h = run(ws, process.execPath, ["--test", "hidden.test.mjs"]);
    const tests = read(ws, "test/todo.test.js") ?? "";
    return [
      check("project tests pass", t.code === 0, t.out.split("\n").slice(-6).join(" | ")),
      check("hidden behaviour tests pass", h.code === 0, h.out.split("\n").slice(-6).join(" | ")),
      check("new tests cover search", /search\(/.test(tests)),
    ];
  },
  judge: {
    task: "Add TodoList.search(query) (case-insensitive substring; empty query returns all) with tests.",
    files: ".",
    rubric: ["search() is correct, including edge cases.", "Tests are meaningful and cover the edge cases.", "Code matches the existing style and the agent verified the result."],
  },
};
