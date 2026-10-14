import { createHash } from "node:crypto";
import { check, read, run } from "../lib/checks.mjs";

const TEST = `import { test } from "node:test";
import assert from "node:assert/strict";
import { Cart } from "../src/cart.js";

test("total multiplies price by quantity", () => {
  const c = new Cart();
  c.add({ id: "a", price: 2.5, quantity: 4 });
  c.add({ id: "b", price: 10, quantity: 1 });
  assert.equal(c.total(), 20);
});

test("adding the same id increases quantity", () => {
  const c = new Cart();
  c.add({ id: "a", price: 3, quantity: 1 });
  c.add({ id: "a", price: 3, quantity: 2 });
  assert.equal(c.items.length, 1);
  assert.equal(c.items[0].quantity, 3);
});

test("percentage discount", () => {
  const c = new Cart();
  c.add({ id: "a", price: 50, quantity: 2 });
  assert.equal(c.totalWithDiscount(15), 85);
});

test("remove deletes an item", () => {
  const c = new Cart();
  c.add({ id: "a", price: 1, quantity: 1 });
  c.add({ id: "b", price: 1, quantity: 1 });
  c.remove("a");
  assert.deepEqual(c.items.map((i) => i.id), ["b"]);
});
`;

const CART = `export class Cart {
  constructor() {
    this.items = [];
  }

  add(item) {
    const existing = this.items.find((i) => i.id === item.id);
    if (existing) {
      existing.quantity = item.quantity;
      return;
    }
    this.items.push({ ...item });
  }

  remove(id) {
    this.items = this.items.filter((i) => i.id === id);
  }

  total() {
    return this.items.reduce((sum, i) => sum + i.price, 0);
  }

  totalWithDiscount(percent) {
    return this.total() * (1 - percent / 10);
  }
}
`;

const sha = (s) => createHash("sha256").update(s ?? "").digest("hex");

export default {
  id: "fix-bugs",
  title: "Find and fix bugs until tests pass (approvals on)",
  approvals: "ask",
  seed: {
    "package.json": JSON.stringify({ name: "cart", version: "1.0.0", type: "module", scripts: { test: "node --test" } }, null, 2) + "\n",
    "src/cart.js": CART,
    "test/cart.test.js": TEST,
  },
  turns: [{ prompt: "The tests in this project fail. Run them, find the bugs in src/ and fix them. Do not change the tests." }],
  checks(ws) {
    const t = run(ws, process.execPath, ["--test"]);
    return [
      check("tests pass", t.code === 0, t.out.split("\n").slice(-8).join(" | ")),
      check("tests untouched", sha(read(ws, "test/cart.test.js")) === sha(TEST)),
      check("fixed in src/cart.js", read(ws, "src/cart.js") !== CART),
    ];
  },
  judge: {
    task: "Fix the bugs in src/cart.js so the existing tests pass without changing the tests.",
    files: "src",
    rubric: ["All four bugs are fixed correctly (quantity merge, remove filter, total uses quantity, discount percent).", "The fix is minimal and keeps the original style.", "The agent verified by running the tests and reported clearly."],
  },
};
