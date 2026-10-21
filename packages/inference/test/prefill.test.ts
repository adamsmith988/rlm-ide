import { describe, expect, it } from "vitest";
import { cpuSessionOptions, prefillChunk } from "../src/generate";

const GiB = 2 ** 30;
const scores = (chunk: number, total: number, heads: number) => chunk * total * heads * 4;

describe("prefillChunk", () => {
  it("keeps the attention scores of every pass well under Electron's 1 GiB allocation limit", () => {
    for (const total of [4144, 4589, 8192, 13039, 32768, 131072]) {
      const c = Math.min(prefillChunk(total, 16), total);
      expect(scores(c, total, 16)).toBeLessThanOrEqual(GiB / 4 + 64 * total * 16 * 4);
    }
  });

  it("reads short prompts in one pass", () => {
    expect(prefillChunk(2000, 16)).toBeGreaterThan(2000);
  });

  it("splits the prompts that crashed into chunks that ran in a utility process", () => {
    expect(prefillChunk(4589, 16)).toBeLessThan(4589);
    expect(prefillChunk(13027, 16)).toBe(321);
  });

  it("uses smaller chunks for models with more heads", () => {
    expect(prefillChunk(8192, 32)).toBeLessThan(prefillChunk(8192, 16));
  });
});

describe("cpuSessionOptions", () => {
  it("turns the memory arena off and sets threads only when known", () => {
    expect(cpuSessionOptions(16)).toEqual({ enableCpuMemArena: false, intraOpNumThreads: 16 });
    expect(cpuSessionOptions(0)).toEqual({ enableCpuMemArena: false });
  });
});
