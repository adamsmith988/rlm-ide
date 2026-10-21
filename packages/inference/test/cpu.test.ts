import os from "node:os";
import { describe, expect, it } from "vitest";
import { coresFromCpuinfo, cpuThreads, physicalCores } from "../src/cpu";

const block = (proc: number, phys: number, core: number) => `processor\t: ${proc}\nphysical id\t: ${phys}\ncore id\t\t: ${core}\nflags\t\t: fpu sse\n`;

describe("physical cores", () => {
  it("counts hyper-threads of one core once", () => {
    const smt = [0, 1, 2, 3].flatMap((c) => [block(c, 0, c), block(c + 4, 0, c)]).join("\n");
    expect(coresFromCpuinfo(smt)).toBe(4);
  });

  it("counts the same core id on two sockets as two cores", () => {
    expect(coresFromCpuinfo([block(0, 0, 0), block(1, 1, 0)].join("\n"))).toBe(2);
  });

  it("returns null without core ids (some ARM and VM kernels)", () => {
    expect(coresFromCpuinfo("processor\t: 0\nBogoMIPS\t: 50.00\n\nprocessor\t: 1\nBogoMIPS\t: 50.00\n")).toBeNull();
  });

  it("detects at least one and at most the logical CPUs on this system", async () => {
    const n = await physicalCores();
    expect(n).toBeGreaterThanOrEqual(1);
    expect(n).toBeLessThanOrEqual(os.availableParallelism());
  });

  it("uses the configured thread count when set", async () => {
    expect(await cpuThreads(6)).toBe(6);
    expect(await cpuThreads(0)).toBe(await physicalCores());
  });
});
