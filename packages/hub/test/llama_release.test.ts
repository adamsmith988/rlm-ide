import { mkdirSync, mkdtempSync, readFileSync, readlinkSync, rmSync, symlinkSync, writeFileSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { afterAll, describe, expect, it } from "vitest";
import { compareVersions, copyTree, extract, findDirWith, moveTree, listReleases, pickLlamaAssets, type LlamaRelease } from "../src/llama_release";

const names = [
  "cudart-llama-bin-win-cuda-12.4-x64.zip",
  "cudart-llama-bin-win-cuda-13.1-x64.zip",
  "cudart-llama-b6500-bin-ubuntu-cuda-12.8-x64.tar.gz",
  "llama-b6500-bin-macos-arm64.tar.gz",
  "llama-b6500-bin-macos-x64.tar.gz",
  "llama-b6500-bin-ubuntu-x64.tar.gz",
  "llama-b6500-bin-ubuntu-vulkan-x64.tar.gz",
  "llama-b6500-bin-ubuntu-cuda-12.8-x64.tar.gz",
  "llama-b6500-bin-ubuntu-s390x.tar.gz",
  "llama-b6500-bin-win-cpu-x64.zip",
  "llama-b6500-bin-win-cpu-arm64.zip",
  "llama-b6500-bin-win-cuda-12.4-x64.zip",
  "llama-b6500-bin-win-cuda-13.1-x64.zip",
  "llama-b6500-bin-win-vulkan-x64.zip",
  "llama-b6500-bin-win-hip-radeon-x64.zip",
  "llama-b6500-bin-win-sycl-x64.zip",
  "llama-b6500-xcframework.zip",
];
const rel: LlamaRelease = { tag: "b6500", assets: names.map((name) => ({ name, url: `https://x/${name}`, size: 1 })) };
const pick = (platform: NodeJS.Platform, arch: string, want: "auto" | "cpu" | "gpu", cudaMax: string | null) => pickLlamaAssets(rel, { platform, arch, want, cudaMax });

describe("compareVersions", () => {
  it("orders numerically, not as text", () => {
    expect(compareVersions("12.4", "12.10")).toBeLessThan(0);
    expect(compareVersions("13.0", "12.10")).toBeGreaterThan(0);
    expect(compareVersions("12.4", "12.4.0")).toBe(0);
  });
});

describe("pickLlamaAssets", () => {
  it("Windows with a new NVIDIA driver: the newest CUDA build it supports, with its runtime", () => {
    const p = pick("win32", "x64", "auto", "13.2")!;
    expect(p.flavor).toBe("cuda");
    expect(p.main.name).toBe("llama-b6500-bin-win-cuda-13.1-x64.zip");
    expect(p.runtime?.name).toBe("cudart-llama-bin-win-cuda-13.1-x64.zip");
    expect(p.cudaVersion).toBe("13.1");
  });

  it("Windows with an older driver skips CUDA builds newer than the driver", () => {
    const p = pick("win32", "x64", "auto", "12.8")!;
    expect(p.main.name).toBe("llama-b6500-bin-win-cuda-12.4-x64.zip");
    expect(p.runtime?.name).toBe("cudart-llama-bin-win-cuda-12.4-x64.zip");
  });

  it("Windows with a driver too old for any CUDA build, or no NVIDIA GPU: Vulkan", () => {
    expect(pick("win32", "x64", "auto", "11.8")!.main.name).toBe("llama-b6500-bin-win-vulkan-x64.zip");
    const p = pick("win32", "x64", "gpu", null)!;
    expect(p.flavor).toBe("vulkan");
    expect(p.runtime).toBeNull();
  });

  it("CPU only when asked, on every system", () => {
    expect(pick("win32", "x64", "cpu", "13.2")!.main.name).toBe("llama-b6500-bin-win-cpu-x64.zip");
    expect(pick("win32", "arm64", "auto", null)!.main.name).toBe("llama-b6500-bin-win-cpu-arm64.zip");
    expect(pick("linux", "x64", "cpu", "12.8")!.main.name).toBe("llama-b6500-bin-ubuntu-x64.tar.gz");
  });

  it("Linux: CUDA with the runtime when the driver allows, else Vulkan", () => {
    const c = pick("linux", "x64", "auto", "12.9")!;
    expect(c.main.name).toBe("llama-b6500-bin-ubuntu-cuda-12.8-x64.tar.gz");
    expect(c.runtime?.name).toBe("cudart-llama-b6500-bin-ubuntu-cuda-12.8-x64.tar.gz");
    expect(pick("linux", "x64", "auto", null)!.main.name).toBe("llama-b6500-bin-ubuntu-vulkan-x64.tar.gz");
  });

  it("macOS: the Metal build for the right chip", () => {
    expect(pick("darwin", "arm64", "auto", null)).toMatchObject({ flavor: "metal", main: { name: "llama-b6500-bin-macos-arm64.tar.gz" } });
    expect(pick("darwin", "x64", "auto", null)!.main.name).toBe("llama-b6500-bin-macos-x64.tar.gz");
  });

  it("nothing for systems without a build", () => {
    expect(pick("freebsd", "x64", "auto", null)).toBeNull();
    expect(pick("linux", "arm64", "auto", null)).toBeNull();
    expect(pickLlamaAssets({ tag: "v0.7.0", assets: [{ name: "nightly-tag.txt", url: "", size: 1 }] }, { platform: "win32", arch: "x64", want: "auto", cudaMax: null })).toBeNull();
  });
});

describe("listReleases", () => {
  it("lists published releases and skips drafts", async () => {
    let asked = "";
    const fake = (async (url: string) => {
      asked = url;
      return new Response(
        JSON.stringify([
          { tag_name: "b2", draft: true, assets: [] },
          { tag_name: "b1", assets: [{ name: "a.zip", browser_download_url: "https://d/a.zip", size: 3 }] },
        ]),
      );
    }) as unknown as typeof fetch;
    const list = await listReleases("ggml-org/llama.cpp", "", fake);
    expect(asked).toContain("/repos/ggml-org/llama.cpp/releases?");
    expect(list).toEqual([{ tag: "b1", assets: [{ name: "a.zip", url: "https://d/a.zip", size: 3 }] }]);
  });

  it("asks for one tag when pinned and reports HTTP errors", async () => {
    const fake = (async () => new Response("rate limited", { status: 403 })) as unknown as typeof fetch;
    await expect(listReleases("ggml-org/llama.cpp", "b6500", fake)).rejects.toThrow(/HTTP 403 rate limited/);
  });
});

describe("unpacking", () => {
  const root = mkdtempSync(path.join(tmpdir(), "llama-rel-"));
  afterAll(() => rmSync(root, { recursive: true, force: true }));

  it("finds the binaries wherever the archive nests them", async () => {
    const src = path.join(root, "src");
    mkdirSync(path.join(src, "build", "bin"), { recursive: true });
    writeFileSync(path.join(src, "build", "bin", "llama-server"), "x");
    const archive = path.join(root, "a.tar.gz");
    execFileSync("tar", ["-czf", archive, "-C", src, "."]);
    const out = path.join(root, "out");
    await extract(archive, out);
    expect(findDirWith(out, "llama-server")).toBe(path.join(out, "build", "bin"));
    expect(findDirWith(out, "missing")).toBeNull();
    expect(existsSync(path.join(out, "build", "bin", "llama-server"))).toBe(true);
  });

  it.skipIf(process.platform === "win32")("keeps relative library links working after the unpack folder is gone", async () => {
    const src = path.join(root, "links");
    mkdirSync(src, { recursive: true });
    writeFileSync(path.join(src, "libllama.so.0.7.0"), "lib");
    symlinkSync("libllama.so.0.7.0", path.join(src, "libllama.so.0"));
    const archive = path.join(root, "links.tar.gz");
    execFileSync("tar", ["-czf", archive, "-C", src, "."]);
    const work = path.join(root, "work");
    await extract(archive, work);
    const copied = path.join(root, "copied");
    copyTree(work, copied);
    const moved = path.join(root, "install", "bin");
    moveTree(work, moved);
    expect(existsSync(work)).toBe(false);
    for (const dir of [copied, moved]) {
      expect(readlinkSync(path.join(dir, "libllama.so.0"))).toBe("libllama.so.0.7.0");
      expect(readFileSync(path.join(dir, "libllama.so.0"), "utf8")).toBe("lib");
    }
  });

  it("explains an archive it cannot unpack", async () => {
    const bad = path.join(root, "bad.tar.gz");
    writeFileSync(bad, "not an archive");
    await expect(extract(bad, path.join(root, "bad"))).rejects.toThrow(/could not unpack bad.tar.gz/);
  });
});
