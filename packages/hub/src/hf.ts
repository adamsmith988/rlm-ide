/** Minimal Hugging Face Hub client: repo info, recursive file tree, file contents, download URLs. */

export const HF_BASE = "https://huggingface.co";

export interface RepoFile {
  path: string;
  size: number;
  /** sha256 of LFS files (the Hub stores it as the LFS oid); null for small git files. */
  sha256: string | null;
}

export interface RawModelInfo {
  id: string;
  sha?: string;
  private?: boolean;
  gated?: false | "auto" | "manual";
  pipeline_tag?: string;
  library_name?: string;
  tags?: string[];
  config?: { architectures?: string[]; model_type?: string; tokenizer_config?: { chat_template?: unknown }; quantization_config?: { quant_method?: string } };
  transformersInfo?: { auto_model?: string; pipeline_tag?: string };
  safetensors?: { total?: number };
  gguf?: { total?: number; context_length?: number; architecture?: string; chat_template?: string };
}

export class HubError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "HubError";
  }
}

/** Accepts "org/name", "name", or a huggingface.co / hf.co URL (optionally /tree/<rev>/...). */
export function parseRepoRef(input: string, revision = ""): { repo: string; revision: string } {
  let s = input.trim().replace(/\/+$/, "");
  let rev = revision.trim();
  const url = /^(?:https?:\/\/)?(?:www\.)?(?:huggingface\.co|hf\.co)\/(.+)$/i.exec(s);
  if (url) {
    const parts = url[1]!.split("/");
    if (parts[0] === "models") parts.shift();
    const treeAt = parts.findIndex((p) => p === "tree" || p === "blob" || p === "resolve");
    if (treeAt >= 0) {
      if (!rev && parts[treeAt + 1]) rev = decodeURIComponent(parts[treeAt + 1]!);
      parts.length = treeAt;
    }
    s = parts.slice(0, 2).join("/");
  }
  if (!/^[\w.-]+(\/[\w.-]+)?$/.test(s) || s.includes("..")) throw new HubError(`"${input}" is not a model repo id; use the form org/name`, 400);
  if (rev && !/^[\w./-]+$/.test(rev)) throw new HubError(`"${rev}" is not a valid revision`, 400);
  return { repo: s, revision: rev || "main" };
}

const encPath = (p: string) => p.split("/").map(encodeURIComponent).join("/");

export class HfClient {
  constructor(
    private readonly token: () => string | null,
    private readonly base = HF_BASE,
    private readonly fetchImpl: typeof fetch = fetch,
  ) {}

  headers(extra: Record<string, string> = {}): Record<string, string> {
    const t = this.token();
    return { ...(t ? { Authorization: `Bearer ${t}` } : {}), ...extra };
  }

  hasToken(): boolean {
    return !!this.token();
  }

  private async get(url: string, what: string, signal?: AbortSignal): Promise<Response> {
    let res: Response;
    try {
      res = await this.fetchImpl(url, { headers: this.headers(), signal, redirect: "follow" });
    } catch (e) {
      if ((e as Error).name === "AbortError") throw e;
      throw new HubError(`cannot reach ${new URL(url).host}: ${(e as Error).message}`, 0);
    }
    if (res.ok) return res;
    const body = await res.text().catch(() => "");
    const msg = (() => {
      try {
        return (JSON.parse(body) as { error?: string }).error ?? body;
      } catch {
        return body;
      }
    })().slice(0, 300);
    if (res.status === 401)
      throw new HubError(this.hasToken() ? `${what}: the Hugging Face token was rejected or has no access (${msg})` : `${what}: not found, private, or gated. Gated and private repos need a Hugging Face token (Models hub → Tools).`, 401);
    if (res.status === 403) throw new HubError(`${what}: access denied. If the repo is gated, accept its license on huggingface.co with the account of your token. (${msg})`, 403);
    if (res.status === 404) throw new HubError(`${what}: not found (${msg || "no such repo or revision"})`, 404);
    throw new HubError(`${what}: HTTP ${res.status} ${msg}`, res.status);
  }

  async info(repo: string, revision: string, signal?: AbortSignal): Promise<RawModelInfo> {
    const url = `${this.base}/api/models/${repo}${revision && revision !== "main" ? `/revision/${encodeURIComponent(revision)}` : ""}`;
    return (await (await this.get(url, `repo ${repo}`, signal)).json()) as RawModelInfo;
  }

  /** Every file of the repo at a commit, following the Hub's pagination. */
  async tree(repo: string, sha: string, signal?: AbortSignal): Promise<RepoFile[]> {
    const out: RepoFile[] = [];
    let url: string | null = `${this.base}/api/models/${repo}/tree/${encodeURIComponent(sha)}?recursive=true&expand=false`;
    while (url) {
      const res = await this.get(url, `file list of ${repo}`, signal);
      const items = (await res.json()) as Array<{ type: string; path: string; size?: number; lfs?: { oid?: string; size?: number } }>;
      for (const it of items) if (it.type === "file") out.push({ path: it.path, size: it.lfs?.size ?? it.size ?? 0, sha256: it.lfs?.oid ?? null });
      const link = res.headers.get("link") ?? "";
      const next = /<([^>]+)>;\s*rel="next"/.exec(link);
      url = next ? new URL(next[1]!, this.base).toString() : null;
    }
    return out;
  }

  resolveUrl(repo: string, sha: string, file: string): string {
    return `${this.base}/${repo}/resolve/${encodeURIComponent(sha)}/${encPath(file)}`;
  }

  /** Small text file (config, tokenizer config); null when missing or not allowed. */
  async text(repo: string, sha: string, file: string, signal?: AbortSignal): Promise<string | null> {
    try {
      return await (await this.get(this.resolveUrl(repo, sha, file), file, signal)).text();
    } catch (e) {
      if (e instanceof HubError && [401, 403, 404].includes(e.status)) return null;
      throw e;
    }
  }

  /** Whether the files themselves can be fetched (gated repos list files but refuse downloads). */
  async canDownload(repo: string, sha: string, file: string): Promise<{ ok: boolean; status: number }> {
    try {
      const res = await this.fetchImpl(this.resolveUrl(repo, sha, file), { method: "HEAD", headers: this.headers(), redirect: "follow" });
      return { ok: res.ok, status: res.status };
    } catch {
      return { ok: false, status: 0 };
    }
  }
}
