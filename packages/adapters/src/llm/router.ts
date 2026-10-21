import { OMNIROUTE_AUTO_MODEL_ID, type ModelConfig, type ProviderConfig, type Settings } from "@rlm/contracts";
import type { ChatModelPort, EmbeddingPort, ModelCatalogPort, ModelLimitsInfo, ModelRouterPort } from "@rlm/core";
import { detectOpenAICompatLimits, OpenAICompatChat, OpenAICompatEmbed, type OpenAICompatConfig } from "./openai_compat";
import { FailoverChat, ModelHealth } from "./failover";
import { isOllama, OllamaChat, ollamaPlan, type OllamaPlan } from "./ollama";

export interface RouterDeps {
  settings: () => Settings;
  apiKey: (providerId: string) => string | null;
  /** Built-in local chat model (FrogNano via the inference host). */
  localChat: (model: ModelConfig) => ChatModelPort;
  /** Built-in local embedder (bge-m3 in-process). Null when not installed. */
  localEmbed: (model: ModelConfig) => Promise<EmbeddingPort | null>;
  /** Max context the local model itself supports (from its config.json), null if not installed. */
  localNativeContext: (model: ModelConfig) => number | null;
  /** User message asking the next model in a failover chain to continue a cut-off reply. */
  continuePrompt: () => string;
  log?: (msg: string, meta: Record<string, unknown>) => void;
}

/** Picks an adapter per role: agent, summarizer, embed. Multi-provider: local ONNX or any
 * OpenAI-compatible API, chosen in Settings → Models. */
export class ProviderRouter implements ModelRouterPort, ModelCatalogPort {
  private readonly chatCache = new Map<string, ChatModelPort>();
  private readonly embedCache = new Map<string, EmbeddingPort>();
  /** Cooldowns of failed models, shared by every failover chain. */
  readonly health = new ModelHealth();

  constructor(private readonly d: RouterDeps) {}

  invalidate(): void {
    this.chatCache.clear();
    this.embedCache.clear();
    this.health.clear();
  }

  /** The role's model followed by its fallbacks: configured chat models only, no repeats. A separate summarizer falls
   * back to the agent chain, so compaction checkpoints and memory distillation never stop with it. */
  chainFor(role: "agent" | "summarizer"): string[] {
    const s = this.d.settings();
    const isChat = (id: string) => s.models.some((m) => m.id === id && m.kind === "chat");
    const ids = [this.modelIdFor(role), s.roles.agent, ...s.roles.agentFallbacks].flatMap((id) => this.expand(id));
    return [...new Set(ids)].filter(isChat);
  }

  /** The "OmniRoute" model stands for its tested free-model chain, best first. */
  private expand(id: string): string[] {
    return id === OMNIROUTE_AUTO_MODEL_ID ? this.d.settings().omniroute.chain : [id];
  }

  /** The configured model a role's id resolves to: the first member for "OmniRoute". */
  private resolve(id: string): string {
    const first = this.expand(id)[0];
    if (!first) throw new Error("OmniRoute has no tested free models yet. Open Settings → Models → Free models and press “Set up free models”.");
    return first;
  }

  modelIdFor(role: "agent" | "summarizer" | "embed"): string {
    const r = this.d.settings().roles;
    if (role === "summarizer") return r.summarizer || r.agent;
    return r[role];
  }

  private model(id: string): { model: ModelConfig; provider: ProviderConfig } {
    const s = this.d.settings();
    const model = s.models.find((m) => m.id === id);
    if (!model) throw new Error(`Model '${id}' is not configured. Choose a model in Settings → Models.`);
    const provider = s.providers.find((p) => p.id === model.providerId);
    if (!provider) throw new Error(`Provider '${model.providerId}' of model '${id}' is missing.`);
    return { model, provider };
  }

  private apiConfig(model: ModelConfig, provider: ProviderConfig): OpenAICompatConfig {
    return { baseUrl: provider.baseUrl, apiKey: this.d.apiKey(provider.id), model: model.name, stallMs: this.d.settings().rlm.modelStallSec * 1000, systemAsUser: model.systemAsUser };
  }

  async chat(role: "agent" | "summarizer"): Promise<ChatModelPort> {
    const s = this.d.settings();
    const ids = this.chainFor(role);
    if (!s.failover.enabled || ids.length === 0) return this.chatFor(this.modelIdFor(role));
    const key = `failover:${ids.join("|")}`;
    const cached = this.chatCache.get(key);
    if (cached) return cached;
    const chain = await Promise.all(
      ids.map(async (id) => {
        const m = s.models.find((x) => x.id === id)!;
        const lim = await this.limits(id);
        return { id, label: m.label || m.name, contextLength: lim.contextLength, maxOutputTokens: lim.maxOutputTokens, port: () => this.chatFor(id) };
      }),
    );
    const port = new FailoverChat(chain, {
      settings: () => this.d.settings().failover,
      health: this.health,
      continuePrompt: this.d.continuePrompt,
      log: this.d.log,
    });
    this.chatCache.set(key, port);
    return port;
  }

  /** Chat adapter for one configured model, independent of role assignments. */
  async chatFor(requested: string): Promise<ChatModelPort> {
    const id = this.resolve(requested);
    const cached = this.chatCache.get(id);
    if (cached) return cached;
    const { model, provider } = this.model(id);
    if (model.kind !== "chat") throw new Error(`Model '${id}' is an embedding model, not a chat model.`);
    let port: ChatModelPort;
    const cfg = provider.kind === "local_onnx" ? null : this.apiConfig(model, provider);
    if (!cfg) port = this.d.localChat(model);
    else if (await isOllama(cfg.baseUrl, cfg.apiKey)) port = new OllamaChat(model.id, cfg, () => this.ollamaPlan(model, cfg));
    else port = new OpenAICompatChat(model.id, cfg);
    this.chatCache.set(id, port);
    return port;
  }

  async embedder(): Promise<EmbeddingPort | null> {
    const id = this.modelIdFor("embed");
    if (!id) return null;
    return this.embedderFor(id);
  }

  /** Embedding adapter for one configured model, independent of role assignments. */
  async embedderFor(id: string): Promise<EmbeddingPort | null> {
    const cached = this.embedCache.get(id);
    if (cached) return cached;
    const { model, provider } = this.model(id);
    if (model.kind !== "embed") throw new Error(`Model '${id}' is a chat model, not an embedding model.`);
    let port: EmbeddingPort | null;
    if (provider.kind === "local_onnx") port = await this.d.localEmbed(model);
    else port = new OpenAICompatEmbed(this.apiConfig(model, provider), model.dim ?? 1024);
    if (port) this.embedCache.set(id, port);
    return port;
  }

  async limits(requested: string): Promise<ModelLimitsInfo> {
    const modelId = requested === OMNIROUTE_AUTO_MODEL_ID && !this.d.settings().omniroute.chain.length ? requested : this.resolve(requested);
    const { model, provider } = this.model(modelId);
    const s = this.d.settings();
    if (provider.kind === "local_onnx") {
      const native = this.d.localNativeContext(model);
      if (!native) return { contextLength: null, maxOutputTokens: null };
      const ctx = Math.min(native, model.contextLength ?? s.local.localContextCap, s.local.localContextCap);
      return { contextLength: ctx, maxOutputTokens: Math.min(s.local.localMaxOutputTokens, Math.floor(ctx / 2)) };
    }
    const cfg = this.apiConfig(model, provider);
    if (await isOllama(cfg.baseUrl, cfg.apiKey)) {
      // The window Ollama is told to run; prompts are built for it, never for the model's larger native window.
      const plan = await this.ollamaPlan(model, cfg).catch((e: unknown) => {
        this.d.log?.("ollama window unknown", { model: model.id, error: String(e) });
        return null;
      });
      if (plan) {
        const out = model.maxOutputTokens ? Math.min(model.maxOutputTokens, Math.floor(plan.numCtx / 2)) : Math.floor(plan.numCtx / 4);
        return { contextLength: plan.numCtx, maxOutputTokens: out };
      }
    }
    return { contextLength: model.contextLength, maxOutputTokens: model.maxOutputTokens };
  }

  private ollamaPlan(model: ModelConfig, cfg: OpenAICompatConfig): Promise<OllamaPlan> {
    const s = this.d.settings().local;
    return ollamaPlan(cfg, { cap: s.ollamaContextCap, kvRamShare: s.ollamaKvRamShare, configured: model.contextLength }).then((p) => {
      this.d.log?.("ollama window", { model: model.id, numCtx: p.numCtx, native: p.native, why: p.why });
      return p;
    });
  }

  /** Auto-detect limits for an API model (vLLM max_model_len, OpenRouter metadata, Ollama). */
  async detect(modelId: string): Promise<ModelLimitsInfo> {
    if (modelId === OMNIROUTE_AUTO_MODEL_ID) return this.limits(modelId);
    const { model, provider } = this.model(modelId);
    if (provider.kind === "local_onnx") return this.limits(modelId);
    return detectOpenAICompatLimits({ baseUrl: provider.baseUrl, apiKey: this.d.apiKey(provider.id), model: model.name });
  }
}
