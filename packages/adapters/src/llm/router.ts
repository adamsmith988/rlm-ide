import type { ModelConfig, ProviderConfig, Settings } from "@rlm/contracts";
import type { ChatModelPort, EmbeddingPort, ModelCatalogPort, ModelLimitsInfo, ModelRouterPort } from "@rlm/core";
import { detectOpenAICompatLimits, OpenAICompatChat, OpenAICompatEmbed, type OpenAICompatConfig } from "./openai_compat";

export interface RouterDeps {
  settings: () => Settings;
  apiKey: (providerId: string) => string | null;
  /** Built-in local chat model (FrogNano via the inference host). */
  localChat: (model: ModelConfig) => ChatModelPort;
  /** Built-in local embedder (bge-m3 in-process). Null when not installed. */
  localEmbed: (model: ModelConfig) => Promise<EmbeddingPort | null>;
  /** Max context the local model itself supports (from its config.json), null if not installed. */
  localNativeContext: (model: ModelConfig) => number | null;
}

/** Picks an adapter per role: agent, summarizer, embed. Multi-provider: local ONNX or any
 * OpenAI-compatible API, chosen in Settings → Models. */
export class ProviderRouter implements ModelRouterPort, ModelCatalogPort {
  private readonly chatCache = new Map<string, ChatModelPort>();
  private readonly embedCache = new Map<string, EmbeddingPort>();

  constructor(private readonly d: RouterDeps) {}

  invalidate(): void {
    this.chatCache.clear();
    this.embedCache.clear();
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
    return { baseUrl: provider.baseUrl, apiKey: this.d.apiKey(provider.id), model: model.name, stallMs: this.d.settings().rlm.modelStallSec * 1000 };
  }

  async chat(role: "agent" | "summarizer"): Promise<ChatModelPort> {
    return this.chatFor(this.modelIdFor(role));
  }

  /** Chat adapter for one configured model, independent of role assignments. */
  async chatFor(id: string): Promise<ChatModelPort> {
    const cached = this.chatCache.get(id);
    if (cached) return cached;
    const { model, provider } = this.model(id);
    if (model.kind !== "chat") throw new Error(`Model '${id}' is an embedding model, not a chat model.`);
    let port: ChatModelPort;
    if (provider.kind === "local_onnx") port = this.d.localChat(model);
    else port = new OpenAICompatChat(model.id, this.apiConfig(model, provider));
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

  async limits(modelId: string): Promise<ModelLimitsInfo> {
    const { model, provider } = this.model(modelId);
    const s = this.d.settings();
    if (provider.kind === "local_onnx") {
      const native = this.d.localNativeContext(model);
      if (!native) return { contextLength: null, maxOutputTokens: null };
      const ctx = Math.min(native, model.contextLength ?? s.local.localContextCap, s.local.localContextCap);
      return { contextLength: ctx, maxOutputTokens: Math.min(s.local.localMaxOutputTokens, Math.floor(ctx / 2)) };
    }
    return { contextLength: model.contextLength, maxOutputTokens: model.maxOutputTokens };
  }

  /** Auto-detect limits for an API model (vLLM max_model_len, OpenRouter metadata, Ollama). */
  async detect(modelId: string): Promise<ModelLimitsInfo> {
    const { model, provider } = this.model(modelId);
    if (provider.kind === "local_onnx") return this.limits(modelId);
    return detectOpenAICompatLimits({ baseUrl: provider.baseUrl, apiKey: this.d.apiKey(provider.id), model: model.name });
  }
}
