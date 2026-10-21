import { z } from "zod";

/** Model hub: download any Hugging Face repo, then deploy it as an OpenAI-compatible endpoint. */
export const HubSettings = z.object({
  /** An existing llama.cpp checkout or install (folder holding llama-server); empty = the one the IDE builds. */
  llamaCppDir: z.string().default(""),
  /** GitHub repository llama.cpp is downloaded (prebuilt) or cloned (source build) from. */
  llamaCppRepo: z.string().regex(/^[\w.-]+\/[\w.-]+$/).default("ggml-org/llama.cpp"),
  /** Release tag to download, such as b11796; empty = the newest build that has an archive for this system. */
  llamaCppRelease: z.string().default(""),
  /** Python used for deployment tools; empty = the IDE's own environment (created on first use). */
  python: z.string().default(""),
  /** Files downloaded at the same time. */
  downloadConcurrency: z.number().int().min(1).max(16).default(4),
  /** Default context for a deployment is the model's own, capped here so the KV cache fits in memory. */
  contextCap: z.number().int().min(512).default(32768),
  /** How long a server may take to load its model before the deployment is marked failed. */
  startTimeoutSec: z.number().int().min(30).default(900),
  /** Restart deployments that were running when the IDE closed. */
  restoreDeployments: z.boolean().default(true),
});
export type HubSettings = z.infer<typeof HubSettings>;

export type HubKind = "chat" | "embed";
export type HubTargetId = "llamacpp" | "transformers" | "onnx";

export interface HubFileView {
  path: string;
  size: number;
  lfs: boolean;
}

export interface ModelProfileView {
  /** What the model is for, as far as its own metadata says; "other" = not a chat or embedding model. */
  kind: HubKind | "other" | "unknown";
  kindReason: string;
  pipelineTag: string | null;
  libraryName: string | null;
  architectures: string[];
  modelType: string | null;
  formats: string[];
  ggufFiles: string[];
  onnxFiles: string[];
  hasHfWeights: boolean;
  contextLength: number | null;
  hasChatTemplate: boolean;
  remoteCode: boolean;
  /** quant_method of a model published already quantized (gptq, awq, fp8, ...). */
  quantizedWith: string | null;
  parameters: number | null;
  notes: string[];
}

export interface HubRepoView {
  repo: string;
  revision: string;
  sha: string;
  gated: false | "auto" | "manual";
  private: boolean;
  access: "ok" | "needs-token" | "needs-approval";
  accessDetail: string;
  tags: string[];
  files: HubFileView[];
  totalBytes: number;
  defaultSelection: string[];
  profile: ModelProfileView;
}

export interface HubJobView {
  id: string;
  repo: string;
  revision: string;
  sha: string;
  dest: string;
  status: "queued" | "running" | "paused" | "verifying" | "done" | "error" | "cancelled";
  files: number;
  filesDone: number;
  bytesDone: number;
  bytesTotal: number;
  speedBps: number;
  etaSec: number | null;
  current: string[];
  error: string | null;
  startedAt: number;
  finishedAt: number | null;
}

export interface HubModelView {
  id: string;
  repo: string;
  revision: string;
  sha: string;
  dir: string;
  bytes: number;
  files: string[];
  downloadedAt: number;
  profile: ModelProfileView;
}

export interface HubOption {
  id: string;
  label: string;
  detail?: string;
}

export interface DeployTargetView {
  id: HubTargetId;
  label: string;
  /** The model can be deployed this way (tools aside). */
  available: boolean;
  reason: string;
  /** Tools to install first, by id: "llamacpp", "python:<pkg>". */
  needs: string[];
  sources: HubOption[];
  /** Quantizations per source id. */
  quantizations: Record<string, HubOption[]>;
  devices: Array<"cpu" | "cuda" | "gpu">;
  kinds: HubKind[];
  dtypes: string[];
}

export const HubDeployRequest = z.object({
  modelId: z.string().min(1),
  target: z.enum(["llamacpp", "transformers", "onnx"]),
  source: z.string().min(1),
  quant: z.string().min(1),
  kind: z.enum(["chat", "embed"]),
  /** "gpu": llama.cpp's Vulkan or Metal build. */
  device: z.enum(["cpu", "cuda", "gpu"]).default("cpu"),
  dtype: z.string().default("auto"),
  contextLength: z.number().int().min(0).default(0),
  trustRemoteCode: z.boolean().default(false),
  label: z.string().max(120).default(""),
});
export type HubDeployRequest = z.infer<typeof HubDeployRequest>;

export interface HubStepView {
  label: string;
  status: "todo" | "running" | "done" | "error" | "skipped";
  detail?: string;
}

export interface HubDeploymentView {
  id: string;
  modelId: string;
  repo: string;
  request: HubDeployRequest;
  status: "preparing" | "starting" | "running" | "stopped" | "error";
  steps: HubStepView[];
  artifact: string | null;
  baseUrl: string | null;
  servedName: string;
  providerId: string | null;
  settingsModelId: string | null;
  contextLength: number | null;
  dim: number | null;
  error: string | null;
  logTail: string;
  createdAt: number;
}

/** "main" serves Transformers and converts to GGUF; "onnx" is separate because optimum pins its own transformers. */
export type HubPythonEnv = "main" | "onnx";

export interface HubPythonView {
  path: string;
  version: string;
  own: boolean;
  packages: Record<string, string | null>;
  cuda: boolean;
  gpu: { name: string; freeMb: number; totalMb: number } | null;
  transformersModelTypes: number;
  onnxExportModelTypes: number;
  onnxQuantizers: string[];
}

export interface HubToolsView {
  python: HubPythonView | null;
  pythonError: string | null;
  /** True while Python is being inspected (the first check at startup takes a few seconds). */
  probing: boolean;
  /** Null while ONNX shares the main environment. */
  onnxPython: HubPythonView | null;
  llamaCpp: {
    dir: string | null;
    server: string | null;
    quantize: string | null;
    convert: string | null;
    backend: "cpu" | "cuda" | "vulkan" | "metal" | null;
    version: string | null;
    error: string | null;
    quantTypes: HubOption[];
    convertArchitectures: number;
  };
  cudaToolkit: boolean;
  installing: { tool: string; log: string } | null;
  hfToken: boolean;
}

export interface HubStateView {
  jobs: HubJobView[];
  models: HubModelView[];
  deployments: HubDeploymentView[];
  tools: HubToolsView;
}
