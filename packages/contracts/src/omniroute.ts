/** State of the IDE-managed OmniRoute gateway, pushed to the renderer on every change. */

export type OmniRoutePhase = "not_installed" | "installing" | "stopped" | "starting" | "running" | "error";

export interface OmniRouteProviderView {
  id: string;
  name: string;
  website: string | null;
  /** Works without an account or key. */
  keyless: boolean;
  /** Has a connection in the gateway (API-key providers) or needs none (keyless). */
  connected: boolean;
}

export interface OmniRouteConnectionView {
  id: string;
  provider: string;
  name: string;
  active: boolean;
  status: string;
  lastError: string | null;
}

export interface OmniRouteModelView {
  /** Gateway model id, e.g. "cfp/zai-org/glm-5.2". */
  id: string;
  provider: string;
  name: string;
  contextLength: number | null;
  maxOutputTokens: number | null;
  toolCalling: boolean;
  /** Last probe: null = not tested. */
  ok: boolean | null;
  latencyMs: number | null;
  error: string | null;
  /** Only follows instructions placed in the user message (the provider drops the system role); null = untested. */
  systemAsUser: boolean | null;
  /** Measured input limit when it is below the test size (web-playground providers); null = the advertised window holds. */
  inputLimitChars: number | null;
  /** Did real agent work through `repl` cells in every protocol test (false: tends to answer without its tools). */
  agentReady: boolean | null;
  /** In the IDE's failover chain, in this order. */
  chainIndex: number | null;
}

export interface OmniRouteView {
  phase: OmniRoutePhase;
  version: string | null;
  node: { path: string | null; version: string | null; ok: boolean; detail: string };
  baseUrl: string;
  dashboardUrl: string;
  /** What the service is doing right now, for the progress line. */
  busy: string | null;
  error: string | null;
  logTail: string;
  providers: OmniRouteProviderView[];
  connections: OmniRouteConnectionView[];
  models: OmniRouteModelView[];
  comboName: string;
}
