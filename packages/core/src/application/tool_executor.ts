import type { ToolCallInfo, ToolSettings } from "@rlm/contracts";
import { callKey, ToolArgError, validateArgs, type ToolDescriptor } from "../domain/tools";
import type { ChatModelPort, PromptsPort, ToolContext, ToolRegistryPort } from "../ports";
import { render } from "../ports";
import { newId } from "./memory_service";

export interface ToolResult {
  ok: boolean;
  result: unknown;
  error?: string;
  durationMs: number;
  deduped?: boolean;
}

export interface ApprovalRequester {
  /** Emits an approval card and resolves when the user decides (or the turn stops). */
  request(turnId: string, call: ToolCallInfo, preview: string | undefined, signal: AbortSignal): Promise<{ approved: boolean; remember: "once" | "session" | "workspace" }>;
}

export interface TurnToolScope {
  turnId: string;
  chatId: string;
  /** Host-side allow list. A call outside it is refused regardless of what the kernel asks. */
  toolIds: ReadonlySet<string>;
  ctx: ToolContext;
  /** Conversation text for the optional action check. */
  conversation: () => string;
  onCall?: (call: ToolCallInfo) => void;
  onResult?: (callId: string, r: ToolResult) => void;
  onEditApplied?: (edit: { editId: string; path: string; before: string; after: string }) => void;
}

/** One executor for every tool call: validation, `tool_ids` enforcement, approvals,
 * de-duplication of repeated side-effecting calls, timeouts. */
export class ToolExecutor {
  /** Per turn: the last side-effecting call. Only an immediate repeat is de-duplicated, because any other
   * side effect in between (an edit, a command) may change what the repeat would do. */
  private readonly turnDedup = new Map<string, { key: string; promise: Promise<ToolResult> } | null>();
  private readonly sessionAllow = new Map<string, Set<string>>();

  constructor(
    private readonly deps: {
      registry: ToolRegistryPort;
      approvals: ApprovalRequester;
      settings: () => ToolSettings;
      persistWorkspaceAllow: (toolId: string) => void;
      prompts: PromptsPort;
      checker: () => Promise<ChatModelPort>;
    },
  ) {}

  beginToolTurn(turnId: string): void {
    this.turnDedup.set(turnId, null);
  }

  endToolTurn(turnId: string): void {
    this.turnDedup.delete(turnId);
  }

  async execute(toolId: string, rawArgs: unknown, scope: TurnToolScope): Promise<ToolResult> {
    const started = Date.now();
    const fail = (error: string): ToolResult => ({ ok: false, result: null, error, durationMs: Date.now() - started });

    if (!scope.toolIds.has(toolId)) return fail(`tool '${toolId}' is not available in this session`);
    const impl = this.deps.registry.get(toolId);
    if (!impl) return fail(`unknown tool '${toolId}'`);

    let args: Record<string, unknown>;
    try {
      args = validateArgs(impl.descriptor.parameters, rawArgs ?? {});
    } catch (e) {
      return fail(e instanceof ToolArgError ? e.message : String(e));
    }

    const call: ToolCallInfo = { callId: newId("call"), toolId, args };
    scope.onCall?.(call);

    const run = async (): Promise<ToolResult> => {
      if (impl.precheck) {
        const refusal = await impl.precheck(args, scope.ctx).catch(() => null);
        if (refusal) return { ...fail(refusal), durationMs: Date.now() - started };
      }
      if (impl.descriptor.sideEffects) {
        const gate = await this.gate(impl.descriptor, call, scope);
        if (gate) return { ...fail(gate), durationMs: Date.now() - started };
      }
      return this.runWithTimeout(impl.descriptor, () => impl.run(args, scope.ctx), started, scope.ctx.signal);
    };

    let promise: Promise<ToolResult>;
    let deduped = false;
    if (impl.descriptor.sideEffects && !impl.descriptor.repeatable && this.turnDedup.has(scope.turnId)) {
      const key = callKey(toolId, args);
      const last = this.turnDedup.get(scope.turnId);
      if (last && last.key === key) {
        promise = last.promise;
        deduped = true;
      } else {
        promise = run();
        this.turnDedup.set(scope.turnId, { key, promise });
      }
    } else {
      promise = run();
      if (impl.descriptor.sideEffects && this.turnDedup.has(scope.turnId)) this.turnDedup.set(scope.turnId, null);
    }

    let before: { path: string; before: string; after: string } | null = null;
    if (!deduped && impl.descriptor.editsFiles && impl.preview) {
      before = await impl.preview(args, scope.ctx).catch(() => null);
    }
    const r = { ...(await promise), deduped: deduped || undefined };
    if (r.ok && !deduped && before && before.before !== before.after) {
      scope.onEditApplied?.({ editId: newId("edit"), ...before });
    }
    scope.onResult?.(call.callId, r);
    return r;
  }

  /** Returns an error string when the call must not run, null when it may. */
  private async gate(d: ToolDescriptor, call: ToolCallInfo, scope: TurnToolScope): Promise<string | null> {
    const s = this.deps.settings();
    if (d.id === "shell.run") {
      const cmd = String(call.args.command ?? "");
      if (s.shellDenyPatterns.some((p) => safeRegex(p)?.test(cmd))) return `command blocked by deny list: ${cmd}`;
    }
    if (s.actionCheck) {
      const verdict = await this.actionCheck(d, call, scope);
      if (verdict) return verdict;
    }
    if (this.isPreApproved(d, call, scope, s)) return null;
    let preview: string | undefined;
    const impl = this.deps.registry.get(d.id);
    if (d.editsFiles && impl?.preview) {
      const p = await impl.preview(call.args, scope.ctx).catch((e: unknown) => ({ path: String(call.args.path ?? ""), before: "", after: `preview failed: ${String(e)}` }));
      preview = JSON.stringify(p);
    }
    const decision = await this.deps.approvals.request(scope.turnId, call, preview, scope.ctx.signal);
    if (!decision.approved) return "the user rejected this action. Do not retry it; ask the user how to proceed.";
    if (decision.remember === "session") this.allowForSession(scope.chatId, d.id);
    if (decision.remember === "workspace") this.deps.persistWorkspaceAllow(d.id);
    return null;
  }

  private isPreApproved(d: ToolDescriptor, call: ToolCallInfo, scope: TurnToolScope, s: ToolSettings): boolean {
    if (d.editsFiles && s.autoApplyEdits) return true;
    if (s.workspaceAllow.includes("*") || s.workspaceAllow.includes(d.id)) return true;
    if (this.sessionAllow.get(scope.chatId)?.has(d.id)) return true;
    if (d.id === "shell.run") {
      const cmd = String(call.args.command ?? "");
      if (s.shellAllowPatterns.some((p) => safeRegex(p)?.test(cmd))) return true;
    }
    return false;
  }

  allowForSession(chatId: string, toolId: string): void {
    const set = this.sessionAllow.get(chatId) ?? new Set<string>();
    set.add(toolId);
    this.sessionAllow.set(chatId, set);
  }

  private async actionCheck(d: ToolDescriptor, call: ToolCallInfo, scope: TurnToolScope): Promise<string | null> {
    const prompt = render(this.deps.prompts.get("action_check.md"), {
      clock: new Date().toISOString(),
      conversation: scope.conversation(),
      observations: "(see conversation)",
      tool: d.id,
      description: d.description,
      arguments: JSON.stringify(call.args),
    });
    const model = await this.deps.checker();
    let out = "";
    for await (const c of model.stream({ messages: [{ role: "user", content: prompt }], maxTokens: 400, temperature: 0, enableThinking: false })) {
      if (c.type === "text") out += c.text;
    }
    const m = /VERDICT:\s*(YES|NO[^\n]*)/i.exec(out);
    if (m && m[1]!.toUpperCase().startsWith("YES")) return null;
    return `action check refused: ${m?.[1] ?? "no verdict"}`;
  }

  private async runWithTimeout(d: ToolDescriptor, fn: () => Promise<unknown>, started: number, signal: AbortSignal): Promise<ToolResult> {
    let timer: ReturnType<typeof setTimeout> | undefined;
    try {
      const result = await Promise.race([
        fn(),
        new Promise<never>((_, rej) => {
          timer = setTimeout(() => rej(new Error(`timed out after ${d.timeoutSec}s`)), d.timeoutSec * 1000);
        }),
        new Promise<never>((_, rej) => {
          if (signal.aborted) rej(new Error("turn stopped"));
          signal.addEventListener("abort", () => rej(new Error("turn stopped")), { once: true });
        }),
      ]);
      return { ok: true, result, durationMs: Date.now() - started };
    } catch (e) {
      return { ok: false, result: null, error: e instanceof Error ? e.message : String(e), durationMs: Date.now() - started };
    } finally {
      if (timer) clearTimeout(timer);
    }
  }
}

function safeRegex(p: string): RegExp | null {
  try {
    return new RegExp(p);
  } catch {
    return null;
  }
}
