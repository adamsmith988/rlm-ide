import { useEffect, useState } from "react";
import type { AppInfo, Settings } from "@rlm/contracts";
import { call } from "../../api";
import { useApp } from "../../store";
import { Circles } from "../../ui/LineArt";
import { Lines, Num, Ratio, Row, Section, Select, Text, Toggle } from "./fields";
import { ModelsSection } from "./ModelsSection";

const CATEGORIES = [
  ["models", "Models"],
  ["memory", "Memory"],
  ["tools", "Tools & safety"],
  ["agent", "Agent & compaction"],
  ["local", "Local inference"],
  ["appearance", "Appearance"],
  ["skills", "Skills & MCP"],
  ["about", "About"],
] as const;
type Cat = (typeof CATEGORIES)[number][0];

export function SettingsScreen() {
  const s = useApp((x) => x.settings);
  const [cat, setCat] = useState<Cat>("models");
  if (!s) return null;
  return (
    <div className="h-full overflow-y-auto">
      <div className="relative overflow-hidden border-b border-line">
        <Circles count={2} size={700} />
        <div className="relative mx-auto max-w-[920px] px-10 pb-6 pt-10">
          <div className="smallcaps">Preferences</div>
          <h1 className="heading m-0 mt-1 text-[44px] leading-none">Settings</h1>
          <div className="mt-6 grid grid-cols-4 gap-2.5 lg:grid-cols-8">
            {CATEGORIES.map(([id, label]) => (
              <button key={id} className={`tile ${cat === id ? "on" : ""}`} onClick={() => setCat(id)}>
                {label}
              </button>
            ))}
          </div>
        </div>
      </div>
      <div className="mx-auto max-w-[920px] px-10 py-8 fade-in" key={cat}>
        {cat === "models" && <ModelsSection s={s} />}
        {cat === "memory" && <MemorySection s={s} />}
        {cat === "tools" && <ToolsSection s={s} />}
        {cat === "agent" && <AgentSection s={s} />}
        {cat === "local" && <LocalSection s={s} />}
        {cat === "appearance" && <AppearanceSection s={s} />}
        {cat === "skills" && <SkillsMcpSection s={s} />}
        {cat === "about" && <AboutSection s={s} />}
      </div>
    </div>
  );
}

function usePatch() {
  const update = useApp((x) => x.updateSettings);
  return (patch: Record<string, unknown>) => void update(patch);
}

function MemorySection({ s }: { s: Settings }) {
  const patch = usePatch();
  const m = s.memory;
  const set = (p: Partial<Settings["memory"]>) => patch({ memory: p });
  return (
    <Section title="Memory" hint="Short-term memory recalls earlier parts of the same chat. Long-term memory keeps durable facts across chats.">
      <Row label="Enable memory" hint="When off, nothing is stored or recalled.">
        <Toggle value={m.enabled} onChange={(v) => set({ enabled: v })} />
      </Row>
      <Row label="Long-term scope" hint="Per project: facts about you follow you everywhere, facts about a project stay in that project. All projects: every fact is recalled everywhere.">
        <Select value={m.longTermScope} options={[["workspace", "Per project"], ["user", "All projects"]]} onChange={(v) => set({ longTermScope: v })} />
      </Row>
      <Row label="Only learn what I said" hint="Facts must be grounded in your messages, not the assistant's.">
        <Toggle value={m.factRequireUserGrounding} onChange={(v) => set({ factRequireUserGrounding: v })} />
      </Row>
      <Row label="Minimum similarity" hint="Recall threshold (cosine).">
        <Ratio value={m.minScore} onChange={(v) => set({ minScore: v })} />
      </Row>
      <Row label="Same-chat recalls" hint="Max chunks recalled from this chat.">
        <Num value={m.sessionLimit} min={0} max={50} onChange={(v) => set({ sessionLimit: v ?? 0 })} />
      </Row>
      <Row label="Long-term recalls" hint="Max facts recalled per turn.">
        <Num value={m.userLimit} min={0} max={50} onChange={(v) => set({ userLimit: v ?? 0 })} />
      </Row>
      <Row label="Duplicate threshold" hint="New facts this similar to an existing one update it instead.">
        <Ratio value={m.factDedupScore} onChange={(v) => set({ factDedupScore: v })} />
      </Row>
      <Row label="Always recall" hint="When you have at most this many facts, all of them are included.">
        <Num value={m.alwaysRecallFacts} min={0} max={50} onChange={(v) => set({ alwaysRecallFacts: v ?? 0 })} />
      </Row>
      <Row label="New facts per turn" hint="How many facts one exchange can add.">
        <Num value={m.factMaxAddsPerTurn} min={0} max={20} onChange={(v) => set({ factMaxAddsPerTurn: v ?? 0 })} />
      </Row>
      <Row label="Fact length" hint="Longer text is treated as pasted content, not a fact.">
        <Num value={m.factMaxChars} min={40} max={2000} onChange={(v) => set({ factMaxChars: v ?? 200 })} suffix="chars" />
      </Row>
      <Row label="Skip short messages" hint="Messages shorter than this are not distilled into facts.">
        <Num value={m.factMinUserChars} min={0} max={200} onChange={(v) => set({ factMinUserChars: v ?? 0 })} suffix="chars" />
      </Row>
      <Row label="Recent turns kept" hint="How many recent exchanges stay in the prompt before older ones are recalled from memory.">
        <Num value={m.historyKeep} min={0} max={50} onChange={(v) => set({ historyKeep: v ?? 0 })} />
      </Row>
      <Row label="Memory in the prompt" hint="Character budget for recalled facts and chat excerpts.">
        <Num value={m.promptMaxChars} min={200} max={20000} onChange={(v) => set({ promptMaxChars: v ?? 4000 })} suffix="chars" />
      </Row>
      <div className="mt-3 text-[12px] leading-relaxed text-dim">
        Each project keeps its own chats and project facts. Facts about you (not tied to a folder) follow you into every project. Opening another folder does not load the previous project's chats.
      </div>
    </Section>
  );
}

function SkillsMcpSection({ s }: { s: Settings }) {
  const patch = usePatch();
  const skills = s.skills;
  const mcp = s.mcp;
  const setSkill = (id: string, p: Partial<(typeof skills)[number]>) => patch({ skills: skills.map((x) => (x.id === id ? { ...x, ...p } : x)) });
  return (
    <>
      <Section title="Skills" hint="Enabled skills are in the agent's instructions. Type /name in the chat to insert one. Names are letters, numbers, dots, and dashes.">
        {skills.map((sk) => (
          <div key={sk.id} className="mb-3 rounded-[12px] border border-line p-3" data-testid="skill-editor">
            <div className="mb-2 flex items-center gap-2">
              <Text value={sk.name} width={180} mono onChange={(v) => setSkill(sk.id, { name: v.trim().replace(/[^\w.-]/g, "").slice(0, 40) || sk.name })} />
              <Text value={sk.description} width={280} placeholder="What this skill is for" onChange={(v) => setSkill(sk.id, { description: v })} />
              <Toggle value={sk.enabled} onChange={(v) => setSkill(sk.id, { enabled: v })} />
              <button className="btn btn-sm btn-danger ml-auto" onClick={() => patch({ skills: skills.filter((x) => x.id !== sk.id) })}>
                Remove
              </button>
            </div>
            <textarea className="input min-h-[88px] w-full p-2 font-code text-[12px]" value={sk.body} placeholder="Instructions inserted when the user types /name" onChange={(e) => setSkill(sk.id, { body: e.target.value })} />
          </div>
        ))}
        <button
          className="btn btn-sm"
          onClick={() => patch({ skills: [...skills, { id: crypto.randomUUID(), name: `skill-${skills.length + 1}`, description: "", body: "", enabled: true }] })}
        >
          Add skill
        </button>
      </Section>
      <Section title="MCP servers" hint="Each server is a local command (not a shell). The agent uses mcp.list and mcp.call. Do not put secrets in the environment box; it is saved in settings.">
        {mcp.map((sv) => (
          <div key={sv.id} className="mb-3 rounded-[12px] border border-line p-3" data-testid="mcp-editor">
            <div className="mb-2 flex flex-wrap items-center gap-2">
              <Text value={sv.name} width={160} placeholder="Name" onChange={(v) => patch({ mcp: mcp.map((x) => (x.id === sv.id ? { ...x, name: v } : x)) })} />
              <Text value={sv.command} width={220} mono placeholder="npx" onChange={(v) => patch({ mcp: mcp.map((x) => (x.id === sv.id ? { ...x, command: v } : x)) })} />
              <Toggle value={sv.enabled} onChange={(v) => patch({ mcp: mcp.map((x) => (x.id === sv.id ? { ...x, enabled: v } : x)) })} />
              <button className="btn btn-sm btn-danger ml-auto" onClick={() => patch({ mcp: mcp.filter((x) => x.id !== sv.id) })}>
                Remove
              </button>
            </div>
            <Text value={sv.args} width={520} mono placeholder="Arguments, quoted when they contain spaces" onChange={(v) => patch({ mcp: mcp.map((x) => (x.id === sv.id ? { ...x, args: v } : x)) })} />
            <textarea
              className="input mt-2 min-h-[64px] w-full p-2 font-code text-[12px]"
              value={sv.env}
              placeholder={"KEY=value\n# no secrets"}
              onChange={(e) => patch({ mcp: mcp.map((x) => (x.id === sv.id ? { ...x, env: e.target.value } : x)) })}
            />
          </div>
        ))}
        <button
          className="btn btn-sm"
          onClick={() => patch({ mcp: [...mcp, { id: crypto.randomUUID(), name: `server-${mcp.length + 1}`, command: "npx", args: "", env: "", enabled: false }] })}
        >
          Add MCP server
        </button>
      </Section>
    </>
  );
}

function ToolsSection({ s }: { s: Settings }) {
  const patch = usePatch();
  const t = s.tools;
  const set = (p: Partial<Settings["tools"]>) => patch({ tools: p });
  const r = s.rlm;
  const setR = (p: Partial<Settings["rlm"]>) => patch({ rlm: p });
  return (
    <>
      <Section title="Approvals" hint="Reads never ask. Writes, deletes, shell commands, and git commits ask first unless you allow them.">
        <Row label="Default approval" hint="How long an 'allow' lasts when you don't pick explicitly.">
          <Select value={t.approvalMode} options={[["ask", "Ask every time"], ["session", "Per chat"], ["workspace", "Per workspace"]]} onChange={(v) => set({ approvalMode: v })} />
        </Row>
        <Row label="Auto-apply edits" hint="Skip approval for file edits. You can still review and revert every edit.">
          <Toggle value={t.autoApplyEdits} onChange={(v) => set({ autoApplyEdits: v })} />
        </Row>
        <Row label="Second-opinion check" hint="Ask the summarizer model to sanity-check risky actions before showing them.">
          <Toggle value={t.actionCheck} onChange={(v) => set({ actionCheck: v })} />
        </Row>
        <Row label="Always allowed tools" hint={'Tool ids auto-approved in this workspace. "*" allows everything.'}>
          <Lines value={t.workspaceAllow} onChange={(v) => set({ workspaceAllow: v })} placeholder="fs.write_file" />
        </Row>
      </Section>
      <Section title="Shell">
        <Row label="Allowed without asking" hint="Regex patterns for commands that run without approval.">
          <Lines value={t.shellAllowPatterns} onChange={(v) => set({ shellAllowPatterns: v })} />
        </Row>
        <Row label="Always blocked" hint="Regex patterns that are refused outright.">
          <Lines value={t.shellDenyPatterns} onChange={(v) => set({ shellDenyPatterns: v })} />
        </Row>
        <Row label="Command timeout">
          <Num value={t.shellTimeoutSec} min={1} onChange={(v) => set({ shellTimeoutSec: v ?? 300 })} suffix="sec" />
        </Row>
      </Section>
      <Section title="Python sandbox">
        <Row label="Block network in the kernel" hint="Linux bubblewrap only. Tools still reach the network through the host.">
          <Toggle value={r.sandboxNoNetwork} onChange={(v) => setR({ sandboxNoNetwork: v })} />
        </Row>
        <Row label="Kernel memory limit">
          <Num value={r.kernelMemoryMb} min={256} onChange={(v) => setR({ kernelMemoryMb: v ?? 4096 })} suffix="MB" />
        </Row>
        <Row label="Cell timeout">
          <Num value={r.cellTimeoutSec} min={5} onChange={(v) => setR({ cellTimeoutSec: v ?? 120 })} suffix="sec" />
        </Row>
        <Row label="Model stall timeout" hint="An API model request fails when the server sends nothing for this long.">
          <Num value={r.modelStallSec} min={10} onChange={(v) => setR({ modelStallSec: v ?? 300 })} suffix="sec" />
        </Row>
      </Section>
    </>
  );
}

function AgentSection({ s }: { s: Settings }) {
  const patch = usePatch();
  const c = s.compact;
  const r = s.rlm;
  const setC = (p: Partial<Settings["compact"]>) => patch({ compact: p });
  const setR = (p: Partial<Settings["rlm"]>) => patch({ rlm: p });
  return (
    <>
      <Section title="Agent loop">
        <Row label="Max steps per turn" hint="Model and REPL iterations before the agent must answer.">
          <Num value={r.maxSteps} min={1} max={1000} onChange={(v) => setR({ maxSteps: v ?? 80 })} />
        </Row>
        <Row label="Check work before answering" hint="After editing files, the agent re-reads your request once and fixes anything it missed.">
          <Toggle value={r.selfReview} onChange={(v) => setR({ selfReview: v })} />
        </Row>
        <Row label="Sub-agent depth" hint="0 disables spawn_agents.">
          <Num value={r.maxSubagentDepth} min={0} max={5} onChange={(v) => setR({ maxSubagentDepth: v ?? 2 })} />
        </Row>
        <Row label="Parallel sub-agents">
          <Num value={r.maxConcurrentChildren} min={1} max={16} onChange={(v) => setR({ maxConcurrentChildren: v ?? 3 })} />
        </Row>
        <Row label="Cell output limit" hint="Longer outputs are truncated with a marker.">
          <Num value={r.cellOutputLimitChars} min={1024} onChange={(v) => setR({ cellOutputLimitChars: v ?? 16000 })} suffix="chars" width={110} />
        </Row>
      </Section>
      <Section title="Runaway replies" hint="A step whose reasoning runs past these limits, or that keeps repeating itself, is stopped and the model is told to act.">
        <Row label="Reasoning per step" hint="Tokens of reasoning before the step is stopped.">
          <Num value={r.thinkingMaxTokens} min={256} onChange={(v) => setR({ thinkingMaxTokens: v ?? 8192 })} suffix="tokens" width={110} />
        </Row>
        <Row label="Reasoning time per step" hint="Slow local models reach the token limit only after a long time.">
          <Num value={r.thinkingMaxSec} min={10} onChange={(v) => setR({ thinkingMaxSec: v ?? 180 })} suffix="sec" />
        </Row>
        <Row label="Stops in a row before giving up">
          <Num value={r.maxCutsInARow} min={1} max={20} onChange={(v) => setR({ maxCutsInARow: v ?? 3 })} />
        </Row>
        <Row label="Hand large messages to the REPL" hint="Messages larger than this share of the context window go to the REPL as user_message; the agent reads them in parts.">
          <Ratio value={r.largeInputShare} onChange={(v) => setR({ largeInputShare: v })} min={0.05} max={0.9} />
        </Row>
      </Section>
      <Section title="Compaction" hint="When history fills its share of the context window, older turns are folded into a structured checkpoint.">
        <Row label="History share of window">
          <Ratio value={c.historyShare} onChange={(v) => setC({ historyShare: v })} min={0.1} max={0.95} />
        </Row>
        <Row label="Compact at" hint="Share of the history budget that triggers compaction.">
          <Ratio value={c.triggerRatio} onChange={(v) => setC({ triggerRatio: v })} min={0.2} />
        </Row>
        <Row label="Keep recent verbatim" hint="Share of the history budget kept word-for-word.">
          <Ratio value={c.keepRecentRatio} onChange={(v) => setC({ keepRecentRatio: v })} min={0.05} max={0.9} />
        </Row>
        <Row label="Output reserve cap">
          <Ratio value={c.maxOutputRatio} onChange={(v) => setC({ maxOutputRatio: v })} min={0.05} max={0.5} />
        </Row>
        <Row label="In-turn compaction" hint="Compact the REPL transcript when one turn nears the window.">
          <Ratio value={r.compactTriggerRatio} onChange={(v) => setR({ compactTriggerRatio: v })} min={0.3} max={0.98} />
        </Row>
      </Section>
    </>
  );
}

function LocalSection({ s }: { s: Settings }) {
  const patch = usePatch();
  const l = s.local;
  const set = (p: Partial<Settings["local"]>) => patch({ local: p });
  const inf = useApp((x) => x.status.inference);
  return (
    <Section title="Local inference" hint="Settings for the built-in FrogNano model.">
      <Row label="Device" hint={inf ? `Current: ${inf.device} · ${inf.status}${inf.detail ? ` — ${inf.detail}` : ""}` : "Starts on first use."}>
        <Select value={l.preferredDevice} options={[["auto", "Auto (GPU, then CPU)"], ["webgpu", "GPU (WebGPU)"], ["cpu", "CPU"]]} onChange={(v) => set({ preferredDevice: v })} />
      </Row>
      <Row label="CPU threads" hint="Used when the model runs on the CPU. 0 = one per physical core, usually fastest.">
        <Num value={l.cpuThreads} min={0} onChange={(v) => set({ cpuThreads: v ?? 0 })} width={100} />
      </Row>
      <Row label="GPU memory headroom" hint="Free GPU memory needed beyond the model's weights before it loads on an NVIDIA GPU; with less, Auto uses the CPU.">
        <Num value={l.gpuHeadroomMB} min={0} onChange={(v) => set({ gpuHeadroomMB: v ?? 1536 })} suffix="MB" width={100} />
      </Row>
      <Row label="Context cap" hint="Upper bound on the local context. Larger needs more GPU memory for the KV cache.">
        <Num value={l.localContextCap} min={2048} onChange={(v) => set({ localContextCap: v ?? 32768 })} suffix="tokens" width={100} />
      </Row>
      <Row label="Max output tokens">
        <Num value={l.localMaxOutputTokens} min={256} onChange={(v) => set({ localMaxOutputTokens: v ?? 4096 })} width={100} />
      </Row>
      <Row label="Thinking" hint="Let the model reason in a hidden scratchpad before acting. Slower, usually better.">
        <Toggle value={l.enableThinking} onChange={(v) => set({ enableThinking: v })} />
      </Row>
    </Section>
  );
}

function AppearanceSection({ s }: { s: Settings }) {
  const patch = usePatch();
  return (
    <Section title="Appearance">
      <Row label="Theme">
        <div className="flex gap-2">
          {(["dark", "light"] as const).map((t) => (
            <button key={t} className={`tile ${s.theme === t ? "on" : ""}`} style={{ minHeight: 0, width: 96, padding: "8px 12px" }} onClick={() => patch({ theme: t })}>
              {t}
            </button>
          ))}
        </div>
      </Row>
      <Row label="Editor font size">
        <Num value={s.fontSize} min={10} max={24} onChange={(v) => patch({ fontSize: v ?? 13 })} suffix="px" width={70} />
      </Row>
      <Row label="About you" hint="A short note the agent always sees, e.g. your role or style preferences.">
        <Text value={s.persona} width={360} onChange={(v) => patch({ persona: v })} placeholder="Senior TS engineer; prefer small diffs" />
      </Row>
    </Section>
  );
}

function AboutSection({ s }: { s: Settings }) {
  const patch = usePatch();
  const [info, setInfo] = useState<(AppInfo & { secretsBackend?: string }) | null>(null);
  useEffect(() => {
    call<AppInfo & { secretsBackend?: string }>("app:info", {}).then(setInfo, () => undefined);
  }, []);
  return (
    <>
      <Section title="RLM IDE">
        <Row label="Version">
          <span className="mono">{info?.version ?? "…"}</span>
        </Row>
        <Row label="Platform">
          <span className="mono">{info?.platform ?? "…"}</span>
        </Row>
        <Row label="Kernel sandbox" hint={info?.sandbox.detail}>
          <span className="mono">{info?.sandbox.mode ?? "…"}</span>
        </Row>
        <Row label="Secret storage">
          <span className="mono">{info?.secretsBackend ?? "…"}</span>
        </Row>
        <Row label="Data folder">
          <span className="mono selectable text-[11.5px]">{info?.userDataDir ?? "…"}</span>
        </Row>
      </Section>
      <Section title="Updates">
        <Row label="Check for updates automatically">
          <Toggle value={s.autoUpdate} onChange={(v) => patch({ autoUpdate: v })} />
        </Row>
        <Row label="Channel">
          <Select value={s.updateChannel} options={[["stable", "Stable"], ["beta", "Beta"]]} onChange={(v) => patch({ updateChannel: v })} />
        </Row>
      </Section>
    </>
  );
}
