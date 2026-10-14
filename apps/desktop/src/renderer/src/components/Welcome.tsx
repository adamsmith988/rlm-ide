import { useEffect, useState } from "react";
import type { LocalModelStatus } from "@rlm/contracts";
import { call } from "../api";
import { useApp } from "../store";
import { Circles, PersonLaptop } from "../ui/LineArt";
import { IFolder, IGear, ITeam, IDownload } from "../ui/icons";

export function Welcome() {
  const { settings, openWorkspace, set } = useApp();
  const [models, setModels] = useState<LocalModelStatus[]>([]);
  useEffect(() => {
    call<LocalModelStatus[]>("models:status", {}).then(setModels, () => undefined);
  }, []);
  const recent = settings?.recentWorkspaces ?? [];
  const missing = models.filter((m) => !m.installed);
  const usesLocal = settings?.providers.find((p) => p.id === settings.models.find((m) => m.id === settings.roles.agent)?.providerId)?.kind === "local_onnx";

  const open = async () => {
    const root = await call<string | null>("dialog:openFolder", {});
    if (root) await openWorkspace(root);
  };

  return (
    <div className="relative flex min-h-0 flex-1 overflow-hidden bg-bg fade-in">
      <Circles count={3} />
      <div className="relative mx-auto flex w-full max-w-[1100px] items-center gap-12 px-16">
        <div className="flex-1">
          <div className="smallcaps mb-5">Recursive language model IDE</div>
          <h1 className="heading m-0 text-[64px] leading-[1.02]">
            Think in code.
            <br />
            Ship with care.
          </h1>
          <p className="mt-5 max-w-[440px] text-[14px] text-dim">
            A local-first agent that plans in a live Python REPL, remembers what matters, and asks before it touches your files.
          </p>
          <div className="mt-8 flex gap-3">
            <button className="btn btn-primary" style={{ height: 42, padding: "0 26px" }} onClick={open}>
              <IFolder size={15} /> Open folder
            </button>
            <button className="btn" style={{ height: 42, padding: "0 22px" }} onClick={() => set({ activity: "team" })} data-testid="welcome-team">
              <ITeam size={15} /> Team hub
            </button>
            <button className="btn" style={{ height: 42, padding: "0 22px" }} onClick={() => set({ activity: "hub" })} data-testid="welcome-hub">
              <IDownload size={15} /> Models hub
            </button>
            <button className="btn" style={{ height: 42, padding: "0 22px" }} onClick={() => set({ activity: "settings" })}>
              <IGear size={15} /> Settings
            </button>
          </div>

          {usesLocal && missing.length > 0 && (
            <div className="card mt-8 max-w-[480px] p-4">
              <div className="smallcaps mb-1">Built-in models</div>
              <div className="text-[13px]">
                {missing.map((m) => m.id.replace("builtin:", "")).join(" and ")} not installed yet.{" "}
                <button className="underline" onClick={() => set({ activity: "settings" })}>
                  Open the Model Manager
                </button>{" "}
                or add an OpenAI-compatible API key.
              </div>
            </div>
          )}

          {recent.length > 0 && (
            <div className="mt-10">
              <div className="smallcaps mb-3">Recent</div>
              <div className="grid max-w-[560px] grid-cols-2 gap-2.5">
                {recent.slice(0, 6).map((r) => (
                  <button key={r} className="card px-4 py-3 text-left transition-transform hover:-translate-y-[1px]" onClick={() => void openWorkspace(r)}>
                    <div className="heading truncate text-[16px]">{r.split(/[\\/]/).pop()}</div>
                    <div className="mono truncate text-[11px] text-faint">{r}</div>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
        <div className="hidden shrink-0 text-ink lg:block">
          <PersonLaptop width={340} />
        </div>
      </div>
    </div>
  );
}
