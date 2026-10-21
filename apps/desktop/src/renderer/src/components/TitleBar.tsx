import { api } from "../api";
import { useApp } from "../store";
import { IAgent, IClose, IExternal, IFolder, IGlobe, IMax, IMin, IPanel, ISearch, ISidebar } from "../ui/icons";
import { LogoMark } from "../ui/LineArt";

export function TitleBar() {
  const { workspace, sidebar, bottom, agent, browser, set, pickFolder, newWindow } = useApp();
  return (
    <header className="drag flex shrink-0 items-center gap-2 border-b border-line bg-bg px-3" style={{ height: "var(--titlebar-h)" }}>
      <div className="flex items-center gap-2">
        <LogoMark />
        <span className="heading text-[15px]">RLM</span>
        {workspace && <span className="text-dim">/ {workspace.name}</span>}
      </div>
      <div className="flex flex-1 justify-center">
        {workspace && (
          <button
            className="nodrag flex h-[24px] w-[380px] max-w-[40vw] items-center gap-2 rounded-full border border-line bg-surface px-3 text-[12px] text-faint hover:border-line-strong"
            onClick={() => set({ palette: "files" })}
          >
            <ISearch size={13} />
            <span className="flex-1 text-left">Search files, run commands</span>
            <span className="mono text-[10.5px]">Ctrl P</span>
          </button>
        )}
      </div>
      <div className="nodrag flex items-center gap-0.5">
        <button className="iconbtn" title="Open folder (Ctrl+O)" data-testid="open-folder" onClick={() => void pickFolder()}>
          <IFolder size={15} />
        </button>
        <button className="iconbtn" title="New window" data-testid="new-window" onClick={() => void newWindow()}>
          <IExternal size={14} />
        </button>
        {workspace && (
          <>
            <button className="iconbtn" title="Toggle sidebar (Ctrl+B)" style={{ color: sidebar ? "var(--text)" : undefined }} onClick={() => set({ sidebar: !sidebar })}>
              <ISidebar />
            </button>
            <button className="iconbtn" title="Toggle panel (Ctrl+J)" style={{ color: bottom ? "var(--text)" : undefined }} onClick={() => set({ bottom: !bottom })}>
              <IPanel />
            </button>
            <button className="iconbtn" title="Toggle browser" data-testid="toggle-browser" style={{ color: browser ? "var(--text)" : undefined }} onClick={() => set({ browser: !browser })}>
              <IGlobe size={15} />
            </button>
            <button className="iconbtn" title="Toggle agent (Ctrl+L)" style={{ color: agent ? "var(--text)" : undefined }} onClick={() => set({ agent: !agent })}>
              <IAgent />
            </button>
            <div className="mx-2 h-4 w-px bg-line" />
          </>
        )}
        <button className="iconbtn" title="Minimize" onClick={() => api.window.minimize()}>
          <IMin size={14} />
        </button>
        <button className="iconbtn" title="Maximize" onClick={() => api.window.maximize()}>
          <IMax size={13} />
        </button>
        <button className="iconbtn hover:!bg-err hover:!text-white" title="Close" onClick={() => api.window.close()}>
          <IClose size={14} />
        </button>
      </div>
    </header>
  );
}
