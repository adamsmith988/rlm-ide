import { useCallback, useEffect, useState } from "react";
import type { DirEntry } from "@rlm/contracts";
import { call, errText } from "../api";
import { useApp } from "../store";
import { IChevron, IFile, IFolder, IPlus, IRefresh } from "../ui/icons";

export function Explorer() {
  const { workspace, treeVersion, set } = useApp();
  const [creating, setCreating] = useState<null | { kind: "file" | "dir"; parent: string }>(null);
  return (
    <div className="flex h-full flex-col">
      <PanelHeader title={workspace?.name ?? "Explorer"}>
        <button className="iconbtn" title="New file" onClick={() => setCreating({ kind: "file", parent: "" })}>
          <IPlus size={15} />
        </button>
        <button className="iconbtn" title="New folder" onClick={() => setCreating({ kind: "dir", parent: "" })}>
          <IFolder size={14} />
        </button>
        <button className="iconbtn" title="Refresh" onClick={() => set({ treeVersion: treeVersion + 1 })}>
          <IRefresh size={14} />
        </button>
      </PanelHeader>
      <div className="min-h-0 flex-1 overflow-auto py-1">
        {creating && creating.parent === "" && <NewEntry kind={creating.kind} parent="" depth={0} done={() => setCreating(null)} />}
        <Dir path="" depth={0} version={treeVersion} onCreate={setCreating} creating={creating} />
      </div>
    </div>
  );
}

export function PanelHeader({ title, children }: { title: string; children?: React.ReactNode }) {
  return (
    <div className="flex h-10 shrink-0 items-center gap-1 px-3">
      <span className="smallcaps flex-1 truncate">{title}</span>
      {children}
    </div>
  );
}

function Dir({
  path,
  depth,
  version,
  onCreate,
  creating,
}: {
  path: string;
  depth: number;
  version: number;
  onCreate: (c: { kind: "file" | "dir"; parent: string } | null) => void;
  creating: { kind: "file" | "dir"; parent: string } | null;
}) {
  const [entries, setEntries] = useState<DirEntry[] | null>(null);
  useEffect(() => {
    call<DirEntry[]>("fs:readDir", { path }).then(setEntries, () => setEntries([]));
  }, [path, version]);
  if (!entries) return null;
  return (
    <>
      {entries.map((e) => (
        <Node key={e.path} entry={e} depth={depth} version={version} onCreate={onCreate} creating={creating} />
      ))}
    </>
  );
}

function Node({
  entry,
  depth,
  version,
  onCreate,
  creating,
}: {
  entry: DirEntry;
  depth: number;
  version: number;
  onCreate: (c: { kind: "file" | "dir"; parent: string } | null) => void;
  creating: { kind: "file" | "dir"; parent: string } | null;
}) {
  const { openFile, activeTab, toast, set, treeVersion } = useApp();
  const [open, setOpen] = useState(false);
  const [menu, setMenu] = useState<{ x: number; y: number } | null>(null);
  const [renaming, setRenaming] = useState(false);
  const active = activeTab === `file:${entry.path}`;

  const close = useCallback(() => setMenu(null), []);
  useEffect(() => {
    if (!menu) return;
    window.addEventListener("click", close);
    return () => window.removeEventListener("click", close);
  }, [menu, close]);

  const act = async (fn: () => Promise<unknown>) => {
    try {
      await fn();
      set({ treeVersion: treeVersion + 1 });
    } catch (e) {
      toast(errText(e), "error");
    }
  };

  return (
    <>
      <div
        className="group flex h-[24px] cursor-pointer items-center gap-1.5 pr-2 text-[12.5px]"
        style={{ paddingLeft: 10 + depth * 12, background: active ? "var(--surface-3)" : undefined }}
        onClick={() => (entry.kind === "dir" ? setOpen(!open) : void openFile(entry.path))}
        onContextMenu={(e) => {
          e.preventDefault();
          setMenu({ x: e.clientX, y: e.clientY });
        }}
        onMouseEnter={(e) => !active && (e.currentTarget.style.background = "var(--surface-2)")}
        onMouseLeave={(e) => !active && (e.currentTarget.style.background = "")}
      >
        {entry.kind === "dir" ? (
          <IChevron size={12} style={{ transform: open ? "rotate(90deg)" : undefined, transition: "transform .12s" }} className="shrink-0 text-faint" />
        ) : (
          <span className="w-3 shrink-0" />
        )}
        {entry.kind === "dir" ? <IFolder size={14} className="shrink-0 text-dim" /> : <IFile size={14} className="shrink-0 text-faint" />}
        {renaming ? (
          <input
            autoFocus
            className="input h-[20px] py-0 text-[12px]"
            defaultValue={entry.name}
            onClick={(e) => e.stopPropagation()}
            onKeyDown={(e) => {
              if (e.key === "Escape") setRenaming(false);
              if (e.key === "Enter") {
                const to = [...entry.path.split("/").slice(0, -1), e.currentTarget.value].filter(Boolean).join("/");
                setRenaming(false);
                void act(() => call("fs:rename", { from: entry.path, to }));
              }
            }}
            onBlur={() => setRenaming(false)}
          />
        ) : (
          <span className="truncate">{entry.name}</span>
        )}
      </div>
      {menu && (
        <div className="card fixed z-50 min-w-[160px] py-1 text-[12.5px]" style={{ left: menu.x, top: menu.y, boxShadow: "var(--shadow-lg)" }}>
          {entry.kind === "dir" && (
            <>
              <MenuItem onClick={() => (setOpen(true), onCreate({ kind: "file", parent: entry.path }))}>New file</MenuItem>
              <MenuItem onClick={() => (setOpen(true), onCreate({ kind: "dir", parent: entry.path }))}>New folder</MenuItem>
            </>
          )}
          <MenuItem onClick={() => setRenaming(true)}>Rename</MenuItem>
          <MenuItem onClick={() => void navigator.clipboard.writeText(entry.path)}>Copy path</MenuItem>
          <MenuItem
            danger
            onClick={() => {
              if (confirm(`Delete ${entry.path}?`)) void act(() => call("fs:delete", { path: entry.path }));
            }}
          >
            Delete
          </MenuItem>
        </div>
      )}
      {entry.kind === "dir" && open && (
        <>
          {creating && creating.parent === entry.path && <NewEntry kind={creating.kind} parent={entry.path} depth={depth + 1} done={() => onCreate(null)} />}
          <Dir path={entry.path} depth={depth + 1} version={version} onCreate={onCreate} creating={creating} />
        </>
      )}
    </>
  );
}

function MenuItem({ children, onClick, danger }: { children: React.ReactNode; onClick: () => void; danger?: boolean }) {
  return (
    <button className="block w-full px-3 py-1.5 text-left hover:bg-surface-2" style={{ color: danger ? "var(--err)" : undefined }} onClick={onClick}>
      {children}
    </button>
  );
}

function NewEntry({ kind, parent, depth, done }: { kind: "file" | "dir"; parent: string; depth: number; done: () => void }) {
  const { set, treeVersion, toast, openFile } = useApp();
  return (
    <div className="flex h-[24px] items-center gap-1.5 pr-2" style={{ paddingLeft: 22 + depth * 12 }}>
      {kind === "dir" ? <IFolder size={14} /> : <IFile size={14} />}
      <input
        autoFocus
        className="input h-[20px] py-0 text-[12px]"
        placeholder={kind === "dir" ? "folder name" : "file name"}
        onKeyDown={async (e) => {
          if (e.key === "Escape") done();
          if (e.key === "Enter" && e.currentTarget.value.trim()) {
            const p = [parent, e.currentTarget.value.trim()].filter(Boolean).join("/");
            try {
              await call("fs:create", { path: p, kind });
              set({ treeVersion: treeVersion + 1 });
              if (kind === "file") void openFile(p);
            } catch (err) {
              toast(errText(err), "error");
            }
            done();
          }
        }}
        onBlur={done}
      />
    </div>
  );
}
