import { useState } from "react";
import { call } from "../api";
import { useApp } from "../store";
import { IPlus, ITrash } from "../ui/icons";
import { PanelHeader } from "./Explorer";

export function ChatsPanel() {
  const { chats, chatId, selectChat, newChat, refreshChats, set, messages } = useApp();
  const [editing, setEditing] = useState<string | null>(null);
  return (
    <div className="flex h-full flex-col">
      <PanelHeader title="Chats">
        <button className="iconbtn" title="New chat (Ctrl+Shift+N)" onClick={() => void newChat()}>
          <IPlus size={15} />
        </button>
      </PanelHeader>
      <div className="min-h-0 flex-1 overflow-auto px-2 pb-2">
        {chats.map((c) => {
          const on = c.id === chatId;
          const running = (messages[c.id] ?? []).some((m) => m.status === "running");
          return (
            <div
              key={c.id}
              className="group mb-1 flex cursor-pointer items-center gap-2 rounded-[10px] px-3 py-2"
              style={{ background: on ? "var(--accent)" : undefined, color: on ? "var(--accent-ink)" : undefined }}
              onClick={() => {
                void selectChat(c.id);
                set({ agent: true });
              }}
              onDoubleClick={() => setEditing(c.id)}
            >
              <div className="min-w-0 flex-1">
                {editing === c.id ? (
                  <input
                    autoFocus
                    className="input h-[22px] py-0"
                    defaultValue={c.title}
                    onClick={(e) => e.stopPropagation()}
                    onBlur={() => setEditing(null)}
                    onKeyDown={async (e) => {
                      if (e.key === "Escape") setEditing(null);
                      if (e.key === "Enter") {
                        await call("chat:rename", { chatId: c.id, title: e.currentTarget.value });
                        setEditing(null);
                        await refreshChats();
                      }
                    }}
                  />
                ) : (
                  <div className="truncate text-[12.5px] font-medium">{c.title}</div>
                )}
                <div className="text-[10.5px] opacity-60">
                  {running ? <span className="pulse">running…</span> : new Date(c.updatedAt).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" })}
                </div>
              </div>
              <button
                className="opacity-0 group-hover:opacity-70 hover:!opacity-100"
                title="Delete chat"
                onClick={async (e) => {
                  e.stopPropagation();
                  if (!confirm(`Delete "${c.title}"?`)) return;
                  await call("chat:delete", { chatId: c.id });
                  if (on) await selectChat(null);
                  await refreshChats();
                }}
              >
                <ITrash size={14} />
              </button>
            </div>
          );
        })}
        {!chats.length && <div className="px-2 text-[12px] text-faint">No chats yet.</div>}
      </div>
    </div>
  );
}
