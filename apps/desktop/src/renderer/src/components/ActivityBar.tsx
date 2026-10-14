import { useApp, type Activity } from "../store";
import { IBrain, IChat, IDownload, IFiles, IGear, IGit, ISearch, ITeam } from "../ui/icons";
import { useTeam } from "./team/teamStore";

const ITEMS: Array<{ id: Activity; label: string; icon: typeof IFiles; needsWs?: boolean }> = [
  { id: "explorer", label: "Explorer", icon: IFiles, needsWs: true },
  { id: "search", label: "Search", icon: ISearch, needsWs: true },
  { id: "git", label: "Source control", icon: IGit, needsWs: true },
  { id: "chats", label: "Chats", icon: IChat, needsWs: true },
];

const SCREENS = new Set<Activity>(["team", "hub", "memory", "settings"]);

export function ActivityBar() {
  const { activity, sidebar, workspace, git, set } = useApp();
  const unread = useTeam((s) => s.unread);
  const pick = (id: Activity) => {
    if (id === activity && !SCREENS.has(id)) set({ sidebar: !sidebar });
    else set({ activity: id, sidebar: true });
  };
  const btn = (id: Activity, label: string, Icon: typeof IFiles, badge?: number) => {
    const on = activity === id && (sidebar || SCREENS.has(id));
    return (
      <button
        key={id}
        title={label}
        aria-label={label}
        data-testid={`activity-${id}`}
        onClick={() => pick(id)}
        className="relative flex h-10 w-10 items-center justify-center rounded-[12px] transition-colors"
        style={{ background: on ? "var(--accent)" : "transparent", color: on ? "var(--accent-ink)" : "var(--text-dim)" }}
      >
        <Icon size={19} />
        {!!badge && (
          <span className="absolute right-0.5 top-0.5 min-w-[15px] rounded-full bg-ink px-1 text-[9.5px] font-semibold leading-[15px] text-bg">{badge > 99 ? "99+" : badge}</span>
        )}
      </button>
    );
  };
  return (
    <nav className="flex shrink-0 flex-col items-center gap-1.5 border-r border-line bg-bg py-2" style={{ width: "var(--activity-w)" }}>
      {ITEMS.filter((i) => !i.needsWs || workspace).map((i) => btn(i.id, i.label, i.icon, i.id === "git" ? git.dirty : undefined))}
      {btn("team", "Team hub", ITeam, unread)}
      {btn("hub", "Models hub", IDownload)}
      <div className="flex-1" />
      {btn("memory", "Memory", IBrain)}
      {btn("settings", "Settings (Ctrl+,)", IGear)}
    </nav>
  );
}
