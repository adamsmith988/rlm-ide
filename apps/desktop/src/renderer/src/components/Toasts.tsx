import { useApp } from "../store";

export function Toasts() {
  const toasts = useApp((s) => s.toasts);
  return (
    <div className="pointer-events-none fixed bottom-10 left-1/2 z-[60] flex -translate-x-1/2 flex-col items-center gap-2.5">
      {toasts.slice(-4).map((t) => (
        <div
          key={t.id}
          role="status"
          className="selectable pointer-events-auto max-w-[560px] rounded-[18px] px-4 py-2 text-[12.5px] leading-[1.4] fade-in"
          style={{ background: t.tone === "error" ? "var(--err)" : "var(--accent)", color: t.tone === "error" ? "#fff" : "var(--accent-ink)", boxShadow: "var(--shadow-lg)" }}
        >
          {t.text}
        </div>
      ))}
    </div>
  );
}
