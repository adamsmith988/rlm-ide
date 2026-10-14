import { useEffect, useState, type ReactNode } from "react";

export function Section({ title, hint, children }: { title: string; hint?: string; children: ReactNode }) {
  return (
    <section className="mb-8">
      <h3 className="heading m-0 text-[22px]">{title}</h3>
      {hint && <p className="mb-0 mt-1 max-w-[620px] text-[12.5px] text-dim">{hint}</p>}
      <div className="mt-4 flex flex-col gap-1">{children}</div>
    </section>
  );
}

export function Row({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <div className="flex items-center gap-6 border-b border-line py-3 last:border-b-0">
      <div className="min-w-0 flex-1">
        <div className="text-[13px] font-medium">{label}</div>
        {hint && <div className="mt-[2px] text-[11.5px] text-dim">{hint}</div>}
      </div>
      <div className="flex shrink-0 items-center gap-2">{children}</div>
    </div>
  );
}

export function Toggle({ value, onChange }: { value: boolean; onChange: (v: boolean) => void }) {
  return (
    <button
      role="switch"
      aria-checked={value}
      className="relative h-[22px] w-[40px] rounded-full border-[1.5px] transition-colors"
      style={{ borderColor: value ? "var(--accent)" : "var(--border-strong)", background: value ? "var(--accent)" : "transparent" }}
      onClick={() => onChange(!value)}
    >
      <span
        className="absolute top-1/2 h-[14px] w-[14px] -translate-y-1/2 rounded-full transition-all"
        style={{ left: value ? 20 : 3, background: value ? "var(--accent-ink)" : "var(--text-dim)" }}
      />
    </button>
  );
}

/** Number input that commits on blur/Enter so typing doesn't spam settings writes. */
export function Num({ value, onChange, min, max, step = 1, width = 96, suffix }: { value: number | null; onChange: (v: number | null) => void; min?: number; max?: number; step?: number; width?: number; suffix?: string }) {
  const [t, setT] = useState(value === null ? "" : String(value));
  useEffect(() => setT(value === null ? "" : String(value)), [value]);
  const commit = () => {
    if (t.trim() === "") return onChange(null);
    let n = Number(t);
    if (!Number.isFinite(n)) return setT(value === null ? "" : String(value));
    if (min !== undefined) n = Math.max(min, n);
    if (max !== undefined) n = Math.min(max, n);
    if (n !== value) onChange(n);
    else setT(String(n));
  };
  return (
    <span className="flex items-center gap-1.5">
      <input className="input mono text-right" style={{ width }} value={t} step={step} onChange={(e) => setT(e.target.value)} onBlur={commit} onKeyDown={(e) => e.key === "Enter" && commit()} />
      {suffix && <span className="text-[11px] text-faint">{suffix}</span>}
    </span>
  );
}

export function Ratio({ value, onChange, min = 0, max = 1 }: { value: number; onChange: (v: number) => void; min?: number; max?: number }) {
  const [v, setV] = useState(value);
  useEffect(() => setV(value), [value]);
  return (
    <span className="flex items-center gap-3">
      <input type="range" min={min} max={max} step={0.01} value={v} className="w-[160px] accent-[var(--text)]" onChange={(e) => setV(Number(e.target.value))} onMouseUp={() => onChange(v)} onKeyUp={() => onChange(v)} />
      <span className="mono w-[42px] text-right text-[12px]">{Math.round(v * 100)}%</span>
    </span>
  );
}

export function Select<T extends string>({ value, options, onChange, width = 180 }: { value: T; options: Array<[T, string]>; onChange: (v: T) => void; width?: number }) {
  return (
    <select className="input cursor-pointer" style={{ width }} value={value} onChange={(e) => onChange(e.target.value as T)}>
      {options.map(([v, l]) => (
        <option key={v} value={v}>
          {l}
        </option>
      ))}
    </select>
  );
}

export function Text({ value, onChange, placeholder, width = 260, mono, type }: { value: string; onChange: (v: string) => void; placeholder?: string; width?: number; mono?: boolean; type?: string }) {
  const [t, setT] = useState(value);
  useEffect(() => setT(value), [value]);
  return (
    <input
      className={`input ${mono ? "mono" : ""}`}
      style={{ width }}
      type={type}
      value={t}
      placeholder={placeholder}
      onChange={(e) => setT(e.target.value)}
      onBlur={() => t !== value && onChange(t)}
      onKeyDown={(e) => e.key === "Enter" && t !== value && onChange(t)}
    />
  );
}

export function Lines({ value, onChange, placeholder }: { value: string[]; onChange: (v: string[]) => void; placeholder?: string }) {
  const [t, setT] = useState(value.join("\n"));
  useEffect(() => setT(value.join("\n")), [value]);
  return (
    <textarea
      className="input mono h-[96px] resize-y text-[11.5px] shrink-0" style={{ width: 360 }}
      value={t}
      placeholder={placeholder}
      onChange={(e) => setT(e.target.value)}
      onBlur={() => onChange(t.split("\n").map((l) => l.trim()).filter(Boolean))}
    />
  );
}

export function Bytes({ n }: { n: number }) {
  const u = ["B", "KB", "MB", "GB"];
  let i = 0;
  let v = n;
  while (v >= 1024 && i < u.length - 1) (v /= 1024), i++;
  return <>{`${v.toFixed(i >= 2 ? 1 : 0)} ${u[i]}`}</>;
}
