import { useEffect, useMemo, useState } from "react";
import type { SearchHit } from "@rlm/contracts";
import { call, errText } from "../api";
import { useApp } from "../store";
import { PanelHeader } from "./Explorer";

export function SearchPanel() {
  const { openFile, toast } = useApp();
  const [q, setQ] = useState("");
  const [hits, setHits] = useState<SearchHit[]>([]);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (q.trim().length < 2) {
      setHits([]);
      return;
    }
    const t = setTimeout(async () => {
      setBusy(true);
      try {
        setHits(await call<SearchHit[]>("fs:search", { query: q, maxResults: 500 }));
      } catch (e) {
        toast(errText(e), "error");
      } finally {
        setBusy(false);
      }
    }, 250);
    return () => clearTimeout(t);
  }, [q, toast]);

  const grouped = useMemo(() => {
    const m = new Map<string, SearchHit[]>();
    for (const h of hits) m.set(h.path, [...(m.get(h.path) ?? []), h]);
    return [...m.entries()];
  }, [hits]);

  return (
    <div className="flex h-full flex-col">
      <PanelHeader title="Search" />
      <div className="px-3 pb-2">
        <input autoFocus className="input" placeholder="Search in files" value={q} onChange={(e) => setQ(e.target.value)} />
        <div className="mt-1.5 text-[11px] text-faint">{busy ? "searching…" : q.trim().length >= 2 ? `${hits.length} results in ${grouped.length} files` : ""}</div>
      </div>
      <div className="min-h-0 flex-1 overflow-auto pb-2">
        {grouped.map(([path, list]) => (
          <div key={path} className="mb-1">
            <div className="truncate px-3 py-1 text-[12px] font-medium">{path}</div>
            {list.map((h, i) => (
              <button key={i} className="flex w-full gap-2 px-3 py-[3px] text-left text-[12px] hover:bg-surface-2" onClick={() => void openFile(h.path, { line: h.line })}>
                <span className="mono w-8 shrink-0 text-right text-faint">{h.line}</span>
                <span className="mono truncate text-dim">{highlight(h.text.trim(), q)}</span>
              </button>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}

function highlight(text: string, q: string) {
  const i = text.toLowerCase().indexOf(q.toLowerCase());
  if (i < 0) return text;
  return (
    <>
      {text.slice(0, i)}
      <mark className="rounded-[3px] bg-ink px-[1px] text-bg">{text.slice(i, i + q.length)}</mark>
      {text.slice(i + q.length)}
    </>
  );
}
