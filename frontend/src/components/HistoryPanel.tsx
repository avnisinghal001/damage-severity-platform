import { useEffect, useRef, useState } from "react";
import { errorMessage, getHistory } from "../api";
import type { HistoryPage } from "../types";

interface Props { revision: string | undefined; disabled: boolean; onLoad(id: string): void }

export function HistoryPanel({ revision, disabled, onLoad }: Props) {
  const [page, setPage] = useState<HistoryPage | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const active = useRef<AbortController | null>(null);

  async function load(more = false) {
    active.current?.abort();
    const controller = new AbortController();
    active.current = controller;
    const timeout = window.setTimeout(() => controller.abort(), 10_000);
    setBusy(true); setError(null);
    const cursor = more && page?.next_before && page.next_before_id
      ? { before: page.next_before, id: page.next_before_id } : undefined;
    try {
      const result = await getHistory(controller.signal, cursor);
      if (!controller.signal.aborted) setPage((previous) => ({ ...result,
        items: more ? [...(previous?.items || []), ...result.items] : result.items }));
    } catch (reason) {
      if (active.current === controller) setError(controller.signal.aborted ? "History request timed out." : errorMessage(reason));
    } finally {
      window.clearTimeout(timeout);
      if (active.current === controller) setBusy(false);
    }
  }

  useEffect(() => { void load(); return () => { active.current?.abort(); active.current = null; }; }, [revision]);
  return <section className="panel input-panel" aria-labelledby="history-title">
    <div className="section-header"><h2 id="history-title">Inspection history</h2>
      <button type="button" className="text-button" disabled={busy} onClick={() => void load()}>Refresh</button></div>
    {error && <p role="alert" className="small">{error}</p>}
    {busy && <p role="status">Loading saved inspections…</p>}
    {!busy && page?.items.length === 0 && <p className="muted">No saved inspections yet.</p>}
    <ul className="history-list">{page?.items.map((item) => <li key={item.id}>
      <button className="history-button" type="button" disabled={disabled} onClick={() => onLoad(item.id)}>
        <strong>{new Date(item.created_at).toLocaleString()}</strong>
        <span>{item.width} × {item.height} px · {item.classification?.label.replaceAll("_", " ") || "Legacy result"}</span>
        <span>Component level: {item.severity?.status === "available" ? item.severity.label : "unavailable"}</span>
        <code title={item.id}>{item.id.slice(0, 12)}…</code>
      </button>
    </li>)}</ul>
    {page?.next_before && <button type="button" className="button secondary compact" disabled={busy}
      onClick={() => void load(true)}>Load more</button>}
  </section>;
}
