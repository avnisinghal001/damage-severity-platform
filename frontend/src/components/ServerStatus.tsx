import { useCallback, useEffect, useRef, useState } from "react";
import { getModel, getReadiness } from "../api";
import type { ModelMetadata, Readiness } from "../types";

export function ServerStatus() {
  const controller = useRef<AbortController | null>(null);
  const [checking, setChecking] = useState(true);
  const [readiness, setReadiness] = useState<Readiness | null>(null);
  const [model, setModel] = useState<ModelMetadata | null>(null);

  const refresh = useCallback(async () => {
    controller.current?.abort();
    const active = new AbortController();
    controller.current = active;
    let timedOut = false;
    const timeout = window.setTimeout(() => { timedOut = true; active.abort(); }, 10_000);
    setChecking(true);
    try {
      const [ready, metadata] = await Promise.allSettled([getReadiness(active.signal), getModel(active.signal)]);
      if (active.signal.aborted && !timedOut) return;
      setReadiness(ready.status === "fulfilled" ? ready.value : null);
      setModel(metadata.status === "fulfilled" ? metadata.value : null);
      setChecking(false);
    } finally {
      window.clearTimeout(timeout);
    }
  }, []);

  useEffect(() => { void refresh(); return () => controller.current?.abort(); }, [refresh]);
  const title = checking ? "Checking backend" : !readiness ? "Backend unavailable"
    : readiness.status === "ready" ? "Backend ready"
    : readiness.database && readiness.severity === false ? "Severity unavailable" : "Backend not ready";
  return <div className="server-status">
    <div><span className={`status-dot ${readiness?.status === "ready" && !checking ? "ready" : ""}`} aria-hidden="true" />
      <span className="status-title" role="status">{title}</span>
      <span className="server-detail">{checking ? "Connecting to FastAPI" : model
        ? `${model.device.toUpperCase()} · epoch ${model.checkpoint_epoch}${readiness && !readiness.database ? " · database not ready" : ""}${readiness?.severity === false ? " · severity unavailable" : ""}`
        : "Check the API connection"}</span></div>
    <button className="text-button" type="button" onClick={() => void refresh()} disabled={checking}
      aria-label="Refresh backend readiness">Refresh</button>
  </div>;
}
