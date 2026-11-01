import { useEffect, useRef, useState } from "react";
import { errorMessage, getInspection, predict, UUID_PATTERN } from "./api";
import { AnalyticsPanel } from "./components/AnalyticsPanel";
import { InputPanel } from "./components/InputPanel";
import { SegmentationViewer } from "./components/SegmentationViewer";
import { ServerStatus } from "./components/ServerStatus";
import type { Inspection } from "./types";

type Operation = "predict" | "retrieve";

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [query, setQuery] = useState("");
  const [inspection, setInspection] = useState<Inspection | null>(null);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [busy, setBusy] = useState<Operation | null>(null);
  const [error, setError] = useState<string | null>(null);
  const active = useRef(false);
  const mounted = useRef(true);
  const controller = useRef<AbortController | null>(null);

  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; controller.current?.abort(); };
  }, []);

  async function run(operation: Operation) {
    if (active.current) return;
    const id = query.trim();
    if (operation === "retrieve" && !UUID_PATTERN.test(id)) {
      setError("Enter a complete inspection UUID, including its hyphens.");
      return;
    }
    if (operation === "predict" && !file) return;
    active.current = true;
    const request = new AbortController();
    controller.current = request;
    let timedOut = false;
    const timeout = window.setTimeout(() => { timedOut = true; request.abort(); }, 120_000);
    setBusy(operation);
    setError(null);
    setInspection(null);
    setSelectedId(null);
    try {
      const result = operation === "predict"
        ? await predict(file!, request.signal) : await getInspection(id, request.signal);
      if (mounted.current && !request.signal.aborted) {
        setInspection(result);
        setQuery(result.id);
      }
    } catch (reason) {
      if (mounted.current) setError(timedOut
        ? `The request timed out.${operation === "predict" ? " The backend may still save the inspection; uploads are not automatically retried." : " Try loading the UUID again."}`
        : errorMessage(reason));
    } finally {
      window.clearTimeout(timeout);
      active.current = false;
      if (mounted.current) setBusy(null);
    }
  }

  const selected = inspection?.analytics.regions.find((row) => row.region_id === selectedId) ?? null;
  return <>
    <a className="skip-link" href="#workspace">Skip to inspection workspace</a>
    <header className="app-header"><div className="header-inner">
      <div className="brand"><span className="brand-mark" aria-hidden="true">C<span>09</span></span>
        <div><strong>Concrete</strong><span>Damage analytics</span></div></div>
      <ServerStatus />
    </div></header>
    <main className="page-shell" id="workspace">
      <div className="page-heading"><div><p className="eyebrow">Research & inspection workspace</p>
        <h1>Concrete damage, in focus.</h1>
        <p className="muted">Explore segmentation masks, image coverage and connected regions.</p></div>
        <span className="subtle-tag">U-Net · 9-class segmentation</span></div>
      <div className="workspace-grid">
        <InputPanel file={file} busy={busy} inspectionId={query} onFile={setFile} onId={setQuery}
          onPredict={() => void run("predict")} onRetrieve={() => void run("retrieve")} />
        <div className="result-column" aria-busy={!!busy}>
          {error && <div className="error-banner" role="alert"><strong>Could not complete the request</strong><p>{error}</p></div>}
          {busy && <div className="panel progress-panel" role="status"><span className="progress-mark" aria-hidden="true" />
            <div><strong>{busy === "predict" ? "Analyzing the image" : "Retrieving the inspection"}</strong>
              <p className="muted">{busy === "predict" ? "The backend is generating a native-resolution mask and region measurements." : "Loading the saved result without running inference again."}</p></div></div>}
          {inspection ? <>
            <section className="inspection-heading" aria-label="Inspection provenance">
              <div><p className="eyebrow">Saved inspection</p><h2 className="inspection-uuid">{inspection.id}</h2>
                <p className="small muted">{new Date(inspection.created_at).toLocaleString()} · {inspection.model.device.toUpperCase()} · checkpoint epoch {inspection.model.checkpoint_epoch}</p></div>
              <div className="checkpoint-meta"><span className="small muted">Checkpoint SHA-256</span>
                <code title={inspection.model.checkpoint_sha256}>{inspection.model.checkpoint_sha256.slice(0, 16)}…</code>
                <span className="small muted release-name">{inspection.model.dataset.release_id || "Release not specified"}</span></div>
            </section>
            <SegmentationViewer key={inspection.id} inspection={inspection} selected={selected} onClear={() => setSelectedId(null)} />
            <AnalyticsPanel key={inspection.id} inspection={inspection} selected={selected} onSelect={setSelectedId} />
          </> : !busy && <section className="panel empty-workspace" aria-labelledby="empty-title">
            <div className="empty-diagram" aria-hidden="true"><span /><span /><span /></div>
            <p className="eyebrow">Ready for an inspection</p><h2 id="empty-title">An image is the starting point.</h2>
            <p>Upload a JPEG or PNG, or load an inspection UUID. Your original image, aligned prediction and region measurements will appear here.</p>
            <div className="empty-features"><span>Native mask alignment</span><span>Pixel-level coverage</span><span>Region exploration</span></div>
            <p className="field-help">Predictions support image analysis. They do not certify structural safety.</p>
          </section>}
        </div>
      </div>
    </main>
    <footer className="app-footer"><span>Damage Analytics & Severity Prediction Platform</span><span>Image-space evidence · No structural safety score</span></footer>
  </>;
}
