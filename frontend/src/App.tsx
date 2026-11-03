import { useCallback, useEffect, useRef, useState } from "react";
import type { DragEvent, ReactNode } from "react";
import { errorMessage } from "./api";
import { analyze, loadEngine, ROLES } from "./local/engine";
import type { Engine, ProgressMap } from "./local/engine";
import { validateFile } from "./files";
import { InputPanel } from "./components/InputPanel";
import { ResultCard } from "./components/ResultCard";
import { SegmentationViewer } from "./components/SegmentationViewer";
import { Scanner } from "./components/Scanner";
import { LoadingScreen } from "./components/LoadingScreen";
import { ClassifierResults } from "./components/ClassifierResults";
import type { ClassId, Inspection } from "./types";

type Operation = "predict";
type Panel = "details" | null;

const emptyProgress = (): ProgressMap =>
  Object.fromEntries(ROLES.map((role) => [role, { loaded: 0, total: 0, done: false }])) as ProgressMap;

function Sheet({ title, onClose, children }: { title: string; onClose(): void; children: ReactNode }) {
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => { dialog.current?.showModal(); }, []);
  return <dialog className="info-sheet" ref={dialog} onCancel={onClose} onClose={onClose} aria-labelledby="sheet-title"
    onClick={(event) => { if (event.target === event.currentTarget) onClose(); }}>
    <div className="sheet-surface">
      <div className="sheet-header"><h2 id="sheet-title">{title}</h2>
        <button className="icon-button" type="button" aria-label="Close panel" onClick={onClose}>×</button></div>
      <div className="sheet-content">{children}</div>
    </div>
  </dialog>;
}

export default function App() {
  const [engine, setEngine] = useState<Engine | null>(null);
  const [progress, setProgress] = useState<ProgressMap>(emptyProgress);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [loaderGone, setLoaderGone] = useState(false);
    const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [inspection, setInspection] = useState<Inspection | null>(null);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [selectedClassId, setSelectedClassId] = useState<ClassId | null>(null);
  const [busy, setBusy] = useState<Operation | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploadOpen, setUploadOpen] = useState(true);
  const [panel, setPanel] = useState<Panel>(null);
  const active = useRef(false);
  const mounted = useRef(true);
  const lastPredictedFile = useRef<File | null>(null);
  const controller = useRef<AbortController | null>(null);

  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; controller.current?.abort(); };
  }, []);

  // Download and start all three models in parallel as soon as the page opens.
  useEffect(() => {
    const request = new AbortController();
    setLoadError(null);
    setProgress(emptyProgress());
    loadEngine((role, value) => setProgress((current) => ({ ...current, [role]: value })), request.signal)
      .then((loaded) => { if (!request.signal.aborted) setEngine(loaded); })
      .catch((reason: unknown) => {
        if (request.signal.aborted) return;
        console.error(reason);
        setLoadError(reason instanceof Error ? reason.message : "Check your connection and try again.");
      });
    return () => request.abort();
  }, []);

  useEffect(() => {
    if (!file) { setPreview(null); return; }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  // Free the previous result's in-memory image and mask when a new one replaces it.
  useEffect(() => {
    if (!inspection) return;
    return () => { URL.revokeObjectURL(inspection.image.url); URL.revokeObjectURL(inspection.mask.url); };
  }, [inspection]);

  const run = useCallback(async (nextFile?: File) => {
    if (active.current || !engine) return;
    const targetFile = nextFile ?? file;
    if (!targetFile) return;
    lastPredictedFile.current = targetFile;
    active.current = true;
    const request = new AbortController();
    controller.current = request;
    setBusy("predict"); setError(null); setInspection(null);
    setSelectedId(null); setSelectedClassId(null);
    setUploadOpen(false); setPanel(null);
    try {
      // Keep the analyzing view on screen long enough to read, even when the result is instant.
      const [result] = await Promise.all([
        analyze(engine, targetFile, request.signal),
        new Promise((resolve) => setTimeout(resolve, 2200)),
      ]);
      if (mounted.current && !request.signal.aborted) setInspection(result);
    } catch (reason) {
      if (mounted.current) {
        setError(errorMessage(reason));
        setUploadOpen(true);
      }
    } finally {
      active.current = false;
      if (mounted.current) setBusy(null);
    }
  }, [engine, file]);

  // Immediately analyze any uploaded file once the crew is on site.
  useEffect(() => {
    if (engine && file && file !== lastPredictedFile.current && !busy && !active.current) {
      void run(file);
    }
  }, [engine, file, busy, run]);

  const chooseFile = useCallback((next: File | null) => {
    setFile(next);
    if (!next) {
      lastPredictedFile.current = null;
      return;
    }
    setInspection(null);
    setSelectedId(null);
    setSelectedClassId(null);
    setError(null);
    setUploadOpen(false);
  }, []);

  async function onWorkspaceDrop(event: DragEvent<HTMLElement>) {
    if (event.defaultPrevented || busy) return;
    const dropped = Array.from(event.dataTransfer?.files ?? []);
    if (!dropped.length) return;
    event.preventDefault();
    if (dropped.length !== 1) {
      setError("Choose one image at a time.");
      return;
    }
    try {
      const accepted = await validateFile(dropped[0]);
      chooseFile(accepted);
    } catch (reason) {
      setError(errorMessage(reason));
    }
  }

  const selected = inspection?.analytics.regions.find((row) => row.region_id === selectedId) ?? null;
  const hasImage = !!inspection || !!preview;
  const showResultCard = !!inspection && !uploadOpen;

  return <div className="app-shell">
    {!loaderGone && <LoadingScreen progress={progress} ready={!!engine} error={loadError}
      onRetry={() => window.location.reload()} onGone={() => setLoaderGone(true)} />}
    <a className="skip-link" href="#workspace">Skip to image</a>
    <header className="app-header"><div className="header-inner">
      <a className="brand" href="#workspace" aria-label="Concrete workspace"><span className="brand-mark" aria-hidden="true">C<span>09</span></span><strong>Concrete</strong></a>
      <nav className="app-actions" aria-label="Workspace actions">
        <button type="button" className="button primary" disabled={!!busy} onClick={() => setUploadOpen(true)}><span aria-hidden="true">+</span> New image</button>
      </nav>
    </div></header>
    <main
      id="workspace"
      className={`image-workspace ${hasImage ? "has-image" : ""} ${showResultCard ? "has-result-card" : ""}`}
      aria-busy={!!busy}
      onDragOver={(event) => { if (!busy) event.preventDefault(); }}
      onDrop={(event) => void onWorkspaceDrop(event)}
    >
      <h1 className="sr-only">Concrete photo inspection</h1>
      {inspection ? (
        <SegmentationViewer
          key={inspection.id}
          inspection={inspection}
          selected={selected}
          selectedClassId={selectedClassId}
          onClear={() => {
            setSelectedId(null);
            setSelectedClassId(null);
          }}
        />
      ) : preview ? (
        busy ? <Scanner src={preview} /> : <img className="workspace-photo" src={preview} alt="Photo ready to analyze" />
      ) : (
        <div className="empty-frame" aria-hidden="true"><i /><i /><i /><i /></div>
      )}
      <div className="workspace-label">{inspection ? "Inspection" : "New inspection"}<span>{inspection ? new Date(inspection.created_at).toLocaleDateString() : ""}</span></div>
      <div className={`upload-float ${!uploadOpen || busy ? "is-hidden" : ""}`} aria-hidden={!uploadOpen || !!busy}>
        {hasImage && <button type="button" className="icon-button upload-close" aria-label="Hide upload panel" onClick={() => setUploadOpen(false)}>×</button>}
        <InputPanel compact file={file} busy={busy} onFile={chooseFile} onPredict={() => void run()} />
      </div>
      {showResultCard && (
        <ResultCard
          inspection={inspection}
          onOpenDetails={() => setPanel("details")}
        />
      )}
      {error && <div className="error-float" role="alert"><p>{error}</p><button className="icon-button" type="button" aria-label="Dismiss error" onClick={() => setError(null)}>×</button></div>}
      {!inspection && !busy && <div className="workspace-bottom"><span>{file ? file.name : "Drop or paste a concrete photo to analyze immediately"}</span><button type="button" className="text-button" onClick={() => setPanel("details")}>How it works</button></div>}
    </main>
    <footer className="status-bar"><div className="server-status"><div><span className={`status-dot ${engine ? "ready" : ""}`} aria-hidden="true" />
      <span className="status-title" role="status">{engine ? "Crew on site" : "Crew arriving"}</span></div></div><span>Visual inspection</span></footer>
    {panel && <Sheet title="About this inspection" onClose={() => setPanel(null)}>
      <section className="help-content"><h3>A closer look at concrete</h3><p>Choose, drop, or paste a JPEG or PNG to analyze it automatically. Use Photo, Damage, and Mask to compare the original image with its highlights.</p>
        <p>Undamaged results show the original image without highlights.</p>
        <p>These are predictions of visible damage, not a structural safety assessment.</p></section>
      {inspection && <><ClassifierResults inspection={inspection} /><details className="model-details"><summary>Technical details</summary>
        <dl><dt>Inspection ID</dt><dd className="hash-wrap">{inspection.id}</dd><dt>Segmentation checkpoint</dt><dd className="hash-wrap">{inspection.model.checkpoint_sha256}</dd>
          <dt>Dataset</dt><dd>{inspection.model.dataset.release_id || "Not specified"}</dd><dt>Runtime</dt><dd>epoch {inspection.model.checkpoint_epoch}</dd></dl>
      </details></>}
    </Sheet>}
  </div>;
}
