import { useEffect, useRef, useState } from "react";
import type { DragEvent, FormEvent } from "react";
import { errorMessage } from "../api";
import { validateFile } from "../files";

interface Props {
  file: File | null;
  busy: "predict" | "retrieve" | null;
  inspectionId: string;
  onFile(file: File | null): void;
  onId(value: string): void;
  onPredict(): void;
  onRetrieve(): void;
}

export function InputPanel({ file, busy, inspectionId, onFile, onId, onPredict, onRetrieve }: Props) {
  const picker = useRef<HTMLInputElement>(null);
  const validation = useRef(0);
  const [preview, setPreview] = useState<string | null>(null);
  const [checking, setChecking] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!file) { setPreview(null); return; }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  useEffect(() => () => { validation.current += 1; }, []);

  async function choose(files: File[]) {
    if (busy) return;
    const token = ++validation.current;
    setError(null);
    onFile(null);
    if (files.length !== 1) {
      setError("Choose one image at a time.");
      setChecking(false);
      return;
    }
    setChecking(true);
    try {
      const accepted = await validateFile(files[0]);
      if (token === validation.current) onFile(accepted);
    } catch (reason) {
      if (token === validation.current) setError(errorMessage(reason));
    } finally {
      if (token === validation.current) setChecking(false);
    }
  }

  function drop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    if (!busy) void choose(Array.from(event.dataTransfer.files));
  }

  function submit(event: FormEvent, action: () => void) {
    event.preventDefault();
    if (!busy && !checking) action();
  }

  return <aside className="input-column" aria-label="Inspection inputs">
    <section className="panel input-panel">
      <p className="eyebrow">01 / New inspection</p>
      <h2>Upload an image</h2>
      <p className="muted">Start with a clear image of the concrete surface.</p>
      <form onSubmit={(event) => submit(event, onPredict)}>
        <div className={`dropzone ${dragging ? "is-dragging" : ""}`}
          onDragOver={(event) => { event.preventDefault(); if (!busy) setDragging(true); }}
          onDragLeave={() => setDragging(false)} onDrop={drop}>
          <svg width="32" height="32" viewBox="0 0 32 32" fill="none" aria-hidden="true">
            <path d="M16 21V6m0 0-6 6m6-6 6 6M6 21v5h20v-5" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <strong>Drop your image here</strong>
          <span className="muted">or choose a file from your device</span>
          <input ref={picker} id="image-upload" type="file" accept="image/jpeg,image/png" className="sr-only" tabIndex={-1}
            aria-label="Choose JPEG or PNG image" disabled={!!busy || checking}
            onChange={(event) => { void choose(Array.from(event.target.files || [])); event.target.value = ""; }} />
          <button type="button" className="button secondary" disabled={!!busy || checking} aria-controls="image-upload"
            onClick={() => picker.current?.click()}>Choose image</button>
        </div>
        <p className="field-help">JPEG or PNG · up to 10 MiB. RGB or grayscale; additional limits are checked by the server.</p>
        {checking && <p role="status" className="muted">Checking file content…</p>}
        {error && <p className="inline-error" role="alert">{error}</p>}
        {file && <div className="preview-block">
          {preview && <img src={preview} alt={`Local preview: ${file.name}`} className="upload-preview"
            onError={() => setError("This browser could not preview the file. The backend will validate the image.")} />}
          <div className="file-row"><div><strong className="file-name">{file.name}</strong>
            <span className="muted small">{(file.size / 1024 ** 2).toFixed(2)} MiB</span></div>
            <button type="button" className="text-button" disabled={!!busy}
              onClick={() => { validation.current += 1; onFile(null); setError(null); }}>Remove</button></div>
          <p className="field-help">Local preview only. Results use the server-oriented image so masks and boxes align.</p>
        </div>}
        <button className="button primary full-width" type="submit" disabled={!file || !!busy || checking}>
          {busy === "predict" ? "Analyzing image…" : "Analyze image"}
        </button>
      </form>
    </section>
    <section className="panel input-panel">
      <p className="eyebrow">02 / Existing inspection</p>
      <h2>Retrieve a result</h2>
      <p className="muted">Open a saved inspection using its UUID.</p>
      <form onSubmit={(event) => submit(event, onRetrieve)}>
        <label htmlFor="inspection-id">Inspection UUID</label>
        <input id="inspection-id" className="uuid-input" type="text" value={inspectionId}
          placeholder="Paste an inspection UUID" spellCheck={false} autoComplete="off"
          maxLength={80} disabled={!!busy} onChange={(event) => onId(event.target.value)}
          aria-describedby="uuid-help" />
        <p className="field-help" id="uuid-help">Only the specified result is loaded; there is no inspection listing endpoint.</p>
        <button type="submit" className="button secondary full-width" disabled={!inspectionId.trim() || !!busy || checking}>
          {busy === "retrieve" ? "Loading inspection…" : "Load inspection"}
        </button>
      </form>
    </section>
    <p className="sidebar-footnote">Segmentation categories describe predicted image content. They do not establish structural condition or safety.</p>
  </aside>;
}
