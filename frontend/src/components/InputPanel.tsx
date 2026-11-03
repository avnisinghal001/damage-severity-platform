import { useCallback, useEffect, useRef, useState } from "react";
import type { DragEvent, FormEvent } from "react";
import { errorMessage } from "../api";
import { prepareClipboardImage, readClipboardImage, validateFile } from "../files";

const IS_MAC = /Mac|iPhone|iPad/.test(navigator.userAgent);
// Safari fires no paste event unless an editable field has focus, so ⌘V there reads the
// clipboard directly. Chrome, Edge and Firefox deliver a normal paste event.
const IS_SAFARI = /^((?!chrome|chromium|crios|android|edg).)*safari/i.test(navigator.userAgent);

function isEditable(target: EventTarget | null): boolean {
  return target instanceof Element
    && !!target.closest("input, textarea, select, [contenteditable]:not([contenteditable='false'])");
}

function clipboardMessage(reason: unknown): string {
  return reason instanceof DOMException && reason.name === "NotAllowedError"
    ? `Clipboard access was blocked. Press ${IS_MAC ? "⌘ V" : "Ctrl V"} on the page instead.`
    : errorMessage(reason);
}

const SAMPLES = [
  { src: "/samples/cracked-concrete.png", name: "Cracked concrete", tag: "01" },
  { src: "/samples/bridge-parapet.png", name: "Bridge parapet", tag: "02" },
  { src: "/samples/exposed-rebar.png", name: "Exposed rebar", tag: "03" },
];

interface Props {
  compact?: boolean;
  file: File | null;
  busy: "predict" | null;
  onFile(file: File | null): void;
  onPredict(): void;
}

export function InputPanel({ compact = false, file, busy, onFile, onPredict }: Props) {
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

  const choose = useCallback(async (files: File[], prepare: (file: File) => Promise<File> = validateFile) => {
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
      const accepted = await prepare(files[0]);
      if (token === validation.current) onFile(accepted);
    } catch (reason) {
      if (token === validation.current) setError(errorMessage(reason));
    } finally {
      if (token === validation.current) setChecking(false);
    }
  }, [busy, onFile]);

  // Counts paste events, so a direct clipboard read never duplicates one.
  const pastes = useRef(0);

  const pasteFromClipboard = useCallback(async (after?: number) => {
    try {
      const image = await readClipboardImage();
      if (after !== undefined && pastes.current !== after) return; // A paste event won.
      if (image) await choose([image], prepareClipboardImage);
      else setError("The clipboard has no image. Copy a photo or screenshot first.");
    } catch (reason) {
      if (after === undefined || pastes.current === after) setError(clipboardMessage(reason));
    }
  }, [choose]);

  useEffect(() => {
    function paste(event: ClipboardEvent) {
      // Leave text fields and rich-text editors to the browser's normal paste behavior.
      if (event.defaultPrevented || busy || isEditable(event.target)) return;
      pastes.current += 1;
      const files = Array.from(event.clipboardData?.files ?? []);
      const image = files.find((item) => item.type.startsWith("image/"));
      if (!files.length) return;
      event.preventDefault();
      if (!image) { setError("Only images can be pasted here."); return; }
      // Screenshots are transparent PNGs, often over the size limit: flatten and fit them.
      void choose([image], prepareClipboardImage);
    }
    function shortcut(event: KeyboardEvent) {
      if (busy || event.key.toLowerCase() !== "v" || event.altKey || event.shiftKey
        || isEditable(event.target)) return;
      // On macOS, Ctrl+V is not the paste shortcut, so no paste event follows it; Safari
      // also skips the paste event for ⌘V outside text fields. Read the clipboard directly.
      const controlOnMac = IS_MAC && event.ctrlKey && !event.metaKey;
      const commandOnSafari = IS_SAFARI && event.metaKey && !event.ctrlKey;
      if (!controlOnMac && !commandOnSafari) return;
      if (controlOnMac) event.preventDefault();
      void pasteFromClipboard(pastes.current);
    }
    document.addEventListener("paste", paste);
    document.addEventListener("keydown", shortcut);
    return () => {
      document.removeEventListener("paste", paste);
      document.removeEventListener("keydown", shortcut);
    };
  }, [busy, choose, pasteFromClipboard]);

  function drop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    if (!busy) void choose(Array.from(event.dataTransfer.files));
  }

  async function pickSample(src: string, name: string) {
    if (busy || checking) return;
    try {
      const blob = await (await fetch(src)).blob();
      await choose([new File([blob], `${name}.png`, { type: "image/png" })]);
    } catch {
      setError("That sample could not be loaded. Check your connection and try again.");
    }
  }

  function submit(event: FormEvent) {
    event.preventDefault();
    if (!busy && !checking) onPredict();
  }

  return <aside className="input-column" aria-label="Inspection inputs">
    <section className="panel input-panel">
      {!compact && <p className="eyebrow">01 / New inspection</p>}
      <h2>{file ? "Ready to inspect" : "Inspect a photo"}</h2>
      <form onSubmit={submit}>
        <div className={`dropzone ${dragging ? "is-dragging" : ""}`}
          onDragOver={(event) => { event.preventDefault(); if (!busy) setDragging(true); }}
          onDragLeave={() => setDragging(false)} onDrop={drop}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <path d="M12 16V4m0 0L7 9m5-5 5 5M4 16v4h16v-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <strong>Drop or paste an image</strong>
          <input ref={picker} id="image-upload" type="file" accept="image/jpeg,image/png" className="sr-only" tabIndex={-1}
            aria-label="Choose JPEG or PNG image" disabled={!!busy || checking}
            onChange={(event) => { void choose(Array.from(event.target.files || [])); event.target.value = ""; }} />
          <div className="dropzone-actions">
            <button type="button" className="button secondary" disabled={!!busy || checking} aria-controls="image-upload"
              onClick={() => picker.current?.click()}>Choose image</button>
            <button type="button" className="button secondary" disabled={!!busy || checking}
              onClick={() => void pasteFromClipboard()}>Paste</button>
          </div>
          <span className="paste-hint">Or press {IS_MAC ? <><kbd>⌘ V</kbd> or <kbd>Ctrl V</kbd></> : <kbd>Ctrl V</kbd>} anywhere on the page</span>
        </div>
        <p className="field-help">JPEG or PNG · up to 10 MB</p>
        {checking && <p role="status" className="muted">Checking file content…</p>}
        <span className="sr-only" role="status">{file ? `${file.name} ready to analyze` : ""}</span>
        {error && <p className="inline-error" role="alert">{error}</p>}
        {file && <div className="preview-block">
          {!compact && preview && <img src={preview} alt={`Local preview: ${file.name}`} className="upload-preview"
            onError={() => setError("This browser could not preview the file. Analysis will still run.")} />}
          <div className="file-row"><div><strong className="file-name">{file.name}</strong>
            <span className="muted small">{(file.size / 1024 ** 2).toFixed(2)} MiB</span></div>
            <button type="button" className="text-button" disabled={!!busy}
              onClick={() => { validation.current += 1; onFile(null); setError(null); }}>Remove</button></div>
          {!compact && <p className="field-help">Preview</p>}
        </div>}
        <button className="button primary full-width" type="submit" disabled={!file || !!busy || checking}>
          {busy === "predict" ? "Analyzing…" : "Analyze"}
        </button>
      </form>
      <div className="sample-block">
        <p className="sample-title"><span>Or pick a sample</span></p>
        <div className="sample-grid">
          {SAMPLES.map((sample) => <button key={sample.src} type="button" className="sample-card"
            disabled={!!busy || checking} onClick={() => void pickSample(sample.src, sample.name)}>
            <span className="sample-frame"><img src={sample.src} alt="" loading="lazy" draggable={false} /></span>
            <span className="sample-meta"><span className="sample-name">{sample.name}</span><span className="sample-tag">{sample.tag}</span></span>
          </button>)}
        </div>
      </div>
    </section>
  </aside>;
}
