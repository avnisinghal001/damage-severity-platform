import { useEffect, useRef, useState } from "react";
import { artifactUrl, errorMessage } from "../api";
import { loadScene } from "../mask";
import type { Scene } from "../mask";
import { CLASSES, classColor } from "../palette";
import type { DamageRegion, Inspection } from "../types";

type View = "original" | "overlay" | "mask";
interface Props { inspection: Inspection; selected: DamageRegion | null; onClear(): void }

export function SegmentationViewer({ inspection, selected, onClear }: Props) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [scene, setScene] = useState<Scene | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [reload, setReload] = useState(0);
  const [view, setView] = useState<View>("overlay");
  const [opacity, setOpacity] = useState(55);
  const [native, setNative] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    let owned: Scene | null = null;
    setScene(null);
    setError(null);
    void loadScene(inspection, controller.signal).then((loaded) => {
      if (controller.signal.aborted) { loaded.dispose(); return; }
      owned = loaded;
      setScene(loaded);
    }).catch((reason: unknown) => {
      if (!controller.signal.aborted) setError(errorMessage(reason));
    });
    return () => { controller.abort(); owned?.dispose(); };
  }, [inspection, reload]);

  useEffect(() => {
    if (!scene || !canvas.current) return;
    const target = canvas.current;
    target.width = inspection.width;
    target.height = inspection.height;
    const context = target.getContext("2d");
    if (!context) { setError("Canvas is unavailable in this browser."); return; }
    context.imageSmoothingEnabled = false;
    context.globalAlpha = 1;
    context.clearRect(0, 0, target.width, target.height);
    if (view === "mask") {
      context.fillStyle = classColor(0);
      context.fillRect(0, 0, target.width, target.height);
    } else {
      context.drawImage(scene.image, 0, 0);
    }
    if (view !== "original") {
      context.globalAlpha = view === "overlay" ? opacity / 100 : 1;
      context.drawImage(scene.foreground, 0, 0);
    }
    context.globalAlpha = 1;
  }, [scene, view, opacity, inspection.width, inspection.height]);

  return <section className="panel viewer-panel" aria-labelledby="viewer-title">
    <div className="section-header"><div><p className="eyebrow">Prediction workspace</p>
      <h2 id="viewer-title">Segmentation viewer</h2></div>
      <a className="text-button" href={artifactUrl(inspection, "mask")}
        download={`inspection-${inspection.id}-ids.png`}>Download ID mask ↗</a></div>
    <div className="viewer-toolbar">
      <div className="segmented" role="group" aria-label="Image view">
        {(["original", "overlay", "mask"] as const).map((mode) => <button key={mode} type="button"
          aria-pressed={view === mode} onClick={() => setView(mode)}>{mode === "mask" ? "Class mask" : mode[0].toUpperCase() + mode.slice(1)}</button>)}
      </div>
      <button className="button compact secondary" type="button" onClick={() => setNative(!native)}
        aria-pressed={native}>{native ? "Fit to panel" : "View 1:1 pixels"}</button>
    </div>
    <div className="opacity-control"><label htmlFor="opacity">Overlay opacity</label>
      <input id="opacity" type="range" min="0" max="100" value={opacity}
        onChange={(event) => setOpacity(Number(event.target.value))} disabled={view !== "overlay"}
        aria-valuetext={`${opacity} percent`} /><output htmlFor="opacity">{opacity}%</output></div>
    {!scene && !error && <div className="viewer-placeholder" role="status">Loading native image and class-ID mask…</div>}
    {error && <div className="viewer-placeholder"><p className="inline-error" role="alert">{error}</p>
      <button type="button" className="button secondary" onClick={() => setReload(reload + 1)}>Retry artifacts</button></div>}
    <div className="canvas-viewport" hidden={!scene || !!error} tabIndex={0} aria-label="Scrollable native image viewer">
      <div className={`canvas-scene ${view === "mask" ? "mask-view" : ""}`} style={{ width: native ? `${inspection.width}px` : "100%" }}>
        <canvas ref={canvas} width={inspection.width} height={inspection.height}
          role="img" aria-label={`${view} view, ${inspection.width} by ${inspection.height} native pixels. Class legend and region measurements follow.`} />
        {selected && scene && <svg className="bbox-layer" viewBox={`0 0 ${inspection.width} ${inspection.height}`}
          aria-label={`Highlighted bounding box of region ${selected.region_id}`} role="img">
          <title>{selected.source_label}, region {selected.region_id}</title>
          {["bbox-halo", "bbox-outline"].map((style) => <rect key={style} className={style}
            x={selected.bbox_xyxy_exclusive[0]} y={selected.bbox_xyxy_exclusive[1]}
            width={selected.bbox_xyxy_exclusive[2] - selected.bbox_xyxy_exclusive[0]}
            height={selected.bbox_xyxy_exclusive[3] - selected.bbox_xyxy_exclusive[1]}
            vectorEffect="non-scaling-stroke" />)}
        </svg>}
      </div>
    </div>
    <div className="viewer-caption"><span>{inspection.width.toLocaleString()} × {inspection.height.toLocaleString()} px · EXIF-oriented alignment</span>
      {selected && <button type="button" className="text-button" onClick={onClear}>Clear region #{selected.region_id}</button>}</div>
    <ul className="legend" aria-label="Exact project class-color legend">
      {CLASSES.map((item) => <li key={item.id}><span className="swatch" style={{ backgroundColor: classColor(item.id) }} aria-hidden="true" />
        <span><span className="legend-id">{item.id}</span>{item.label}</span></li>)}
    </ul>
    <p className="field-help legend-note">Background is transparent in overlay view and gray in class-mask view. Display scaling does not change the native mask or measurements.</p>
  </section>;
}
