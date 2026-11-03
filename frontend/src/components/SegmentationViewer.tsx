import { useEffect, useMemo, useRef, useState } from "react";
import { artifactUrl, errorMessage } from "../api";
import { loadScene } from "../mask";
import type { Scene } from "../mask";
import { CLASSES, classColor } from "../palette";
import type { ClassId, DamageRegion, Inspection } from "../types";

type View = "original" | "overlay" | "mask";

interface Props {
  inspection: Inspection;
  selected: DamageRegion | null;
  selectedClassId: ClassId | null;
  onClear(): void;
}

/** Halve repeatedly until within 2x of the target, so every source pixel contributes (no aliasing). */
function stepDown(source: CanvasImageSource & { width: number; height: number }, width: number, height: number): CanvasImageSource {
  let current: CanvasImageSource & { width: number; height: number } = source;
  while (current.width / 2 >= width && current.height / 2 >= height) {
    const half = document.createElement("canvas");
    half.width = Math.max(1, Math.floor(current.width / 2));
    half.height = Math.max(1, Math.floor(current.height / 2));
    const context = half.getContext("2d");
    if (!context) return current;
    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = "high";
    context.drawImage(current, 0, 0, half.width, half.height);
    current = half;
  }
  return current;
}

export function SegmentationViewer({
  inspection,
  selected,
  selectedClassId,
  onClear,
}: Props) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const viewport = useRef<HTMLDivElement>(null);
  const [loaded, setLoaded] = useState<{ inspection: Inspection; scene: Scene } | null>(null);
  const scene = loaded?.inspection === inspection ? loaded.scene : null;
  const [fitWidth, setFitWidth] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [reload, setReload] = useState(0);
  const [view, setView] = useState<View>("overlay");
  const [opacity, setOpacity] = useState(62);
  const [native, setNative] = useState(false);

  const undamaged =
    inspection.severity?.status === "available" &&
    inspection.severity.label === "Undamaged";
  const displayedView = undamaged ? "original" : view;

  // Ensure overlay is visible when the user selects a class or region from the card.
  useEffect(() => {
    if (!undamaged && (selectedClassId !== null || selected !== null) && view === "original") {
      setView("overlay");
    }
  }, [selectedClassId, selected, undamaged, view]);

  useEffect(() => {
    const controller = new AbortController();
    let owned: Scene | null = null;
    setLoaded(null);
    setError(null);
    void loadScene(inspection, controller.signal, !undamaged)
      .then((loadedScene) => {
        if (controller.signal.aborted) {
          loadedScene.dispose();
          return;
        }
        owned = loadedScene;
        setLoaded({ inspection, scene: loadedScene });
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) setError(errorMessage(reason));
      });
    return () => {
      controller.abort();
      owned?.dispose();
    };
  }, [inspection, reload, undamaged]);

  useEffect(() => {
    const element = viewport.current;
    if (!element) return;
    const observer = new ResizeObserver(() => {
      const style = getComputedStyle(element);
      const width =
        element.clientWidth -
        parseFloat(style.paddingLeft) -
        parseFloat(style.paddingRight);
      const height =
        element.clientHeight -
        parseFloat(style.paddingTop) -
        parseFloat(style.paddingBottom);
      setFitWidth(
        Math.max(1, Math.min(width, (height * inspection.width) / inspection.height))
      );
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, [scene, inspection.width, inspection.height]);

  const activeFilterClass: ClassId | null =
    selectedClassId ?? (selected ? selected.class_id : null);

  // Draw at the size it is shown (device pixels), never larger than the photo itself. Letting the
  // browser shrink a full-size canvas with CSS is low quality, so shrink here in gentle steps.
  const outWidth = native || !fitWidth
    ? inspection.width
    : Math.min(inspection.width, Math.max(1, Math.round(fitWidth * (window.devicePixelRatio || 1))));
  const outHeight = Math.max(1, Math.round((outWidth * inspection.height) / inspection.width));

  // Step-down is the costly part; redo it only when the size changes, not on every slider move.
  const photo = useMemo(
    () => (scene ? stepDown(scene.image, outWidth, outHeight) : null),
    [scene, outWidth, outHeight],
  );

  useEffect(() => {
    if (!scene || !photo || !canvas.current) return;
    const target = canvas.current;
    target.width = outWidth;
    target.height = outHeight;
    const context = target.getContext("2d");
    if (!context) {
      setError("Canvas is unavailable in this browser.");
      return;
    }
    const scaled = outWidth < inspection.width;
    context.imageSmoothingEnabled = scaled;
    context.imageSmoothingQuality = "high";
    context.globalAlpha = 1;
    context.clearRect(0, 0, target.width, target.height);
    if (displayedView === "mask") {
      context.fillStyle = "#1c201d";
      context.fillRect(0, 0, target.width, target.height);
    } else {
      context.drawImage(photo, 0, 0, outWidth, outHeight);
    }
    if (displayedView !== "original") {
      const maskLayer = scene.renderFiltered(activeFilterClass);
      const baseAlpha = displayedView === "overlay" ? opacity / 100 : 1;
      context.globalAlpha =
        activeFilterClass !== null && displayedView === "overlay"
          ? Math.max(0.78, baseAlpha)
          : baseAlpha;
      context.imageSmoothingEnabled = scaled && displayedView !== "mask";
      context.drawImage(maskLayer, 0, 0, outWidth, outHeight);
    }
    context.globalAlpha = 1;
  }, [
    scene,
    photo,
    displayedView,
    opacity,
    activeFilterClass,
    inspection.width,
    inspection.height,
    outWidth,
    outHeight,
  ]);

  const highlightedRegions = useMemo(() => {
    if (undamaged) return [];
    if (selected) return [selected];
    if (selectedClassId !== null) {
      return inspection.analytics.regions.filter(
        (r) => r.class_id === selectedClassId
      );
    }
    return [];
  }, [undamaged, selected, selectedClassId, inspection.analytics.regions]);

  const activeClassLabel =
    activeFilterClass !== null
      ? CLASSES.find((c) => c.id === activeFilterClass)?.label ?? null
      : null;

  return (
    <section className="viewer-panel" aria-label="Photo viewer">
      {error && (
        <div className="viewer-placeholder">
          <p className="inline-error" role="alert">
            {error}
          </p>
          <button
            type="button"
            className="button secondary"
            onClick={() => setReload(reload + 1)}
          >
            Try again
          </button>
        </div>
      )}
      <div
        ref={viewport}
        className={`canvas-viewport ${native ? "native-size" : ""}`}
        hidden={!scene || !!error}
        tabIndex={0}
        aria-label="Photo, scroll to explore at full size"
      >
        <div
          className={`canvas-scene ${displayedView === "mask" ? "mask-view" : ""}`}
          style={{ width: `${native ? inspection.width : fitWidth}px` }}
        >
          <canvas
            ref={canvas}
            width={inspection.width}
            height={inspection.height}
            role="img"
            aria-label={`${displayedView} view, ${inspection.width} by ${inspection.height} pixels.${undamaged ? " Undamaged; no highlights." : ""}`}
          />
          {!undamaged && highlightedRegions.length > 0 && scene && (
            <svg
              className="bbox-layer"
              viewBox={`0 0 ${inspection.width} ${inspection.height}`}
              aria-label={
                selected
                  ? `Highlighted bounding box of region ${selected.region_id}`
                  : `Highlighted regions for ${activeClassLabel ?? "selected category"}`
              }
              role="img"
            >
              <title>
                {selected
                  ? `${selected.source_label}, region ${selected.region_id}`
                  : `${activeClassLabel ?? "Selected category"} (${highlightedRegions.length} regions)`}
              </title>
              {highlightedRegions.map((reg) => {
                const x = reg.bbox_xyxy_exclusive[0];
                const y = reg.bbox_xyxy_exclusive[1];
                const w = reg.bbox_xyxy_exclusive[2] - reg.bbox_xyxy_exclusive[0];
                const h = reg.bbox_xyxy_exclusive[3] - reg.bbox_xyxy_exclusive[1];
                const strokeColor = classColor(reg.class_id);
                return (
                  <g key={reg.region_id}>
                    <rect
                      className="bbox-halo"
                      x={x}
                      y={y}
                      width={w}
                      height={h}
                      vectorEffect="non-scaling-stroke"
                    />
                    <rect
                      className="bbox-outline"
                      x={x}
                      y={y}
                      width={w}
                      height={h}
                      style={{ stroke: strokeColor }}
                      vectorEffect="non-scaling-stroke"
                    />
                  </g>
                );
              })}
            </svg>
          )}
        </div>
      </div>
      <div className="viewer-dock">
        <div className="segmented" role="group" aria-label="Image view">
          <button
            type="button"
            aria-pressed={displayedView === "original"}
            onClick={() => setView("original")}
          >
            Photo
          </button>
          <button
            type="button"
            aria-pressed={displayedView === "overlay"}
            disabled={undamaged}
            onClick={() => setView("overlay")}
            title={
              undamaged
                ? "Highlights are hidden for Undamaged results"
                : "Show damage highlights on photo"
            }
          >
            Damage
          </button>
          <button
            type="button"
            aria-pressed={displayedView === "mask"}
            disabled={undamaged}
            onClick={() => setView("mask")}
            title={
              undamaged
                ? "Mask view is hidden for Undamaged results"
                : "Show segmented mask view"
            }
          >
            Mask
          </button>
        </div>
        {!undamaged && (
          <div className="opacity-control">
            <label className="sr-only" htmlFor="opacity">
              Highlight strength
            </label>
            <input
              id="opacity"
              type="range"
              min="0"
              max="100"
              value={opacity}
              title="Highlight strength"
              onChange={(event) => setOpacity(Number(event.target.value))}
              disabled={displayedView !== "overlay"}
              aria-valuetext={`${opacity} percent`}
            />
          </div>
        )}
        <details className="viewer-options">
          <summary aria-label="More image options" title="Colors, zoom and download">
            •••
          </summary>
          <div className="viewer-popover">
            <button
              type="button"
              className="menu-action"
              onClick={() => setNative(!native)}
              aria-pressed={native}
            >
              {native ? "Fit image" : "View full size"}
            </button>
            {!undamaged && (
              <button
                type="button"
                className="menu-action"
                onClick={() => setView("mask")}
                aria-pressed={displayedView === "mask"}
              >
                Segmented mask only
              </button>
            )}
            <a
              className="menu-action"
              href={artifactUrl(inspection, "mask")}
              download={`inspection-${inspection.id}-ids.png`}
            >
              Download mask
            </a>
            {!undamaged && (
              <ul className="legend" aria-label="Damage colors">
                {CLASSES.filter((item) => item.id > 0).map((item) => (
                  <li key={item.id}>
                    <span
                      className="swatch"
                      style={{ backgroundColor: classColor(item.id) }}
                      aria-hidden="true"
                    />
                    {item.label}
                  </li>
                ))}
              </ul>
            )}
          </div>
        </details>
      </div>
      <div className="viewer-caption">
        <span>
          {inspection.width.toLocaleString()} × {inspection.height.toLocaleString()}
        </span>
        {undamaged && (
          <span className="undamaged-note" role="status">
            No highlights
          </span>
        )}
        {!undamaged && (selected || selectedClassId !== null) && (
          <button type="button" className="text-button" onClick={onClear}>
            Clear selection
          </button>
        )}
      </div>
    </section>
  );
}
