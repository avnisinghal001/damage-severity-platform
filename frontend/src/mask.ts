import { artifactBitmap, artifactUrl } from "./api";
import { CLASSES } from "./palette";
import type { ClassId, Inspection } from "./types";

export interface Scene {
  image: ImageBitmap;
  foreground: HTMLCanvasElement;
  classIds: Uint8Array | null;
  renderFiltered(activeClassId: ClassId | null): HTMLCanvasElement;
  dispose(): void;
}

/** Decode the L-mode PNG at native size. Never resize or infer IDs from display colors. */
export async function loadScene(inspection: Inspection, signal: AbortSignal, includeMask = true): Promise<Scene> {
  let image: ImageBitmap | undefined;
  let mask: ImageBitmap | undefined;
  const scratch = document.createElement("canvas");
  const foreground = document.createElement("canvas");
  const filteredCanvas = document.createElement("canvas");
  let lastFilteredClass: ClassId | null | undefined = undefined;

  try {
    // Sequential loading keeps cleanup deterministic if one artifact fails.
    image = await artifactBitmap(artifactUrl(inspection, "image"), signal);
    if (!includeMask) {
      if (signal.aborted) throw new DOMException("Aborted", "AbortError");
      if (image.width !== inspection.width || image.height !== inspection.height) {
        throw new Error("Original image dimensions disagree with this inspection.");
      }
      const original = image;
      return {
        image: original,
        foreground,
        classIds: null,
        renderFiltered: () => foreground,
        dispose: () => {
          original.close();
          foreground.width = foreground.height = 0;
          filteredCanvas.width = filteredCanvas.height = 0;
        },
      };
    }
    mask = await artifactBitmap(artifactUrl(inspection, "mask"), signal);
    if (signal.aborted) throw new DOMException("Aborted", "AbortError");
    for (const bitmap of [image, mask]) {
      if (bitmap.width !== inspection.width || bitmap.height !== inspection.height) {
        throw new Error("Image/mask dimensions disagree. The overlay was not displayed.");
      }
    }
    scratch.width = foreground.width = filteredCanvas.width = inspection.width;
    scratch.height = foreground.height = filteredCanvas.height = inspection.height;
    const source = scratch.getContext("2d", { willReadFrequently: true });
    const target = foreground.getContext("2d");
    if (!source || !target) throw new Error("Canvas is unavailable in this browser.");
    source.imageSmoothingEnabled = false;
    source.drawImage(mask, 0, 0);
    const raw = source.getImageData(0, 0, scratch.width, scratch.height).data;
    const colored = target.createImageData(scratch.width, scratch.height);
    const classIds = new Uint8Array(scratch.width * scratch.height);
    for (let i = 0, px = 0; i < raw.length; i += 4, px += 1) {
      const id = raw[i];
      if (id > 8 || raw[i + 1] !== id || raw[i + 2] !== id || raw[i + 3] !== 255) {
        throw new Error("Expected an opaque grayscale class-ID mask with values 0–8.");
      }
      classIds[px] = id;
      const rgb = CLASSES[id].rgb;
      colored.data[i] = rgb[0];
      colored.data[i + 1] = rgb[1];
      colored.data[i + 2] = rgb[2];
      colored.data[i + 3] = id === 0 ? 0 : 255;
    }
    target.putImageData(colored, 0, 0);
    const original = image;

    const renderFiltered = (activeClassId: ClassId | null): HTMLCanvasElement => {
      if (activeClassId === null) return foreground;
      if (lastFilteredClass === activeClassId) return filteredCanvas;
      const ctx = filteredCanvas.getContext("2d");
      if (!ctx) return foreground;
      const imgData = ctx.createImageData(filteredCanvas.width, filteredCanvas.height);
      for (let px = 0, i = 0; px < classIds.length; px += 1, i += 4) {
        const id = classIds[px];
        if (id === 0) continue;
        const rgb = CLASSES[id].rgb;
        imgData.data[i] = rgb[0];
        imgData.data[i + 1] = rgb[1];
        imgData.data[i + 2] = rgb[2];
        imgData.data[i + 3] = id === activeClassId ? 255 : 36;
      }
      ctx.putImageData(imgData, 0, 0);
      lastFilteredClass = activeClassId;
      return filteredCanvas;
    };

    return {
      image: original,
      foreground,
      classIds,
      renderFiltered,
      dispose: () => {
        original.close();
        foreground.width = foreground.height = 0;
        filteredCanvas.width = filteredCanvas.height = 0;
      },
    };
  } catch (error) {
    image?.close();
    foreground.width = foreground.height = 0;
    filteredCanvas.width = filteredCanvas.height = 0;
    throw error;
  } finally {
    mask?.close();
    scratch.width = scratch.height = 0;
  }
}
