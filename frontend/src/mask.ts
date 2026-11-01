import { artifactBitmap, artifactUrl } from "./api";
import { CLASSES } from "./palette";
import type { Inspection } from "./types";

export interface Scene {
  image: ImageBitmap;
  foreground: HTMLCanvasElement;
  dispose(): void;
}

/** Decode the L-mode PNG at native size. Never resize or infer IDs from display colors. */
export async function loadScene(inspection: Inspection, signal: AbortSignal): Promise<Scene> {
  let image: ImageBitmap | undefined;
  let mask: ImageBitmap | undefined;
  const scratch = document.createElement("canvas");
  const foreground = document.createElement("canvas");
  try {
    // Sequential loading keeps cleanup deterministic if one artifact fails.
    image = await artifactBitmap(artifactUrl(inspection, "image"), signal);
    mask = await artifactBitmap(artifactUrl(inspection, "mask"), signal);
    if (signal.aborted) throw new DOMException("Aborted", "AbortError");
    for (const bitmap of [image, mask]) {
      if (bitmap.width !== inspection.width || bitmap.height !== inspection.height) {
        throw new Error("Image/mask dimensions disagree. The overlay was not displayed.");
      }
    }
    scratch.width = foreground.width = inspection.width;
    scratch.height = foreground.height = inspection.height;
    const source = scratch.getContext("2d", { willReadFrequently: true });
    const target = foreground.getContext("2d");
    if (!source || !target) throw new Error("Canvas is unavailable in this browser.");
    source.imageSmoothingEnabled = false;
    source.drawImage(mask, 0, 0);
    const raw = source.getImageData(0, 0, scratch.width, scratch.height).data;
    const colored = target.createImageData(scratch.width, scratch.height);
    for (let i = 0; i < raw.length; i += 4) {
      const id = raw[i];
      if (id > 8 || raw[i + 1] !== id || raw[i + 2] !== id || raw[i + 3] !== 255) {
        throw new Error("Expected an opaque grayscale class-ID mask with values 0–8.");
      }
      const rgb = CLASSES[id].rgb;
      colored.data[i] = rgb[0];
      colored.data[i + 1] = rgb[1];
      colored.data[i + 2] = rgb[2];
      colored.data[i + 3] = id === 0 ? 0 : 255;
    }
    target.putImageData(colored, 0, 0);
    const original = image;
    return { image: original, foreground, dispose: () => {
      original.close();
      foreground.width = foreground.height = 0;
    } };
  } catch (error) {
    image?.close();
    foreground.width = foreground.height = 0;
    throw error;
  } finally {
    mask?.close();
    scratch.width = scratch.height = 0;
  }
}
