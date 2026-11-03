/** Pillow-compatible image preprocessing, DOM-free so it can run in Node for parity checks. */

export interface RgbImage {
  data: Uint8ClampedArray; // interleaved RGB
  width: number;
  height: number;
}

/** Python's round(): halves go to the nearest even integer. */
export function pyRound(x: number): number {
  const floor = Math.floor(x);
  const diff = x - floor;
  if (diff < 0.5) return floor;
  if (diff > 0.5) return floor + 1;
  return floor % 2 === 0 ? floor : floor + 1;
}

export function rgbaToRgb(rgba: Uint8ClampedArray, width: number, height: number): RgbImage {
  const data = new Uint8ClampedArray(width * height * 3);
  for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) {
    data[j] = rgba[i];
    data[j + 1] = rgba[i + 1];
    data[j + 2] = rgba[i + 2];
  }
  return { data, width, height };
}

interface Kernel { start: Int32Array; size: Int32Array; weights: Float64Array; stride: number }

/** Mirrors Pillow's precompute_coeffs for the BILINEAR (triangle) filter. */
function kernel(inSize: number, outSize: number): Kernel {
  const scale = inSize / outSize;
  const filterScale = Math.max(scale, 1);
  const support = filterScale;
  const stride = Math.ceil(support) * 2 + 1;
  const start = new Int32Array(outSize);
  const size = new Int32Array(outSize);
  const weights = new Float64Array(outSize * stride);
  const inverse = 1 / filterScale;
  for (let out = 0; out < outSize; out += 1) {
    const center = (out + 0.5) * scale;
    let min = Math.trunc(center - support + 0.5);
    if (min < 0) min = 0;
    let max = Math.trunc(center + support + 0.5);
    if (max > inSize) max = inSize;
    max -= min;
    let total = 0;
    for (let x = 0; x < max; x += 1) {
      const distance = Math.abs((x + min - center + 0.5) * inverse);
      const weight = distance < 1 ? 1 - distance : 0;
      weights[out * stride + x] = weight;
      total += weight;
    }
    if (total !== 0) for (let x = 0; x < max; x += 1) weights[out * stride + x] /= total;
    start[out] = min;
    size[out] = max;
  }
  return { start, size, weights, stride };
}

const clamp8 = (v: number): number => (v < 0 ? 0 : v > 255 ? 255 : Math.floor(v + 0.5));

/** Separable resize with Pillow's horizontal-then-vertical passes and uint8 intermediate. */
export function resizeBilinear(src: RgbImage, width: number, height: number): RgbImage {
  const horizontal = kernel(src.width, width);
  const middle = new Uint8ClampedArray(width * src.height * 3);
  for (let y = 0; y < src.height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const from = horizontal.start[x];
      const count = horizontal.size[x];
      let r = 0, g = 0, b = 0;
      for (let k = 0; k < count; k += 1) {
        const w = horizontal.weights[x * horizontal.stride + k];
        const p = (y * src.width + from + k) * 3;
        r += w * src.data[p]; g += w * src.data[p + 1]; b += w * src.data[p + 2];
      }
      const o = (y * width + x) * 3;
      middle[o] = clamp8(r); middle[o + 1] = clamp8(g); middle[o + 2] = clamp8(b);
    }
  }
  const vertical = kernel(src.height, height);
  const data = new Uint8ClampedArray(width * height * 3);
  for (let y = 0; y < height; y += 1) {
    const from = vertical.start[y];
    const count = vertical.size[y];
    for (let x = 0; x < width; x += 1) {
      let r = 0, g = 0, b = 0;
      for (let k = 0; k < count; k += 1) {
        const w = vertical.weights[y * vertical.stride + k];
        const p = ((from + k) * width + x) * 3;
        r += w * middle[p]; g += w * middle[p + 1]; b += w * middle[p + 2];
      }
      const o = (y * width + x) * 3;
      data[o] = clamp8(r); data[o + 1] = clamp8(g); data[o + 2] = clamp8(b);
    }
  }
  return { data, width, height };
}

/** ImageOps.contain + mean-colour letterbox + normalisation, as in task07 prepare_image. */
export function classifierTensor(
  image: RgbImage, size: number, mean: number[], std: number[],
): Float32Array {
  const ratio = image.width / image.height;
  let width = size, height = size;
  if (ratio > 1) height = pyRound((image.height / image.width) * size);
  else if (ratio < 1) width = pyRound((image.width / image.height) * size);
  const resized = resizeBilinear(image, width, height);
  const fill = mean.map((m) => pyRound(m * 255));
  const left = (size - width) >> 1, top = (size - height) >> 1;
  const plane = size * size;
  const out = new Float32Array(3 * plane);
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const inside = x >= left && x < left + width && y >= top && y < top + height;
      const p = ((y - top) * width + (x - left)) * 3;
      for (let c = 0; c < 3; c += 1) {
        const value = inside ? resized.data[p + c] : fill[c];
        out[c * plane + y * size + x] = (value / 255 - mean[c]) / std[c];
      }
    }
  }
  return out;
}

export interface Frame { width: number; height: number; left: number; top: number }

/** Letterbox + normalisation for the segmentation model (task09 prepare_pair, inference). */
export function segmentationTensor(
  image: RgbImage, targetH: number, targetW: number, mean: number[], std: number[],
): { tensor: Float32Array; frame: Frame } {
  const scale = Math.min(targetW / image.width, targetH / image.height);
  const width = Math.max(1, Math.min(targetW, pyRound(image.width * scale)));
  const height = Math.max(1, Math.min(targetH, pyRound(image.height * scale)));
  const left = (targetW - width) >> 1, top = (targetH - height) >> 1;
  const resized = resizeBilinear(image, width, height);
  const plane = targetH * targetW;
  const out = new Float32Array(3 * plane); // padding stays exactly 0, as in Python
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const p = (y * width + x) * 3;
      for (let c = 0; c < 3; c += 1) {
        out[c * plane + (y + top) * targetW + (x + left)] = (resized.data[p + c] / 255 - mean[c]) / std[c];
      }
    }
  }
  return { tensor: out, frame: { width, height, left, top } };
}

/**
 * Unpad the class scores, upsample them bilinearly to the native size, then take the
 * per-pixel argmax. Interpolating scores (not class IDs) keeps region outlines smooth
 * instead of blocky at the model's 256-pixel working resolution.
 */
export function restoreMask(
  logits: ArrayLike<number>, classes: number, targetW: number, targetH: number,
  frame: Frame, width: number, height: number,
): Uint8Array {
  const plane = targetW * targetH;
  const out = new Uint8Array(width * height);
  const x0 = new Int32Array(width), x1 = new Int32Array(width), wx = new Float32Array(width);
  for (let x = 0; x < width; x += 1) {
    const position = Math.min(frame.width - 1, Math.max(0, ((x + 0.5) * frame.width) / width - 0.5));
    x0[x] = Math.floor(position);
    x1[x] = Math.min(frame.width - 1, x0[x] + 1);
    wx[x] = position - x0[x];
  }
  for (let y = 0; y < height; y += 1) {
    const position = Math.min(frame.height - 1, Math.max(0, ((y + 0.5) * frame.height) / height - 0.5));
    const y0 = Math.floor(position), y1 = Math.min(frame.height - 1, y0 + 1), wy = position - y0;
    const row0 = (frame.top + y0) * targetW + frame.left;
    const row1 = (frame.top + y1) * targetW + frame.left;
    for (let x = 0; x < width; x += 1) {
      const a = row0 + x0[x], b = row0 + x1[x], c = row1 + x0[x], d = row1 + x1[x];
      const fx = wx[x];
      let best = 0, bestScore = -Infinity;
      for (let k = 0; k < classes; k += 1) {
        const o = k * plane;
        const top = logits[o + a] * (1 - fx) + logits[o + b] * fx;
        const bottom = logits[o + c] * (1 - fx) + logits[o + d] * fx;
        const score = top * (1 - wy) + bottom * wy;
        if (score > bestScore) { bestScore = score; best = k; }
      }
      out[y * width + x] = best;
    }
  }
  return out;
}

/** Mirror a CHW tensor left-right (for the test-time-augmentation pass). */
export function flipHorizontal(source: Float32Array, channels: number, height: number, width: number): Float32Array {
  const out = new Float32Array(source.length);
  for (let c = 0; c < channels; c += 1) {
    for (let y = 0; y < height; y += 1) {
      const row = (c * height + y) * width;
      for (let x = 0; x < width; x += 1) out[row + x] = source[row + width - 1 - x];
    }
  }
  return out;
}

/** Per-pixel softmax over the class axis of a CHW score map. */
export function softmaxMap(logits: ArrayLike<number>, classes: number, plane: number): Float32Array {
  const out = new Float32Array(classes * plane);
  for (let i = 0; i < plane; i += 1) {
    let max = -Infinity;
    for (let k = 0; k < classes; k += 1) max = Math.max(max, logits[k * plane + i]);
    let total = 0;
    for (let k = 0; k < classes; k += 1) { const v = Math.exp(logits[k * plane + i] - max); out[k * plane + i] = v; total += v; }
    for (let k = 0; k < classes; k += 1) out[k * plane + i] /= total;
  }
  return out;
}

export function softmax(logits: ArrayLike<number>): number[] {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i += 1) max = Math.max(max, logits[i]);
  const exp = Array.from({ length: logits.length }, (_, i) => Math.exp(logits[i] - max));
  const total = exp.reduce((a, b) => a + b, 0);
  return exp.map((v) => v / total);
}
