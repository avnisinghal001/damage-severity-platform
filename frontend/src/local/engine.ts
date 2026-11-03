/** In-browser inference: three ONNX models, loaded in parallel, run with onnxruntime-web (WASM). */
import * as ort from "onnxruntime-web/wasm";
import { CLASSES } from "../palette";
import type { ClassifierProvenance, CrackResult, Inspection, SeverityResult } from "../types";
import { classifierTensor, flipHorizontal, restoreMask, rgbaToRgb, segmentationTensor, softmax, softmaxMap } from "./preprocess";

ort.env.wasm.wasmPaths = "/ort-1220/";
ort.env.wasm.numThreads = 1; // runs on every browser; no cross-origin isolation required
ort.env.wasm.proxy = false;

export type Role = "segmentation" | "severity" | "crack";
export const ROLES: Role[] = ["segmentation", "severity", "crack"];
export interface Progress { loaded: number; total: number; done: boolean }
export type ProgressMap = Record<Role, Progress>;

interface Entry { file: string; sha256: string; epoch: number; bytes: number; dataset: { release_id?: string };
  mean: number[]; std: number[] }
interface Manifest {
  segmentation: Entry & { height: number; width: number; classes: number };
  crack: Entry & { size: number; threshold: number };
  severity: Entry & { size: number; classes: string[] };
}
export interface Engine {
  manifest: Manifest;
  sessions: Record<Role, ort.InferenceSession>;
}

const MAX_BYTES = 10 * 1024 * 1024;
const MAX_PIXELS = 12_000_000;

async function fetchBytes(url: string, expected: number, onBytes: (n: number) => void, signal?: AbortSignal) {
  const response = await fetch(url, { signal });
  if (!response.ok || !response.body) throw new Error(`Could not download ${url} (HTTP ${response.status}).`);
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onBytes(Math.min(received, expected));
  }
  const all = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) { all.set(chunk, offset); offset += chunk.length; }
  return all;
}

/** Download and initialise all three models at the same time. */
export async function loadEngine(onProgress: (role: Role, progress: Progress) => void, signal?: AbortSignal): Promise<Engine> {
  const manifest = await (await fetch("/models/manifest.json", { signal })).json() as Manifest;
  const sessions = await Promise.all(ROLES.map(async (role) => {
    const entry = manifest[role];
    onProgress(role, { loaded: 0, total: entry.bytes, done: false });
    let bytes: Uint8Array | undefined;
    for (let attempt = 1; !bytes; attempt += 1) {
      try {
        bytes = await fetchBytes(`/models/${entry.file}`, entry.bytes,
          (loaded) => onProgress(role, { loaded, total: entry.bytes, done: false }), signal);
      } catch (reason) {
        if (signal?.aborted || attempt === 3) throw reason;
        await new Promise((resolve) => setTimeout(resolve, 800 * attempt));
      }
    }
    const session = await ort.InferenceSession.create(bytes, {
      executionProviders: ["wasm"], graphOptimizationLevel: "all",
    });
    onProgress(role, { loaded: entry.bytes, total: entry.bytes, done: true });
    return session;
  }));
  return { manifest, sessions: { segmentation: sessions[0], severity: sessions[1], crack: sessions[2] } };
}

const LIMITS = [
  "Uncalibrated softmax scores; arbitrary photographs are not a validated input domain.",
  "Predictions of visible damage only, not a structural safety assessment.",
];

function provenance(entry: Entry, architecture: "resnet18"): ClassifierProvenance {
  return { checkpoint_sha256: entry.sha256, epoch: entry.epoch, architecture,
    dataset: entry.dataset ?? {}, device: "cpu", limitations: LIMITS };
}

function canvasBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) =>
    canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error("Could not encode the image."))), "image/png"));
}

async function decode(file: File) {
  if (file.size > MAX_BYTES) throw new Error("Choose an image no larger than 10 MiB.");
  let bitmap: ImageBitmap;
  try {
    bitmap = await createImageBitmap(file); // applies EXIF orientation
  } catch {
    throw new Error("The image is malformed or cannot be decoded.");
  }
  const { width, height } = bitmap;
  if (width * height > MAX_PIXELS || Math.max(width, height) > 8192) {
    bitmap.close();
    throw new Error("Image dimensions exceed the supported limit (12 megapixels).");
  }
  const canvas = document.createElement("canvas");
  canvas.width = width; canvas.height = height;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("Canvas is unavailable in this browser.");
  context.fillStyle = "#fff"; // flatten transparency like a photo
  context.fillRect(0, 0, width, height);
  context.drawImage(bitmap, 0, 0);
  bitmap.close();
  return { canvas, context, width, height };
}

export async function analyze(engine: Engine, file: File, signal?: AbortSignal): Promise<Inspection> {
  const { canvas, context, width, height } = await decode(file);
  if (signal?.aborted) throw new DOMException("Aborted", "AbortError");
  const { manifest, sessions } = engine;
  const rgb = rgbaToRgb(context.getImageData(0, 0, width, height).data, width, height);

  const seg = manifest.segmentation, crack = manifest.crack, sev = manifest.severity;
  const segIn = segmentationTensor(rgb, seg.height, seg.width, seg.mean, seg.std);
  const crackIn = classifierTensor(rgb, crack.size, crack.mean, crack.std);
  const sevIn = classifierTensor(rgb, sev.size, sev.mean, sev.std);
  const run = (role: Role, data: Float32Array, size: number[]) =>
    sessions[role].run({ input: new ort.Tensor("float32", data, size) });
  const scoresOf = (output: Record<string, ort.Tensor>) => Object.values(output)[0].data as Float32Array;

  // Every model sees the photo and its mirror image; the two predictions are averaged.
  const plane = seg.height * seg.width;
  const flipped = {
    segmentation: flipHorizontal(segIn.tensor, 3, seg.height, seg.width),
    crack: flipHorizontal(crackIn, 3, crack.size, crack.size),
    severity: flipHorizontal(sevIn, 3, sev.size, sev.size),
  };
  // All six passes are queued as one parallel batch.
  const [segA, segB, crackA, crackB, sevA, sevB] = await Promise.all([
    run("segmentation", segIn.tensor, [1, 3, seg.height, seg.width]),
    run("segmentation", flipped.segmentation, [1, 3, seg.height, seg.width]),
    run("crack", crackIn, [1, 3, crack.size, crack.size]),
    run("crack", flipped.crack, [1, 3, crack.size, crack.size]),
    run("severity", sevIn, [1, 3, sev.size, sev.size]),
    run("severity", flipped.severity, [1, 3, sev.size, sev.size]),
  ]);
  if (signal?.aborted) throw new DOMException("Aborted", "AbortError");

  const classCount = seg.classes;
  const mirrored = flipHorizontal(softmaxMap(scoresOf(segB), classCount, plane), classCount, seg.height, seg.width);
  const direct = softmaxMap(scoresOf(segA), classCount, plane);
  const scores = new Float32Array(direct.length);
  for (let i = 0; i < scores.length; i += 1) scores[i] = (direct[i] + mirrored[i]) / 2;
  const ids = restoreMask(scores, classCount, seg.width, seg.height, segIn.frame, width, height);
  const counts = new Array<number>(9).fill(0);
  for (let i = 0; i < ids.length; i += 1) counts[ids[i]] += 1;
  const total = width * height;

  const average = (a: ArrayLike<number>, b: ArrayLike<number>) => {
    const pa = softmax(a), pb = softmax(b);
    return pa.map((value, i) => (value + pb[i]) / 2);
  };
  const crackProbability = average(scoresOf(crackA), scoresOf(crackB))[1];
  const crackClass = crackProbability >= crack.threshold ? 1 : 0;
  const crackResult: CrackResult = {
    class_id: crackClass as 0 | 1, label: crackClass ? "crack" : "no_crack",
    crack_probability: crackProbability, calibrated: false, decision_threshold: crack.threshold,
    model: provenance(crack, "resnet18"),
  };

  const probs = average(scoresOf(sevA), scoresOf(sevB));
  const best = probs.indexOf(Math.max(...probs));
  const entropy = -probs.reduce((s, p) => s + p * Math.log(Math.max(p, 1e-12)), 0) / Math.log(probs.length);
  const severity: SeverityResult = {
    status: "available", class_id: best as 0 | 1 | 2 | 3,
    label: sev.classes[best] as "Heavy" | "Minor" | "Moderate" | "Undamaged",
    confidence: probs[best], calibrated: false,
    probabilities: probs.map((probability, i) => ({ class_id: i, label: sev.classes[i], probability })),
    normalized_entropy: Math.min(1, Math.max(0, entropy)), width, height,
    model: provenance(sev, "resnet18"),
  };

  // Native-size L-mode style mask (class ID in every channel), read back by mask.ts.
  const maskCanvas = document.createElement("canvas");
  maskCanvas.width = width; maskCanvas.height = height;
  const maskContext = maskCanvas.getContext("2d")!;
  const maskData = maskContext.createImageData(width, height);
  for (let i = 0, j = 0; i < ids.length; i += 1, j += 4) {
    maskData.data[j] = maskData.data[j + 1] = maskData.data[j + 2] = ids[i];
    maskData.data[j + 3] = 255;
  }
  maskContext.putImageData(maskData, 0, 0);
  const [imageBlob, maskBlob] = await Promise.all([canvasBlob(canvas), canvasBlob(maskCanvas)]);

  const classes = CLASSES.map((c) => ({
    class_id: c.id as Inspection["analytics"]["classes"][number]["class_id"], source_label: c.label,
    area_pixels: counts[c.id], image_area_percent: (100 * counts[c.id]) / total, connected_regions: null,
  }));
  return {
    id: crypto.randomUUID(), created_at: new Date().toISOString(), status: "completed", width, height,
    image: { url: URL.createObjectURL(imageBlob), sha256: "", media_type: "image/png" },
    mask: { url: URL.createObjectURL(maskBlob), sha256: "", media_type: "image/png" },
    detected_classes: classes.filter((c) => c.class_id > 0 && c.area_pixels > 0),
    model: {
      checkpoint_sha256: seg.sha256, checkpoint_epoch: seg.epoch, dataset: seg.dataset ?? {}, device: "cpu",
      class_mapping: CLASSES.map((c) => ({ class_id: c.id, source_label: c.label })),
      alignment: "EXIF-oriented resolution; argmax, unpad, nearest-neighbor restoration",
      confidence: "Uncalibrated softmax", severity_scope: "Independent Phi-Net image prediction",
    },
    classification: crackResult, severity, features: null,
    analytics: {
      schema_version: 1, measurement_space: "image_pixels", image: { width, height, area_pixels: total },
      classes, regions: [], uncertainty: { calibrated: false, status: "uncalibrated_softmax_only" },
      spalling_summary: { status: "not_assessed", label: null, candidate_source_labels: [],
        unsupported_class_ids: [], supervision: "none", structural_severity_or_safety: "not_assessed" },
      limitations: LIMITS,
    },
  };
}
