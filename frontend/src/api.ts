import { CLASSES } from "./palette";
import type { HistoryPage, Inspection, ModelMetadata, Readiness } from "./types";

const API = "/api";
export const UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export class ApiError extends Error {
  constructor(message: string, readonly requestId?: string, readonly retryAfter?: string) {
    super(message);
    this.name = "ApiError";
  }
}

export function errorMessage(error: unknown): string {
  const message = error instanceof Error ? error.message : "The request could not be completed.";
  return error instanceof ApiError
    ? `${message}${error.retryAfter ? ` Retry after ${error.retryAfter} seconds.` : ""}${error.requestId ? ` Request ID: ${error.requestId}` : ""}`
    : message;
}

async function checked(response: Response): Promise<Response> {
  if (response.ok) return response;
  const body = await response.json().catch(() => null);
  const detail = body?.error;
  const message = typeof detail?.message === "string" ? detail.message
    : response.status === 503 ? "The backend is unavailable or busy. Try again shortly."
    : `Backend request failed (HTTP ${response.status}).`;
  throw new ApiError(message, detail?.request_id || response.headers.get("x-request-id") || undefined,
    response.headers.get("retry-after") || undefined);
}

async function json(path: string, options: RequestInit = {}): Promise<unknown> {
  let response: Response;
  try {
    response = await fetch(`${API}${path}`, { cache: "no-store", ...options });
  } catch (error) {
    if (options.signal?.aborted) throw error;
    throw new ApiError("Cannot reach the backend. Check that FastAPI and the Vite proxy are running.");
  }
  return (await checked(response)).json().catch(() => {
    throw new ApiError("The backend returned an invalid JSON response.");
  });
}

function checkModel(value: ModelMetadata): ModelMetadata {
  if (!value || !/^[0-9a-f]{64}$/.test(value.checkpoint_sha256)
    || !["cpu", "mps"].includes(value.device) || !Number.isSafeInteger(value.checkpoint_epoch)
    || !Array.isArray(value.class_mapping)
    || value.class_mapping.length !== 9 || !value.dataset
    || (value.dataset.release_id !== undefined && typeof value.dataset.release_id !== "string")
    || !value.class_mapping.every((row) => row && CLASSES.some((c) => c.id === row.class_id && c.label === row.source_label))
    || new Set(value.class_mapping.map((row) => row.class_id)).size !== 9) {
    throw new ApiError("The model's class mapping is incompatible with this dashboard.");
  }
  return value;
}

/** Reject incompatible coordinates/taxonomies rather than draw a misleading overlay. */
function checkInspection(value: unknown): Inspection {
  const v = value as Inspection;
  if (!v || !UUID_PATTERN.test(v.id) || v.status !== "completed"
    || !Number.isSafeInteger(v.width) || !Number.isSafeInteger(v.height) || v.width < 1 || v.height < 1
    || v.width * v.height > 16_777_216 || typeof v.image?.url !== "string" || typeof v.mask?.url !== "string"
    || v.analytics?.schema_version !== 1 || v.analytics.measurement_space !== "image_pixels"
    || v.analytics.image?.width !== v.width || v.analytics.image?.height !== v.height
    || !Array.isArray(v.analytics.classes) || !Array.isArray(v.analytics.regions)
    || !Array.isArray(v.detected_classes) || !v.analytics.uncertainty
    || typeof v.analytics.uncertainty.status !== "string"
    || !Array.isArray(v.analytics.limitations) || !v.analytics.limitations.every((item) => typeof item === "string")
    || typeof v.created_at !== "string" || !Number.isFinite(Date.parse(v.created_at))) {
    throw new ApiError("Unsupported inspection response or image size. Native alignment could not be verified.");
  }
  checkModel(v.model);
  checkClassifiers(v);
  if (v.analytics.classes.length !== 9 || new Set(v.analytics.classes.map((c) => c.class_id)).size !== 9
    || !v.analytics.classes.every((c) => CLASSES.some((known) => known.id === c.class_id && known.label === c.source_label)
      && Number.isSafeInteger(c.area_pixels) && c.area_pixels >= 0
      && (c.class_id === 0 ? c.connected_regions === null
        : typeof c.connected_regions === "number" && Number.isSafeInteger(c.connected_regions) && c.connected_regions >= 0)
      && Number.isFinite(c.image_area_percent) && c.image_area_percent >= 0 && c.image_area_percent <= 100)
    || v.analytics.classes.reduce((n, c) => n + c.area_pixels, 0) !== v.width * v.height) {
    throw new ApiError("The class-area report is incompatible or incomplete.");
  }
  for (const region of v.analytics.regions) {
    const box = region.bbox_xyxy_exclusive;
    if (!Number.isSafeInteger(region.region_id) || region.region_id < 1
      || !CLASSES.some((c) => c.id > 0 && c.id === region.class_id && c.label === region.source_label)
      || !Array.isArray(box) || box.length !== 4 || !box.every(Number.isSafeInteger)
      || box[0] < 0 || box[1] < 0 || box[2] > v.width || box[3] > v.height
      || box[0] >= box[2] || box[1] >= box[3]
      || !region.uncertainty || typeof region.uncertainty.status !== "string"
      || !Array.isArray(region.uncertainty.review_flags)
      || !region.uncertainty.review_flags.every((flag) => typeof flag === "string")
      || !Number.isSafeInteger(region.area_pixels) || region.area_pixels < 1
      || !Number.isFinite(region.image_area_percent) || region.image_area_percent < 0 || region.image_area_percent > 100
      || !region.spalling_assessment || typeof region.spalling_assessment.status !== "string"
      || (region.spalling_assessment.label !== null && typeof region.spalling_assessment.label !== "string")) {
      throw new ApiError("A region has invalid coordinates or unsupported analytics.");
    }
  }
  if (new Set(v.analytics.regions.map((row) => row.region_id)).size !== v.analytics.regions.length) {
    throw new ApiError("The region report contains duplicate identifiers.");
  }
  // Resolve both now so malformed references cannot become clickable URLs later.
  artifactUrl(v, "image");
  artifactUrl(v, "mask");
  return v;
}

function checkClassifiers(v: Inspection): void {
  const crack = v.classification;
  if (crack && (crack.calibrated !== false || ![0, 1].includes(crack.class_id)
    || crack.label !== ["no_crack", "crack"][crack.class_id]
    || !Number.isFinite(crack.crack_probability) || crack.crack_probability < 0 || crack.crack_probability > 1
    || !Number.isFinite(crack.decision_threshold) || crack.decision_threshold < 0 || crack.decision_threshold > 1.000001
    || crack.class_id !== Number(crack.crack_probability >= crack.decision_threshold)
    || !/^[0-9a-f]{64}$/.test(crack.model?.checkpoint_sha256))) {
    throw new ApiError("Invalid crack classification contract.");
  }
  const severity = v.severity;
  if (!severity) return;
  if (severity.status === "unavailable" && typeof severity.message === "string") return;
  if (severity.status !== "available") throw new ApiError("Invalid severity availability status.");
  const labels = ["Heavy", "Minor", "Moderate", "Undamaged"];
  if (severity.calibrated !== false || labels[severity.class_id] !== severity.label
    || severity.width !== v.width || severity.height !== v.height
    || !Number.isFinite(severity.confidence) || severity.confidence < 0 || severity.confidence > 1
    || !Array.isArray(severity.probabilities) || severity.probabilities.length !== 4
    || !severity.probabilities.every((p, i) => p.class_id === i && p.label === labels[i]
      && Number.isFinite(p.probability) && p.probability >= 0 && p.probability <= 1)
    || Math.abs(severity.probabilities.reduce((n, p) => n + p.probability, 0) - 1) > 1e-5
    || !/^[0-9a-f]{64}$/.test(severity.model?.checkpoint_sha256)) {
    throw new ApiError("Invalid Phi-Net class mapping, probabilities or image alignment.");
  }
}

export async function getHistory(signal: AbortSignal, cursor?: { before: string; id: string }): Promise<HistoryPage> {
  const params = new URLSearchParams({ limit: "20" });
  if (cursor) { params.set("before", cursor.before); params.set("before_id", cursor.id); }
  const value = await json(`/inspections?${params}`, { signal }) as HistoryPage;
  if (!Array.isArray(value.items) || !value.items.every((item) => UUID_PATTERN.test(item.id)
      && Number.isFinite(Date.parse(item.created_at)))
    || (value.next_before_id !== null && !UUID_PATTERN.test(value.next_before_id))) {
    throw new ApiError("Invalid inspection history response.");
  }
  return value;
}

export async function predict(file: File, signal: AbortSignal): Promise<Inspection> {
  const form = new FormData();
  form.append("file", file); // Let the browser set the multipart boundary.
  return checkInspection(await json("/inspections", { method: "POST", body: form, signal }));
}

export async function getInspection(id: string, signal: AbortSignal): Promise<Inspection> {
  if (!UUID_PATTERN.test(id)) throw new ApiError("Enter a complete inspection UUID.");
  return checkInspection(await json(`/inspections/${encodeURIComponent(id)}`, { signal }));
}

export async function getModel(signal: AbortSignal): Promise<ModelMetadata> {
  return checkModel(await json("/models/current", { signal }) as ModelMetadata);
}

export async function getReadiness(signal: AbortSignal): Promise<Readiness> {
  const response = await fetch(`${API}/ready`, { signal, cache: "no-store" });
  if (response.status !== 503) await checked(response);
  const body = await response.json() as Readiness;
  if (!["ready", "not_ready"].includes(body.status)
    || typeof body.model !== "boolean" || typeof body.database !== "boolean") {
    throw new ApiError("The readiness response could not be read.");
  }
  return body;
}

/** Task 05 returns backend-root paths; keep artifacts on the same proxy origin. */
export function artifactUrl(inspection: Inspection, kind: "image" | "mask"): string {
  const url = new URL(inspection[kind].url, `${window.location.origin}/`);
  const expected = `/inspections/${inspection.id}/artifacts/${kind}`;
  if (url.pathname !== expected || url.search || url.hash) {
    throw new ApiError("The backend returned an unexpected artifact reference.");
  }
  return `${API}${expected}`;
}

export async function artifactBitmap(url: string, signal: AbortSignal): Promise<ImageBitmap> {
  if (typeof createImageBitmap !== "function") throw new ApiError("This browser does not support native mask decoding. Use a current browser.");
  const response = await checked(await fetch(url, { signal, cache: "no-store" }));
  return createImageBitmap(await response.blob(), { colorSpaceConversion: "none", premultiplyAlpha: "none" });
}
