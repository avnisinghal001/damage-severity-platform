export class ApiError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ApiError";
  }
}

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "The request could not be completed.";
}

/** Results live in memory as blob: URLs created by the analysis step. */
export function artifactUrl(inspection: { image: { url: string }; mask: { url: string } }, kind: "image" | "mask"): string {
  return inspection[kind].url;
}

export async function artifactBitmap(url: string, signal: AbortSignal): Promise<ImageBitmap> {
  if (typeof createImageBitmap !== "function") throw new ApiError("This browser does not support native mask decoding. Use a current browser.");
  const response = await fetch(url, { signal });
  if (!response.ok) throw new ApiError("The image could not be read.");
  return createImageBitmap(await response.blob(), { colorSpaceConversion: "none", premultiplyAlpha: "none" });
}
