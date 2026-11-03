export const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;

/** A lightweight content check; decoding, modes and dimensions remain server-validated. */
export async function validateFile(file: File): Promise<File> {
  if (!file.size) throw new Error("Choose a nonempty image.");
  if (file.size > MAX_UPLOAD_BYTES) throw new Error("Choose an image no larger than 10 MiB.");
  const bytes = new Uint8Array(await file.slice(0, 12).arrayBuffer());
  const png = [137, 80, 78, 71, 13, 10, 26, 10].every((v, i) => bytes[i] === v);
  const jpeg = bytes[0] === 255 && bytes[1] === 216 && bytes[2] === 255;
  if (!png && !jpeg) throw new Error("The file content must be JPEG or PNG.");
  const mime = png ? "image/png" : "image/jpeg";
  if (file.type && file.type !== "application/octet-stream" && file.type !== mime) {
    throw new Error("The image content does not match its declared file type.");
  }
  return file.type === mime ? file : new File([file], file.name, { type: mime, lastModified: file.lastModified });
}

// Server defaults (api/config.py UploadLimits); decode_upload refuses anything larger.
const MAX_PIXELS = 4_000_000;
const MAX_DIMENSION = 8192;

/** The first image on the system clipboard, read directly (needs a click or key press). */
export async function readClipboardImage(): Promise<File | null> {
  if (!navigator.clipboard?.read) throw new Error("This browser cannot read images from the clipboard. Press ⌘ V or Ctrl V instead.");
  for (const item of await navigator.clipboard.read()) {
    const type = item.types.find((entry) => entry.startsWith("image/"));
    if (type) return new File([await item.getType(type)], "pasted-image", { type });
  }
  return null;
}

/**
 * Copied images are usually transparent PNG screenshots, often larger than the upload limit;
 * the server refuses both. Flatten anything the browser can decode onto white and scale it
 * into the limits as an RGB JPEG, then run the normal content check.
 */
export async function prepareClipboardImage(file: File): Promise<File> {
  const unreadable = "This browser cannot read the copied image. Copy it as JPEG or PNG.";
  const bitmap = await createImageBitmap(file).catch(() => { throw new Error(unreadable); });
  const { width, height } = bitmap;
  const scale = Math.min(1, Math.sqrt(MAX_PIXELS / (width * height)), MAX_DIMENSION / Math.max(width, height));
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.floor(width * scale));
  canvas.height = Math.max(1, Math.floor(height * scale));
  const context = canvas.getContext("2d");
  if (!context) { bitmap.close(); throw new Error(unreadable); }
  context.fillStyle = "#fff";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  bitmap.close();
  const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.95));
  canvas.width = canvas.height = 0;
  if (!blob) throw new Error(unreadable);
  const base = file.name && file.name !== "image.png" ? file.name.replace(/\.[^.]*$/, "") : "pasted-image";
  return validateFile(new File([blob], `${base}.jpg`, { type: "image/jpeg" }));
}
