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
