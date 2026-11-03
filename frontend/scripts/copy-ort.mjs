// Copies the ONNX Runtime WebAssembly binary into public/ort-1220 so it is served same-origin.
import { cpSync, mkdirSync } from "node:fs";
mkdirSync("public/ort-1220", { recursive: true });
cpSync("node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm", "public/ort-1220/ort-wasm-simd-threaded.wasm");
cpSync("node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs", "public/ort-1220/ort-wasm-simd-threaded.mjs");
console.log("copied ort runtime");
