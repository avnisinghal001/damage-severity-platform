# Concrete frontend

The current app runs three trained ONNX models locally with React, TypeScript, Vite and
ONNX Runtime WebAssembly: RC1841 damage segmentation, SDNET2018 crack classification,
and PEER Φ-Net Task 7 visible damage-level classification.

See the [project README](../README.md) for the complete model/dataset inventory, exact
training settings, recorded validation results, preprocessing, display policies and
scientific limitations.

## Run

Requires Node.js 20.19+ and a browser supporting WebAssembly and Canvas. From this folder:

```bash
npm ci
npm run dev
```

Open http://127.0.0.1:5173. Wait for the three model loaders, then choose, drop, paste
or select a sample image. The active inference path does not call a backend or save
inspection history. Image/mask results live in browser memory.

```bash
npm run typecheck
npm run build
npm run preview
```

Preview: http://127.0.0.1:4173. Production output: `dist/`. Both dev and build copy the
installed runtime into `public/ort-1220/`; keep the committed package lockfile.
Deploy the entire build at the site root so `/models/`, `/ort-1220/` and `/samples/`
remain available. For Vercel, set the project root to `frontend`; `vercel.json` supplies
the build/output settings. No prediction API or database environment variables are
required for this version.

The result card currently sets the coverage headline to zero and hides highlights
when Φ-Net predicts Undamaged, even though the raw segmentation mask is still
generated. Its “surface area” caption refers to image pixels, not physical area.
Scores are uncalibrated; visible component labels do not assess structural safety.
