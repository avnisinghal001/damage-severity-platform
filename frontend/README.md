# Task 06 dashboard

React + TypeScript + Vite, without UI, charting, routing or state-management libraries.
Tasks 01–05 are unchanged. Start the Task 05 backend separately using the root README.

Run manually from the repository root (Node.js 20.19+):

```bash
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`. Vite forwards `/api/*` to `http://127.0.0.1:8000/*`,
including the image and mask artifact URLs. Backend CORS changes are unnecessary.
For another backend address, set `BACKEND_URL` before running Vite, for example:

```bash
BACKEND_URL=http://127.0.0.1:8001 npm run dev
npm run typecheck
npm run build
npm run preview
```

Preview runs on port 4173 with the same proxy. A production static host needs an
equivalent same-origin `/api` reverse proxy; the Vite proxy is not part of `dist/`.
No API URL or database credentials need to be embedded in the browser bundle.

Upload one JPEG/PNG or paste an inspection UUID. The client checks file signatures,
MIME consistency and a 10 MiB limit; FastAPI remains authoritative for decoding,
color modes, dimensions and additional configured limits. Submissions are protected
against duplicates and are never automatically retried. A timed-out POST may still
be saved by the backend. Local previews are distinct from the server-oriented images.

The viewer decodes native class-ID PNG bytes, validates IDs 0–8 and matching image/mask
dimensions, and uses the exact `damage_analytics/labels.py` palette. Original, overlay,
class-mask and 1:1-pixel views share native coordinates. Background is transparent in
overlay view and gray in mask view. Downloads retain the original ID PNG. Visualization
is limited to 16 megapixels to bound browser allocations; incompatible data fails visibly.

`src/api.ts` owns typed requests, response checks and safe artifact proxy resolution;
`mask.ts` handles native mask decoding; `palette.ts` mirrors the Python taxonomy;
`components/` separates inputs, backend status, visualization and paginated analytics.
Region selection highlights an exclusive-maximum bounding box. Confidence remains
uncalibrated, areas refer to image pixels, and no structural safety score is displayed.
There is no fabricated inspection listing/history API.

Dependencies were not installed and no server was launched during implementation.
Type checking, build and browser verification are left for the commands above.
