# Task 12: integrated application

Image upload → native LR-ASPP segmentation/Task 09 features → crack ResNet18 → independent
Φ-Net HTTP service → PostgreSQL → React overlays, results and inspection history.
No retraining, automatic model selection or severity training on RC1841 features occurs.

## Checkpoints and interpretation

`config/task12.toml` explicitly SHA-pins these external files:

| Role | Checkpoint | Epoch |
|---|---|---:|
| Segmentation | `outputs/task09/experiments/lraspp_matched_s42/best.pt` | 21 |
| Crack, confirmed by the user | `outputs/task07/experiments/sdnet_resnet_pilot/best.pt` | 5 |
| Component damage level | `outputs/task10/experiments/phi7_resnet18_s42/best.pt` | 16 |

Crack uses the saved validation threshold **0.40030238032341003** and original IDs
`0=no_crack, 1=crack`. Φ-Net retains **0=Heavy, 1=Minor, 2=Moderate, 3=Undamaged**;
numeric IDs are not ordinal distances. Segmentation IDs/colors remain 0–8 from `labels.py`.
The newer Task 08 classifier is not silently substituted. No Task 11 selection package
or Docker configuration existed in this checkout; explicit paths and the user's choice
resolve the missing integration metadata without inventing a Task 11 selection result.

The backend loads one segmentation and one crack model per worker; the independent
service loads Φ-Net once. Startup checks each file's hash before/after loading. All
reloads use `pretrained=False`; serving never downloads model weights or reads datasets.
Checkpoint-embedded dataset identities describe provenance, not a fresh release audit.

Uploads are verified JPEG/PNG, size-bounded, single-frame RGB/grayscale images. EXIF
orientation is validated/applied once; all models receive sanitized RGB. Segmentation
reuses Task 09 letterbox/normalization and argmax → unpad → nearest-native restoration.
Images, masks, boxes and geometry share the oriented original coordinates. The dashboard
compatibility adapter translates Task 09 measurements without recomputing components.
The complete validated report remains in `features`. Coverage uses raw pixels; no mask
pixels are removed and reported regions are capped at 1,000, with truncation disclosed.

Both classifiers retain the original full-image letterbox and ImageNet normalization.
An uploaded RGB photograph is not a stored Φ-Net tensor: no Caffe inverse is applied.
The checkpoint's inferred training color semantics and unknown source identities remain
provisional limitations. SDNET patch training also does not establish whole-photo validity.
There is no verified material/domain gate or OOD rejection. Models can disagree; no
unsupported fusion rule resolves them. Confidence is uncalibrated. Φ-Net labels concern
visible component/image damage, not whole-building severity, capacity or safety. Pixel
coverage is not physical surface area. No RC1841 labels are treated as paired Φ-Net targets.

## Services, HTTP and storage

| Backend endpoint (prefix `/api` in frontend) | Behavior |
|---|---|
| `POST /inspections` | One multipart `file`; infer and persist |
| `GET /inspections` | History, `limit=1..100`, paired `before`/`before_id` cursor |
| `GET /inspections/{uuid}` | Saved results without inference |
| `GET /inspections/{uuid}/artifacts/{image\|mask}` | Hash-verified PNG / native class-ID mask |
| `GET /models/current` | Resident segmentation metadata |
| `GET /health`, `/ready` | Liveness / model, DB migration and severity readiness |

The independent service exposes `POST /predict`, `/models/current`, `/health`, `/ready`.
The backend sends sanitized PNG pixels over HTTP and validates labels, probabilities,
the model SHA, dimensions and RGB pixel hash. Its pooled client closes at shutdown.
Model/decoding/HTTP/SQL work stays off the async event loop; one admitted upload per
worker prevents simultaneous model use while health/retrieval remain available.

Severity failures/timeouts/bad responses produce `severity.status="unavailable"` with
no guessed label/confidence. Valid segmentation/crack results may still be saved with
HTTP 201. `status="completed"` means processed/persisted, not that every model predicted;
`/ready` returns 503 during dependency outages. Segmentation/crack or DB failures report
errors. On ambiguous SQL commit failure, files remain for reconciliation by UUID rather
than risking deletion of a committed inspection's artifacts. Uploads are not auto-retried.

Migration 002 adds nullable features/classification/severity JSONB and a history index.
Version-1 records remain retrievable with null new fields. Connections use UTC so create
and retrieve JSON timestamps match on non-UTC database hosts. Binary artifacts remain
outside PostgreSQL. History uses `(created_at, UUID)` pagination. App startup never migrates.

```text
config/task12.toml                   explicit external model pins
compose.yaml                        four services + explicit migration job
deploy/{python,frontend}.Dockerfile  Python serving / Vite+Nginx images
deploy/nginx.conf, constraints.txt   same-origin proxy / direct Python runtime pins
src/damage_analytics/serving/
  contracts.py                      validated independent prediction schemas
  classifiers.py, segmentation.py   reused ML pipelines and dashboard adapter
  severity_app.py, severity_client.py separate service and bounded HTTP gateway
  configure.py, migrate.py           private environment generation / explicit migrations
  verify.py                         real upload/persistence/native-artifact check
migrations/002_integrated_predictions.sql
frontend/src/components/{ClassifierResults,HistoryPanel}.tsx
tests/test_task12.py, tests/run_task12_live.py
```

Existing API settings/routes/service/repository/schemas and React results were extended.
The Canvas/palette and Tasks 01–10 ML code/checkpoints remain intact. Compose mounts only
selected checkpoint files, read-only; datasets are not mounted. PostgreSQL and artifact
volumes persist across container restarts. Only `127.0.0.1:8080` is published. Compose's
explicit migration job waits for PostgreSQL and applies missing versions under a lock.
Nginx and Vite both proxy `/api` without changing backend CORS. Credentials are not in JS.

New backend variables: `DAMAGE_API_CRACK_CHECKPOINT`, `DAMAGE_API_CRACK_CHECKPOINT_SHA256`,
`DAMAGE_API_SEVERITY_URL`, `DAMAGE_API_SEVERITY_CHECKPOINT_SHA256`, optional
`DAMAGE_API_SEVERITY_TIMEOUT_SECONDS` (60). Existing database/segmentation/storage/device
and upload-limit variables still apply. Severity uses `SEVERITY_CHECKPOINT`,
`SEVERITY_CHECKPOINT_SHA256`, optional `SEVERITY_DEVICE` and `SEVERITY_CPU_THREADS`.
Native macOS supports CPU/MPS settings; **Linux Docker uses CPU**. MPS was not tested.
The internal severity upload budget is 16 MiB to accommodate lossless PNG expansion of
a valid 4 MP JPEG; the external API keeps its 10 MiB upload cap. When customizing limits,
align `SEVERITY_MAX_PIXELS`/`SEVERITY_MAX_DIMENSION` with the API and allow PNG expansion
in `SEVERITY_MAX_UPLOAD_BYTES`. `SEVERITY_UPLOAD_TIMEOUT_SECONDS` is also configurable.

## Exact Mac commands

Run from the repository root, with Docker Desktop installed/running. Existing local
packages were sufficient; none were installed. A fresh environment can use the existing
extras: `python -m pip install -e '.[backend,classification,dev]'` (run manually, outside
any ongoing training environment). Docker builds install runtime dependencies inside
images; they do not download pretrained model weights. Keep `.env.task12` with its DB
volume: the generator uses a random password, mode 0600 and refuses overwriting credentials.

```bash
source .venv/bin/activate
python -m damage_analytics.serving.configure
docker compose --env-file .env.task12 config --quiet
docker compose --env-file .env.task12 up --build -d
docker compose --env-file .env.task12 ps --all
docker compose --env-file .env.task12 logs --tail=50 migrate backend severity
curl --fail --retry 30 --retry-delay 2 --retry-connrefused http://127.0.0.1:8080/api/ready

# Existing frozen-validation image, not final test. Use a fresh report filename each run.
python -m damage_analytics.serving.verify \
  --base-url http://127.0.0.1:8080/api \
  --image outputs/task02/versions/rc1841-curated-v1-d34771dbb662406a/images/000016.png \
  --output outputs/task12/verification/docker_upload_001.json
python -m json.tool outputs/task12/verification/docker_upload_001.json
open http://127.0.0.1:8080
```

In React, upload a JPEG/PNG, adjust the overlay opacity, select a region, inspect separate
classification results, then load an inspection from history. The verification CLI
checks real image upload, both classifier contracts, persisted retrieval/history,
artifact hashes/native alignment, exact mask-area counts and malformed-input rejection.
It requires available severity; it does not pass partial results as complete integration.

```bash
# Verify persistence after a backend restart.
export INSPECTION_ID="$(python -c 'import json; print(json.load(open("outputs/task12/verification/docker_upload_001.json"))["inspection_id"])')"
docker compose --env-file .env.task12 restart backend
curl --fail --retry 30 --retry-delay 2 --retry-connrefused "http://127.0.0.1:8080/api/inspections/$INSPECTION_ID"
docker compose --env-file .env.task12 exec -T postgres psql -U damage -d damage \
  -c 'SELECT id, created_at, width, height FROM inspections ORDER BY created_at DESC LIMIT 5;'

# Check dependency failure explicitly; readiness is expected to return 503.
docker compose --env-file .env.task12 stop severity
curl -i http://127.0.0.1:8080/api/ready
curl --fail -F file=@outputs/task02/versions/rc1841-curated-v1-d34771dbb662406a/images/000016.png \
  http://127.0.0.1:8080/api/inspections -o outputs/task12/verification/docker_outage_001.json
python -c 'import json; print(json.load(open("outputs/task12/verification/docker_outage_001.json"))["severity"])'
docker compose --env-file .env.task12 start severity
curl --fail --retry 30 --retry-delay 2 --retry-connrefused http://127.0.0.1:8080/api/ready

# Preserve database/artifact volumes when stopping.
docker compose --env-file .env.task12 down
```

For an existing native Task 05 database, set its existing `DAMAGE_API_DATABASE_URL` and
explicitly run `python -m damage_analytics.serving.migrate` before using this backend.
No existing user DB was migrated here; verification used a new disposable Task 12 cluster.

## Tests and actual execution

```bash
python -m pytest tests/test_task12.py tests/test_api_core.py tests/test_api_http.py -q
python -m ruff check src/damage_analytics/api src/damage_analytics/serving tests/test_task12.py tests/run_task12_live.py
npm --prefix frontend run build
```

Live PostgreSQL tests require `DAMAGE_TASK12_TEST_DATABASE_URL` pointing to a dedicated
disposable DB. The optional native check starts two temporary loopback Uvicorn processes,
uses real checkpoints and stops them afterward; it additionally requires a DB name
starting `task12_test` and free ports 18850/18851:

```bash
# Set DAMAGE_TASK12_TEST_DATABASE_URL to your disposable database first.
python tests/run_task12_live.py \
  --image outputs/task02/versions/rc1841-curated-v1-d34771dbb662406a/images/000016.png \
  --output outputs/task12/verification/native_manual_001
```

Executed: **38 tests passed**, including real PostgreSQL JSONB round-trip, UTC JSON
equality, migration idempotence and preservation of version-1 rows. Both HTTP contracts,
wrong model/input rejection, unavailability, single startup loads, class semantics,
pagination, EXIF orientation, 4 MP JPEG→PNG transport expansion and native geometry are
covered. Ruff, TypeScript and the
React production build passed. Installed Starlette emits one HTTPX test-client deprecation
warning; dependencies were not replaced. No full prior ML suite was run.

The real native check used a 666×1000 validation image, all three pinned checkpoints,
separate HTTP processes and actual PostgreSQL. Upload, severity input/model binding,
saved retrieval/history, artifacts/coverage, invalid input and persisted partial results
after stopping severity passed. See `outputs/task12/verification/native_002/native_result.json`
and `outputs/task12/verification/integrated_tests.xml`. A prior failed native check exposed
the timezone serialization defect; the corrected run passed. No training, checkpoint
modification, dataset download or final-test performance inspection occurred.

**Not executed:** Docker builds/Compose runtime/Nginx, a browser interaction test, MPS
serving, load tests or new model-quality evaluation. Docker is unavailable here. YAML
syntax was independently parsed; this does not establish Docker runtime correctness.
Run the Mac launch/verify commands to close those deployment checks. Temporary native
API processes and the disposable PostgreSQL cluster were stopped after verification.
