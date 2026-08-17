# Miner Quickstart

This project includes three miner services:
- `upscaling-video2x`: Upscaling service that wraps Video2X.
- `upscaling-ffmpeg`: Upscaling service using FFmpeg directly.
- `compression`: Compression microservice.

## Prerequisites
- Run `sudo apt update`
- Docker + Docker Compose
- NVIDIA drivers and `nvidia-container-toolkit`
- Storage environment variables (S3-compatible) in a new `miner/.env` file derived from `miner/.env.template`
- A shared local work directory. The default is `/tmp/vidaio-miner-video-tmp` via `MINER_SHARED_DIR`.

The service APIs are published on localhost-only Docker ports. `neurons/miner.py` downloads validator payloads, places local files in `MINER_SHARED_DIR`, forwards local paths to the containers, uploads processed outputs, and returns presigned URLs. Remote URL input is disabled inside the service containers by default.

```bash
mkdir -p /tmp/vidaio-miner-video-tmp
chmod 777 /tmp/vidaio-miner-video-tmp
```

## Queueing
Each processing container runs a bounded in-process queue. By default, a service accepts 2 active jobs and 5 waiting jobs; additional requests receive HTTP 429 backpressure.

Tune the limits in `miner/.env`:
```bash
MAX_CONCURRENT_UPSCALING=2
MAX_QUEUE_SIZE_UPSCALING=5
MAX_CONCURRENT_COMPRESSION=2
MAX_QUEUE_SIZE_COMPRESSION=5
```

Queue status is available from `/queue` on each service.

For Modal deployments, the worker functions set `MAX_QUEUE_SIZE_*` to `0`, `MAX_CONCURRENT_*` to `MODAL_SERVICE_MAX_CONCURRENT` (default 5), and `@modal.concurrent(max_inputs=MODAL_GPU_WORKER_MAX_INPUTS)` (default 5). Modal owns the pending-input queue and scales out containers instead of persisting work in the Python process.

Compression adds a duration-aware Modal router. Short compression jobs route to `compress_l4_l40s` on `MODAL_SHORT_COMPRESSION_GPU`, which defaults to `L4,L40S` so Modal tries L4 first and falls back to L40S. Jobs with unknown duration, or duration greater than `MODAL_LONG_COMPRESSION_THRESHOLD_SECONDS` (default 1200 seconds), route to `compress_rtx_pro_6000` on RTX PRO 6000 and use chunked compression.

## Temporary File Cleanup
`MINER_SHARED_DIR` on the host and `SHARED_VOLUME_PATH` inside each container are temporary work areas. They should not accumulate files during normal operation:

- `neurons/miner.py` downloads validator inputs into `MINER_SHARED_DIR`, forwards the mounted container path to the processing service, uploads the processed output, then removes both the input and output.
- Local-only Docker services keep successful outputs just long enough for `neurons/miner.py` to upload them, then the miner removes them from the host shared directory.
- Remote-mode services download inputs into `SHARED_VOLUME_PATH`, upload processed outputs directly to S3-compatible storage, then remove their local input and output files in a `finally` cleanup path.
- Active inputs and outputs are tracked so the cleanup worker does not delete a file that is currently being processed or uploaded.

Crash and interruption paths are handled by low-frequency cleanup workers in the miner and service containers. By default, untracked local files expire after:

```text
min(MINER_STORAGE_S3_PRESIGNED_EXPIRY, 604800) + PRESIGNED_URL_CLEANUP_GRACE_SECONDS
```

With the default 1 hour presigned URL expiry and 600 second grace period, local temp files are eligible for cleanup after 4200 seconds. The same worker also keeps the shared directory under a 9 GB soft cap by deleting the oldest untracked files older than `MINER_CLEANUP_MIN_FILE_AGE_SECONDS`. This is intended for small persistent volumes, including Modal-style 10 GB storage.

Processed outputs uploaded to the configured S3-compatible bucket are also temporary. The cleanup worker scans only `MINER_STORAGE_CLEANUP_PREFIXES` and deletes objects older than `MINER_STORAGE_OBJECT_TTL_SECONDS`. When `MINER_STORAGE_OBJECT_TTL_SECONDS` is blank, it uses the same presigned URL expiry plus `PRESIGNED_URL_CLEANUP_GRACE_SECONDS`, so objects are retained through URL validity and then removed after the grace window. The default prefixes are `processing/` and `upscaling/`, which cover miner-owned outputs and avoid broad bucket cleanup.

Tune cleanup in `miner/.env`:
```bash
MINER_CLEANUP_ENABLED=true
MINER_CLEANUP_INTERVAL_SECONDS=300
MINER_CLEANUP_MAX_VOLUME_BYTES=9000000000
MINER_CLEANUP_MIN_FILE_AGE_SECONDS=60
MINER_TEMP_FILE_TTL_SECONDS=
PRESIGNED_URL_CLEANUP_GRACE_SECONDS=600
MINER_STORAGE_CLEANUP_ENABLED=true
MINER_STORAGE_CLEANUP_PREFIXES=processing/,upscaling/
MINER_STORAGE_OBJECT_TTL_SECONDS=
```

Set `MINER_STORAGE_OBJECT_TTL_SECONDS` only if you want bucket objects to live longer or shorter than the presigned URL expiry plus grace. Set `MINER_STORAGE_CLEANUP_ENABLED=false` when another lifecycle policy owns deletion for these prefixes.

On Modal, GPU worker cleanup loops are disabled. The `cleanup_expired_artifacts` scheduled function in `miner/modal_workers.py` runs on CPU, deletes expired S3-compatible objects under the configured prefixes, and removes stale Modal volume files older than the same TTL. This keeps cleanup from competing with upscaling/compression GPU containers.

Set `MODAL_VOLUME_FILE_TTL_SECONDS` only if Modal volume files should use a different TTL from bucket objects.

## Long Video Compression
Long-video compression can split the source at keyframes near the target chunk duration, compress chunks in parallel, then merge the encoded chunks with the ffmpeg concat demuxer and `-c copy`.

Defaults:

```bash
COMPRESSION_CHUNKING_ENABLED=true
COMPRESSION_CHUNK_MIN_DURATION_SECONDS=1200
COMPRESSION_CHUNK_TARGET_SECONDS=600
COMPRESSION_CHUNK_PARALLELISM=2
MODAL_LONG_COMPRESSION_THRESHOLD_SECONDS=1200
MODAL_LONG_COMPRESSION_CHUNK_SECONDS=600
MODAL_LONG_COMPRESSION_CHUNK_PARALLELISM=4
MODAL_SHORT_COMPRESSION_GPU=L4,L40S
MODAL_LONG_COMPRESSION_GPU=RTX-PRO-6000
MODAL_SERVICE_MAX_CONCURRENT=5
MODAL_GPU_WORKER_MAX_INPUTS=5
MODAL_COMPRESSION_TIMEOUT_SECONDS=14400
```

For a 3 hour video, the long Modal worker targets roughly 18 ten-minute chunks and compresses several chunks at once inside the RTX PRO 6000 container. Additional API requests can share a warm Modal GPU container up to `MODAL_GPU_WORKER_MAX_INPUTS`, then Modal can still scale out to additional containers.

Caveats:

- `MODAL_SHORT_COMPRESSION_GPU` accepts a comma-separated Modal GPU fallback list. `L4,L40S` keeps short AV1 jobs on the cheapest suitable AV1-capable GPU first, while allowing L40S when L4 capacity is unavailable.
- L4/L40S provide AV1-capable NVENC for short compression jobs. Keep `MODAL_SHORT_COMPRESSION_GPU` on AV1-capable GPU types when AV1 requests are expected.
- Source splitting with `-c copy -f segment` cuts on existing keyframes, so chunks are close to the target duration but not exact. Very long source GOPs produce uneven chunks; sources with too few keyframes can fall back to single-pass compression.
- The merge step uses stream copy. This is safe only when every encoded segment has compatible codec, resolution, pixel format, profile, time base, and audio layout. The worker enforces one encoder configuration across all chunks; changing per-chunk settings would break this.
- NVENC/NVDEC jobs usually do not occupy much VRAM. RTX PRO 6000 is useful here for encoder/decoder throughput and parallel segment work, not because ffmpeg should consume 96 GB of VRAM.
- If chunked compression cannot produce a useful split, the service falls back to single-pass compression. If an individual chunk encode or concat merge fails, the request fails so the miner does not return a partial artifact.

## Video2X Upscaling Miner
1. Complete host/container preparation in `miner/upscaling/SETUP.md`.
2. Start service:
   ```bash
   docker compose --profile upscaling-video2x up -d upscaling-video2x
   ```
3. Point the miner at the Video2X profile port:
   ```bash
   export MINER_UPSCALING_SERVICE_URL=http://localhost:8003
   ```

## FFmpeg Upscaling Miner
1. (Optional) Review build/runtime notes in `miner/upscaling/ffmpeg/SETUP.md`.
2. Start service:
   ```bash
   docker compose --profile upscaling-ffmpeg up -d upscaling-ffmpeg
   ```
3. Point the miner at the FFmpeg profile port:
   ```bash
   export MINER_UPSCALING_SERVICE_URL=http://localhost:8005
   ```

## Compression Miner
```bash
docker compose up -d compression
```

## Miner Forwarding
`neurons/miner.py` forwards directly to the container services:
- `MINER_UPSCALING_SERVICE_URL` defaults to `http://localhost:8003`.
- `MINER_COMPRESSION_SERVICE_URL` defaults to `http://localhost:8004`.
- `MINER_SHARED_VOLUME_PATH` defaults to `MINER_SHARED_DIR`, then `/tmp/vidaio-miner-video-tmp`.

## Modal Serverless Workers
`miner/modal_workers.py` deploys these Modal entrypoints:

- `upscale_video2x`: Video2X upscaling worker.
- `upscale_ffmpeg`: FFmpeg upscaling worker.
- `compress`: CPU compression router.
- `compress_l4_l40s`: short-video compression worker using L4 with L40S fallback by default.
- `compress_rtx_pro_6000`: long-video chunked compression worker.

GPU workers use `cpu=16.0`, `min_containers=0`, `max_containers=5` by default, `scaledown_window=300`, and up to 5 concurrent inputs per container. Upscaling and long compression use RTX PRO 6000 by default; short compression uses `MODAL_SHORT_COMPRESSION_GPU` (`L4,L40S` by default). A burst of 3 to 5 requests can be handled inside a warm worker; larger bursts are handled by Modal scale-out. After the idle window, the GPU functions scale back to zero.

Compression is duration-routed: `compress` probes the URL duration, `compress_l4_l40s` uses `gpu=["L4", "L40S"]` by default for short videos, and `compress_rtx_pro_6000` uses `gpu="RTX-PRO-6000"` plus parallel chunk encoding for long videos.

Modal service request bodies use `video_paths` with 1 to 5 input URLs. Synthetic miner payloads can carry multiple `reference_video_urls`; organic jobs continue to use a single URL.

Modal's Logs table is keyed by app/run name, so deployed worker rows stay under `vidaio-miner-workers`. Each worker prints structured JSON log events with `task_id`, `worker`, and `modal_input_id`; search the log tab for the task ID to isolate a request.

Install the Modal SDK in the miner environment:

```bash
pip install -r requirements.txt
```

Authenticate Modal with `modal setup`. The SDK reads `~/.modal.toml` from `modal token new`, or these environment variables:

```bash
export MODAL_TOKEN_ID=your-modal-token-id
export MODAL_TOKEN_SECRET=your-modal-token-secret
```

Create the Modal secret used by all workers:

```bash
modal secret create vidaio-miner-secrets \
  MINER_STORAGE_PROVIDER=backblaze \
  MINER_STORAGE_S3_ACCESS_KEY_ID=your-access-key \
  MINER_STORAGE_S3_SECRET_ACCESS_KEY=your-secret-key \
  MINER_STORAGE_S3_REGION=us-east-005 \
  MINER_STORAGE_S3_BUCKET_NAME=your-bucket-name \
  MINER_STORAGE_S3_ENDPOINT_URL=https://s3.us-east-005.backblazeb2.com \
  MINER_STORAGE_S3_PRESIGNED_EXPIRY=3600 \
  MINER_STORAGE_CLEANUP_PREFIXES=processing/,upscaling/ \
  PRESIGNED_URL_CLEANUP_GRACE_SECONDS=600
```

Deploy the Modal app from the repository root:

```bash
cd miner
modal deploy modal_workers.py
```

`modal_workers.py` loads `miner/.env` during deployment when `python-dotenv` is installed, so the Modal GPU, timeout, and chunking knobs above can live there.

Configure `miner/.env` for Modal processing:

```bash
MINER_PROCESSING_BACKEND=modal
MODAL_APP_NAME=vidaio-miner-workers
MODAL_MINER_SECRET_NAME=vidaio-miner-secrets
MINER_MODAL_UPSCALING_FUNCTION=upscale_video2x
MINER_MODAL_COMPRESSION_FUNCTION=compress
MODAL_SHORT_COMPRESSION_GPU=L4,L40S
MODAL_LONG_COMPRESSION_GPU=RTX-PRO-6000
MODAL_SERVICE_MAX_CONCURRENT=5
MODAL_GPU_WORKER_MAX_INPUTS=5
MODAL_LONG_COMPRESSION_THRESHOLD_SECONDS=1200
MODAL_LONG_COMPRESSION_CHUNK_SECONDS=600
MODAL_LONG_COMPRESSION_CHUNK_PARALLELISM=4
MODAL_COMPRESSION_TIMEOUT_SECONDS=14400
MODAL_TOKEN_ID=your-modal-token-id
MODAL_TOKEN_SECRET=your-modal-token-secret
```

To use the FFmpeg upscaler instead of Video2X:

```bash
MINER_MODAL_UPSCALING_FUNCTION=upscale_ffmpeg
```

Run the miner with the same wallet/subtensor arguments you use today:

```bash
python3 neurons/miner.py \
  --wallet.name [Your_Wallet_Name] \
  --wallet.hotkey [Your_Hotkey_Name] \
  --subtensor.network finney \
  --netuid 85 \
  --axon.port [port] \
  --logging.debug
```

To keep the miner running with PM2, launch it from the repository root with the repository on `PYTHONPATH`:

```bash
PYTHONPATH=. pm2 start neurons/miner.py --name video-miner --interpreter python3 -- \
  --wallet.name [Your_Wallet_Name] \
  --wallet.hotkey [Your_Hotkey_Name] \
  --subtensor.network finney \
  --netuid 85 \
  --axon.port [port] \
  --logging.debug
```

For direct SDK tests, run:

```bash
cd miner
modal run modal_workers.py --worker video2x --video-url "https://example.com/input.mp4" --scale 2
modal run modal_workers.py --worker ffmpeg --video-url "https://example.com/input.mp4" --scale 2
modal run modal_workers.py --worker compression --video-url "https://example.com/input.mp4" --codec AV1 --codec-mode CRF --cq 35
modal run modal_workers.py --worker compression-l4-l40s --video-url "https://example.com/input.mp4" --codec AV1 --codec-mode CRF --cq 35
modal run modal_workers.py --worker compression-rtx --video-url "https://example.com/input.mp4" --codec AV1 --codec-mode CRF --cq 35
```

For direct HTTP endpoint tests, use the URLs printed by `modal deploy` for `upscaling_video2x_api`, `upscaling_ffmpeg_api`, `compression_api`, and `compression_l4_l40s_api`:

```bash
curl -X POST "$MODAL_VIDEO2X_URL/upscale" \
  -H "Content-Type: application/json" \
  -d '{"video_paths":["https://example.com/input.mp4"],"scale":2,"task_id":"manual-video2x"}'

curl -X POST "$MODAL_FFMPEG_URL/upscale" \
  -H "Content-Type: application/json" \
  -d '{"video_paths":["https://example.com/input.mp4"],"scale":2,"task_id":"manual-ffmpeg"}'

curl -X POST "$MODAL_COMPRESSION_URL/compress" \
  -H "Content-Type: application/json" \
  -d '{"video_paths":["https://example.com/input.mp4"],"task_id":"manual-compress","codec":"AV1","codec_mode":"CRF","cq":35}'

curl -X POST "$MODAL_COMPRESSION_L4_L40S_URL/compress" \
  -H "Content-Type: application/json" \
  -d '{"video_paths":["https://example.com/input.mp4"],"task_id":"manual-compress-short","codec":"AV1","codec_mode":"CRF","cq":35,"chunked":false}'
```
## Competition mode (Phase 1 contract)

The miner can expose inference handlers, competition handlers, or both without
sharing their state. Set `MINER_MODES=inference,competition` for the recommended
dual-mode configuration. The initial competition type is only `COMPRESSION`.

Competition submission uses `MINER_COMPETITION_REPOSITORY_URL` and reads a raw
PAT from the environment variable named by `MINER_COMPETITION_GITHUB_PAT_ENV`.
Use a fine-grained, repository-scoped, contents-read-only PAT with an expiry.
The PAT is returned over the Bittensor submission protocol by explicit product
decision; it must never appear in logs, URLs, Git configuration, Redis, SQLite,
or W&B.

Each submission poll also carries the validator's persisted result for the
previous revision: `ACCEPTED`, `REJECTED`, `REVIEW_REQUIRED`, or `NOT_RECEIVED`,
plus its reason code, safe detail, pinned commit, and revision number. The miner
logs this feedback before responding. Until the contender finalisation deadline,
keep the configured repository updated (or change
`MINER_COMPETITION_REPOSITORY_URL`); the next poll will clone and validate its
current HEAD. A successful replacement atomically removes the older validator
artifact. After finalisation, the last accepted/pinned revision is immutable.

### Compression competition solution contract

The repository root includes `competition_solution.json`, which declares the
required `/health` and `/compress` routes. Competition requests use structured
`items` with local paths only: inputs must be below `/evaluation-inputs`, outputs
must be new `.mp4` paths below `/output`, and batches contain one to five unique
items. The validator creates each output parent directory before the call. Start
the service with `DISABLE_REMOTE_IO=true`; it rejects URLs, path
traversal, symlink escapes, duplicate outputs, and overwrites, then verifies that
each produced file is a smaller AV1 MP4 with the source dimensions, timing, YUV
pixel format, and square pixels preserved.

CQ selection lives in the compression service for both inference and
competition requests. An explicit `cq` remains the highest-priority legacy
override; otherwise `compression_type` maps Low/Medium/High to CQ 40/35/30, or
the same tiers are inferred from VMAF targets below 89, from 89 to below 93,
and 93 or above. Requests with no quality signal retain the CQ 35 default.

The legacy compression-inference request (`video_paths`, codec/mode/quality
options, and optional remote URLs) remains supported and unchanged. Before
submitting a repository, run:

```bash
python scripts/competition_preflight.py \
  competitions/manifests/examples/compression-competition.json \
  --repository .
```

The warmup fixture is a redistribution-safe five-second synthetic clip at
`competitions/fixtures/compression_warmup_input.mp4`.

## Competition SDK: prepare, validate, and publish a solution

Customize the compression implementation in `miner/compression/` and any supporting
files under `miner/`. Do not put a PAT, `.env`, wallet, Modal credentials, or S3
credentials in the solution directory.

The miner SDK creates a clean standalone repository under
`miner/.competition-sdk/<repository-name>/`. It copies the customized `miner/`
tree, solution descriptor, dependencies, and warmup fixture while excluding
local environments, bytecode, caches, Git metadata, and SDK exports.

Prepare the export first:

```bash
python miner/competition_sdk.py prepare \
  --repository your-github-user/private-compressor
```

If `--repository` is omitted, the SDK asks for a repository name. An existing
SDK export is reused; pass `--refresh` after changing the source solution to
rebuild it. Refresh only deletes directories carrying the SDK marker and refuses
to replace an arbitrary directory.

### Run the validator-equivalent Modal batch-of-one preflight

`miner/common_preflight.py` is shared by miners and the validator. Its runtime
probe does not mock inference. It:

1. Checks the standalone repository and five-second fixture.
2. Builds the prepared `miner/compression/Dockerfile` as a Modal Image without
   importing the submitted Python into the local SDK process.
3. Uploads `compression_warmup_input.mp4` to a disposable Modal input Volume and
   mounts it at `/evaluation-inputs/compression_warmup_input.mp4` read-only.
4. Mounts a separate, disposable v2 contender output Volume at `/output`
   read-write, proves that the input mount rejects writes, and explicitly tests
   the same `sync` commit used by the validator.
5. Starts the service in a validator-shaped Modal Sandbox: no secrets or OIDC
   identity, blocked outbound networking, validator-selected GPU, CPU request
   and hard limit, timeout, localhost-only route access, and no exposed ports.
6. Calls the service's local `/health` route and requires remote I/O to be off.
7. Sends exactly one CRF competition item using `hotkey`, `evaluation_id`, local
   `/evaluation-inputs` and `/output` paths, codec `AV1`, and the manifest VMAF
   threshold. Live evaluation also sends VBR variants with `codec_mode=VBR` and
   a validator-selected `target_bitrate` of 5, 8, or 10 Mbps.
8. Requires exactly one ordered response containing only the assigned output path.
9. Verifies that the input remains present and that only the requested output is
   created.
10. Uses `ffprobe` to verify AV1/MP4, unchanged dimensions, frame/duration
    tolerance, YUV pixel format, square pixels, and an output smaller than the
    source.
11. Probes the GPU model/count and CPU allocation in the running Sandbox,
    measures wall time through termination, and applies the validator's locked
    Modal Sandbox rates to those observed resources. The JSON report and stderr
    progress include `estimated_consumed_balance_usd`.

Install the miner dependencies and authenticate the Modal SDK. `validate` and a
`publish` without a matching validation receipt require a working Modal login;
`prepare` remains an offline operation:

```bash
pip install -r miner/requirements.txt
uvx modal setup
```

Validate the prepared export in the default `dev` Modal environment:

```bash
python miner/competition_sdk.py validate \
  --repository your-github-user/private-compressor \
  --manifest competitions/manifests/examples/compression-competition.json
```

Progress is streamed to stderr while the final report remains JSON on stdout.
The SDK prints each local/static step, disposable Volume creation and upload,
Modal's live image-build status (including its current/latest build line), the
Sandbox ID, service stdout/stderr, readiness and read-only-mount probes, the
batch-of-one `/compress` inference result, and Sandbox/Volume cleanup.
It then prints the estimated consumed balance. This is a public-rate compute
estimate before credits or reservations, not an immediate Modal invoice: it
excludes image-building, storage, taxes, and other workspace usage. Modal's
billing report data can be delayed and may require a supported workspace plan.
The locked 2026-07-21 GPU rates are B300 `$0.001972/s`, B200 `$0.001736/s`, H200
`$0.001261/s`, H100 `$0.001097/s`, RTX PRO 6000 `$0.000842/s`, A100 80 GB
`$0.000694/s`, A100 40 GB `$0.000583/s`, L40S `$0.000542/s`, A10 `$0.000306/s`,
L4 `$0.000222/s`, and T4 `$0.000164/s`. Modal's separate Sandbox CPU rate is
`$0.00003942/physical-core/s`. Requested and allocated resources are reported
separately; the estimate uses the allocation observed in the Sandbox and excludes
memory, CPU utilization reconciliation, and bursting not visible in that probe.

Use `--modal-environment`, `--modal-gpu`, `--modal-cpu`,
`--modal-cpu-limit`, and `--modal-timeout` only when the competition operator
specifies different manifest resources. Defaults are environment `dev`, GPU
`L40S`, CPU request/limit `16/32`, and a 30-minute timeout. The SDK terminates
the Sandbox and deletes both disposable Volumes after every pass or failure.
Modal image builds and GPU runtime can incur charges.

For `validate` and `publish`, pass the operator-provided manifest with
`--manifest`. The SDK reads `modal_build_timeout` from it, defaults to `10m`
when the field or option is absent, and builds the untrusted Dockerfile in a
killable client process. If that deadline expires, the SDK stops the build App,
rejects validation, and does not publish. The build timeout is included in the
validation receipt, so a receipt created under a different deadline is not
reused.

The image is always built from the prepared export—not a newer or older source
tree—so the tested code is exactly what will be uploaded. The common preflight
runs inside the same Sandbox as the service and accesses it only through
`127.0.0.1`; no public endpoint or tunnel is created.

The SDK isolates the frequently edited `miner/compression/app.py` and the shared
preflight into small final Modal Image layers. The heavyweight Docker toolchain
build uses a stable placeholder for `app.py`, so Python-only revisions reuse the
pinned FFmpeg toolchain build cache. Changes to the Dockerfile, native toolchain,
models, or Python requirements deliberately invalidate that base cache.

During a competition, the validator builds each pinned contender revision once,
records the immutable Modal Image ID, and keeps one Sandbox warm while sending
successive batches to it. It does not rebuild the image for each batch. A rebuild
is needed only for a different pinned revision or an intentional image change;
sandbox rollover reuses the already built image ID.

### Publish privately to GitHub

`validate` stores a local receipt outside the sanitized export. `publish`
always repeats the quick static check, then reuses a successful Modal result for
24 hours by default only when the export tree and all Modal resource settings
match exactly. This avoids building and running the same Sandbox twice:

```bash
python miner/competition_sdk.py publish \
  --repository your-github-user/private-compressor \
  --manifest competitions/manifests/examples/compression-competition.json
```

Use `--revalidate` to force a fresh Sandbox run. Publishing directly without a
matching receipt still performs the complete Modal validation before touching
GitHub.

The SDK asks for the GitHub PAT with hidden input. It never accepts a PAT as a
command-line argument, writes it to the export, embeds it in a URL, or adds a Git
remote. Git receives it only through a temporary askpass environment. For
non-interactive use, set `GITHUB_TOKEN` or select another variable with
`--pat-env`; avoid shell history and `.env` files.

By default the SDK creates a new **private** repository and refuses to push if
GitHub does not confirm it is private. If that repository already exists,
GitHub returns HTTP 422; update it with:

```bash
python miner/competition_sdk.py publish \
  --repository your-github-user/private-compressor \
  --use-existing \
  --refresh
```

`--refresh` rebuilds the sanitized export from the current local source.
`--use-existing` verifies that the remote is private, fetches its current
default branch, and creates a normal fast-forward child commit containing the
new export. It never force-pushes, so a concurrent remote update is rejected
instead of overwritten.

The commit author defaults to the GitHub account authenticated by the PAT,
using GitHub's account-specific no-reply email. The optional
`--git-author-name` and `--git-author-email` flags can override that identity.
The upload credential needs permission to create/administer the private
repository and push contents; GitHub documents the repository creation
endpoints and token permissions at
<https://docs.github.com/en/rest/repos/repos>. Use a separate, expiring,
contents-read-only PAT in `MINER_COMPETITION_GITHUB_PAT` for validator submission.

The publish result prints the canonical HTTPS `.git` URL to configure as
`MINER_COMPETITION_REPOSITORY_URL`.
