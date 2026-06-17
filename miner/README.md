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

For Modal deployments, the worker functions set `MAX_QUEUE_SIZE_*` to `0` and `@modal.concurrent(max_inputs=1)`. Modal owns the pending-input queue and scales out containers instead of persisting work in the Python process.

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
`miner/modal_workers.py` deploys three Modal workers:

- `upscale_video2x`: Video2X upscaling worker.
- `upscale_ffmpeg`: FFmpeg upscaling worker.
- `compress`: compression worker.

Each worker uses `cpu=16.0`, `gpu="RTX-PRO-6000"`, `min_containers=0`, `max_containers=5` by default, `scaledown_window=300`, and one input per container. A burst of 3 to 5 requests is handled by Modal scale-out; after the idle window, the GPU functions scale back to zero.

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
modal deploy miner/modal_workers.py
```

Configure `miner/.env` for Modal processing:

```bash
MINER_PROCESSING_BACKEND=modal
MODAL_APP_NAME=vidaio-miner-workers
MODAL_MINER_SECRET_NAME=vidaio-miner-secrets
MINER_MODAL_UPSCALING_FUNCTION=upscale_video2x
MINER_MODAL_COMPRESSION_FUNCTION=compress
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

For direct SDK tests, run:

```bash
modal run miner/modal_workers.py --worker video2x --video-url "https://example.com/input.mp4" --scale 2
modal run miner/modal_workers.py --worker ffmpeg --video-url "https://example.com/input.mp4" --scale 2
modal run miner/modal_workers.py --worker compression --video-url "https://example.com/input.mp4" --codec AV1 --codec-mode CRF --cq 35
```

For direct HTTP endpoint tests, use the URLs printed by `modal deploy` for `upscaling_video2x_api`, `upscaling_ffmpeg_api`, and `compression_api`:

```bash
curl -X POST "$MODAL_VIDEO2X_URL/upscale" \
  -H "Content-Type: application/json" \
  -d '{"video_path":"https://example.com/input.mp4","scale":2,"task_id":"manual-video2x"}'

curl -X POST "$MODAL_FFMPEG_URL/upscale" \
  -H "Content-Type: application/json" \
  -d '{"video_path":"https://example.com/input.mp4","scale":2,"task_id":"manual-ffmpeg"}'

curl -X POST "$MODAL_COMPRESSION_URL/compress" \
  -H "Content-Type: application/json" \
  -d '{"video_path":"https://example.com/input.mp4","task_id":"manual-compress","codec":"AV1","codec_mode":"CRF","cq":35}'
```
