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
