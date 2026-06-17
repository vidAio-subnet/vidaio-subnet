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
On the normal path, `neurons/miner.py` removes the downloaded input and processed output from `MINER_SHARED_DIR` after upload. The container services also remove their remote-mode temporary files after upload.

For crash and interruption paths, the miner and service containers run cleanup workers over `MINER_SHARED_DIR` / `SHARED_VOLUME_PATH`. By default, files are expired after the presigned URL expiry plus 600 seconds, and the cleanup worker keeps the shared volume under a 9 GB soft cap for Modal-style persistent storage.

Tune cleanup in `miner/.env`:
```bash
MINER_CLEANUP_ENABLED=true
MINER_CLEANUP_INTERVAL_SECONDS=300
MINER_CLEANUP_MAX_VOLUME_BYTES=9000000000
MINER_CLEANUP_MIN_FILE_AGE_SECONDS=60
MINER_TEMP_FILE_TTL_SECONDS=
PRESIGNED_URL_CLEANUP_GRACE_SECONDS=600
```

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
