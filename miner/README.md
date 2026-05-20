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
- A shared local work directory. The default is `/tmp/vidaio-miner-video-tmp` via `ORGANIC_PROXY_SHARED_DIR`.

The service containers run on an internal Docker network with no egress. `neurons/miner.py` downloads validator payloads, places local files in `ORGANIC_PROXY_SHARED_DIR`, forwards local paths to the containers, uploads processed outputs, and returns presigned URLs.

```bash
mkdir -p /tmp/vidaio-miner-video-tmp
chmod 777 /tmp/vidaio-miner-video-tmp
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
- `MINER_SHARED_VOLUME_PATH` defaults to `ORGANIC_PROXY_SHARED_DIR`, then `/tmp/vidaio-miner-video-tmp`.
