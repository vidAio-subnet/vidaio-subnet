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

## Export Prebuilt Compose
`miner/export_compose.py` builds the selected miner services locally, uploads the resulting images to Docker Hub, and writes an exported `docker-compose.yml` that contains `image:` references only. The exported compose file has no `build:` blocks, so it can be shared as a single deployment file once the recipient can authenticate to the private registry.

1. Copy `miner/.env.template` to `miner/.env`.
2. Fill in the Docker Hub export settings:
   - `DOCKERHUB_USERNAME`
   - `DOCKERHUB_NAMESPACE`
   - `DOCKERHUB_WRITE_API_KEY`
   - `DOCKERHUB_READONLY_API_KEY`
3. Run the exporter:
   ```bash
   python3 miner/export_compose.py
   ```
4. Enter the project root when prompted:
   - `miner` exports compression, Video2X upscaling, and FFmpeg upscaling.
   - `miner/upscaling` exports only the Video2X upscaling wrapper.
   - `miner/upscaling/ffmpeg` exports only the FFmpeg upscaling service.
   - `miner/compression` exports only compression.

By default the exporter creates or uses private Docker Hub repositories named like `DOCKERHUB_NAMESPACE/vidaio-miner-compression:<tag>`. It verifies the write API key by creating/using the private repositories and pushing the built images. If the private repositories do not already exist, the write key must also be allowed to create repositories in that namespace, such as a Docker Hub token with `repo:admin` scope or equivalent namespace permission. It verifies the read-only API key by confirming repository read access, inspecting the pushed image manifests, and checking that the token cannot push.

Useful non-interactive example:
```bash
python3 miner/export_compose.py \
  --project-root miner/upscaling/ffmpeg \
  --tag ffmpeg-$(date -u +%Y%m%d%H%M%S) \
  --output-dir miner/exported/upscaling-ffmpeg
```

The recipient of a private export must log in with the read-only API key before running the exported file:
```bash
echo "$DOCKERHUB_READONLY_API_KEY" | docker login docker.io \
  --username "$DOCKERHUB_USERNAME" \
  --password-stdin
docker compose -f docker-compose.yml --profile upscaling-ffmpeg up -d
```

For offline compose-generation tests only, use `--skip-registry-check --skip-build --skip-push`; production exports should leave those checks enabled.
