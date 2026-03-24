# SN85 (Vidaio) Miner Docker Deployment

Production-grade containerized deployment for the Bittensor Subnet 85 video processing miner with GPU support, health monitoring, and automated failover.

## Features

- **GPU Support**: CUDA 12.4 with NVIDIA runtime for hardware-accelerated encoding
- **Multi-Service Architecture**: Separate containers for miner, upscaling, compression, and Redis
- **Health Monitoring**: Built-in health checks with degradation detection
- **Chutes Integration**: Optional remote inference for scene classification
- **PM2 Process Management**: Automatic restart and process monitoring
- **Automated Deployment**: One-command deployment with validation

## Quick Start

### 1. Prerequisites

```bash
# Install Docker and Docker Compose
# https://docs.docker.com/get-docker/

# Install NVIDIA Container Toolkit (for GPU support)
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
```

### 2. Configure Environment

```bash
cd docker
cp .env.example .env
nano .env  # Edit with your configuration
```

**Required settings:**
- `WALLET_NAME` / `WALLET_HOTKEY`: Your Bittensor wallet
- `BUCKET_*`: S3-compatible storage credentials (Backblaze/Cloudflare/AWS)

**Optional settings:**
- `USE_CHUTES=true` + `CHUTES_API_KEY`: Enable remote scene classification
- `GPU device reservations`: Adjust GPU allocation in docker-compose.miner.yml

### 3. Deploy

```bash
# Deploy with validation
./deploy.sh deploy

# Or using docker compose directly
docker compose -f docker-compose.miner.yml up -d
```

## Usage

### Monitor Status

```bash
# Real-time system monitor (Ctrl+C to exit)
./deploy.sh monitor

# Health check
./deploy.sh health

# Container status
./deploy.sh status
```

### View Logs

```bash
# Miner logs
./deploy.sh logs

# Specific service logs
./deploy.sh logs redis
./deploy.sh logs miner
```

### Manage Deployment

```bash
# Start/Stop/Restart
./deploy.sh start
./deploy.sh stop
./deploy.sh restart

# Shell access
./deploy.sh shell

# Update to latest code
./deploy.sh update

# Cleanup old images
./deploy.sh cleanup
```

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `WALLET_NAME` | Yes | default | Bittensor wallet name |
| `WALLET_HOTKEY` | Yes | default | Wallet hotkey name |
| `WALLET_PATH` | No | ~/.bittensor/wallets | Path to wallets |
| `BT_NETWORK` | No | finney | Bittensor network |
| `BT_NETUID` | No | 85 | Subnet UID |
| `BT_AXON_PORT` | No | 8091 | Miner axon port |
| `BUCKET_TYPE` | Yes | - | backblaze, amazon_s3, cloudflare |
| `BUCKET_NAME` | Yes | - | S3 bucket name |
| `BUCKET_COMPATIBLE_ENDPOINT` | Yes | - | S3 endpoint URL |
| `BUCKET_COMPATIBLE_ACCESS_KEY` | Yes | - | S3 access key |
| `BUCKET_COMPATIBLE_SECRET_KEY` | Yes | - | S3 secret key |
| `USE_CHUTES` | No | false | Enable Chutes inference |
| `CHUTES_API_KEY` | When USE_CHUTES | - | Chutes API key |
| `CHUTES_SCENE_CHUTE_ID` | No | scene-classifier-v1 | Chute endpoint |
| `WANDB_API_KEY` | No | - | Weights & Biases API key |

### Running Modes

#### GPU Mode (Recommended)

Uses local RTX 4090 or equivalent for encoding and inference:

```bash
# Default configuration - GPU detected automatically
./deploy.sh deploy
```

#### CPU-Only Mode (with Chutes)

For machines without GPU, use Chutes for scene classification:

```bash
# In .env:
USE_CHUTES=true
CHUTES_API_KEY=sk_xxx

./deploy.sh deploy
```

Note: Video encoding will be CPU-bound, significantly slower.

#### Hybrid Mode

Use Chutes for inference and local GPU for encoding:

```bash
# In .env:
USE_CHUTES=true
CHUTES_API_KEY=sk_xxx

# GPU will be used for encoding, Chutes for scene classification
./deploy.sh deploy
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Host System                        │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                sn85_miner Container                     │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │  │
│  │  │video-miner  │  │video-       │  │video-       │   │  │
│  │  │(Bittensor)  │  │upscaler     │  │compressor   │   │  │
│  │  │             │  │             │  │             │   │  │
│  │  │PM2 managed  │  │PM2 managed  │  │PM2 managed  │   │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘   │  │
│  │  ┌─────────────┐  ┌─────────────┐                     │  │
│  │  │file-deleter │  │healthcheck │                     │  │
│  │  │             │  │             │                     │  │
│  │  └─────────────┘  └─────────────┘                     │  │
│  └───────────────────────────────────────────────────────┘  │
│                          │                                  │
│  ┌───────────────────────┼───────────────────────────────┐  │
│  │              sn85_redis Container                      │  │
│  │               (Job Queue)                              │  │
│  └───────────────────────┬───────────────────────────────┘  │
│                          │                                  │
│  ┌───────────────────────┴───────────────────────────────┐  │
│  │              NVIDIA GPU (if available)                  │  │
│  │         (Encoding & Local Inference)                    │  │
│  └─────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## File Structure

```
docker/
├── Dockerfile.miner          # Multi-stage miner image build
├── docker-compose.miner.yml  # Production orchestration
├── .env.example              # Environment template
├── deploy.sh                 # Deployment automation
├── entrypoint.sh             # Container startup script
├── pm2.config.js             # Process management config
├── healthcheck.sh            # Health validation
├── monitor.sh                # System monitoring
└── README.md                 # This file
```

## Troubleshooting

### Container won't start

```bash
# Check logs
docker logs sn85_miner

# Validate config
docker exec sn85_miner /app/healthcheck.sh

# Check wallet permissions
ls -la ~/.bittensor/wallets
```

### GPU not detected

```bash
# Test NVIDIA runtime
docker run --rm --runtime=nvidia nvidia/cuda:12.0-base nvidia-smi

# If fails, install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    tee /etc/apt/sources.list.d/nvidia-docker.list
apt-get update && apt-get install -y nvidia-container-toolkit
systemctl restart docker
```

### Redis connection failed

```bash
# Check Redis is running
docker ps | grep sn85_redis

# Test Redis
docker exec sn85_redis redis-cli ping

# Check logs
docker logs sn85_redis
```

### High memory usage

```bash
# Enter container and check
docker exec -it sn85_miner /bin/bash
pm2 status
free -h

# Cleanup temp files
./deploy.sh cleanup
```

### PM2 processes not starting

```bash
# Check PM2 logs
docker exec sn85_miner pm2 logs

# Restart services
docker exec sn85_miner pm2 restart all
```

## Security

- Wallet directory is mounted **read-only** (`:ro`)
- Runs as non-root `miner` user
- No secret data in image layers
- Health checks expose minimal information

## Performance Tuning

### GPU Utilization

Monitor GPU usage and adjust `WORKER_PROCESSES` in environment:

```bash
# Monitor GPU
docker exec sn85_miner nvidia-smi

# Adjust worker count
# In .env: WORKER_PROCESSES=2
```

### Memory Limits

Adjust in `docker-compose.miner.yml`:

```yaml
deploy:
  resources:
    limits:
      memory: 32G
    reservations:
      memory: 16G
```

### Disk I/O

Mount temp directory to fast SSD:

```yaml
volumes:
  - /mnt/fast_ssd/temp:/data/temp
```

## Development

### Build image locally

```bash
docker build -f docker/Dockerfile.miner -t sn85-miner:dev .
```

### Run with shell

```bash
docker run -it --rm sn85-miner:dev shell
```

### Test health check

```bash
docker run --rm sn85-miner:dev health
```

## License

MIT License - See main project LICENSE file.
