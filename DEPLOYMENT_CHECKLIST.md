# SN85 Miner Deployment Checklist

Pre-deployment checklist for Bittensor SN85 (Vidaio) miner launch.

## Pre-Flight Status

| Component | Status | Location | Notes |
|-----------|--------|----------|-------|
| Encoder configs | Ready | `services/compress/utils/encoder_configs.py` | VMAF 93+ compliant CQ/CRF values |
| Docker deployment | Ready | `docker/` | Multi-stage, CUDA 12.4, PM2-managed |
| Health monitoring | Ready | `scripts/healthcheck.sh` | GPU, Redis, disk, memory checks |
| Scoring simulator | Ready | `scoring_simulator.py` | Validator score prediction |
| Benchmark tool | Ready | `benchmark_compression.py` | CQ optimization per target VMAF |

## Required Credentials

### Blockers (Cannot launch without)

| Credential | Purpose | Provider | Action |
|------------|---------|----------|--------|
| **CHUTES_API_KEY** | Remote inference (upscaling) | Chutes | Export as env var |
| **TAO funds** | Subnet registration | Bittensor | Register on netuid 85 |

### Optional (Can launch without, add later)

| Credential | Purpose | Provider | Fallback |
|------------|---------|----------|----------|
| BUCKET_COMPATIBLE_ACCESS_KEY | Cloud storage | Backblaze B2 | Local-only mode |
| BUCKET_COMPATIBLE_SECRET_KEY | Cloud storage | Backblaze B2 | Local-only mode |
| PEXELS_API_KEY | Stock footage API | Pexels | Skip synthetic jobs |
| WANDB_API_KEY | Experiment tracking | Weights & Biases | Disable logging |

## Hardware Requirements

| Component | Minimum | Recommended | Status |
|-----------|---------|-------------|--------|
| GPU | RTX 3090 | RTX 4090 | Pending operator |
| VRAM | 12 GB | 24 GB | Pending operator |
| System RAM | 32 GB | 64 GB DDR5 | Pending operator |
| Storage | 500 GB SSD | 1 TB NVMe Gen4 | Pending operator |
| OS | Ubuntu 22.04 | Ubuntu 24.04 LTS | Pending operator |

## Deployment Options

### Option A: Full Chutes (Immediate, no hardware)
- **Requirements**: CHUTES_API_KEY only
- **Upscale**: Remote (Chutes)
- **Compress**: CPU fallback (libsvtav1)
- **Expected**: Lower throughput, competitive scores
- **Command**:
  ```bash
  export CHUTES_API_KEY="your_key"
  pm2 start "python neurons/miner.py --netuid 85 --chutes" --name sn85-miner
  ```

### Option B: Local GPU (Best performance)
- **Requirements**: RTX 4090, CUDA 12.4, Video2X
- **Upscale**: Local (Video2X)
- **Compress**: NVENC (av1_nvenc/hevc_nvenc)
- **Expected**: Maximum throughput and scores
- **Command**:
  ```bash
  pm2 start "python neurons/miner.py --netuid 85 --axon.port 8091" --name sn85-miner
  ```

### Option C: Hybrid (Recommended)
- **Requirements**: RTX 4090 + CHUTES_API_KEY
- **Upscale**: Remote (Chutes) - frees local GPU for encoding
- **Compress**: NVENC (local)
- **Expected**: Balanced throughput, optimal for 24GB VRAM

## Launch Sequence

```bash
# 1. Verify environment
cd vidaio-subnet
python miner_monitor.py --save

# 2. Start Redis
sudo systemctl start redis

# 3. Start services (if local upscaling)
pm2 start services/upscaling/server.py --name upscaler
pm2 start services/compress/server.py --name compressor
pm2 start services/miner_utilities/file_deletion_server.py --name deleter

# 4. Start miner
pm2 start "python neurons/miner.py \
  --wallet.name YOUR_WALLET \
  --wallet.hotkey YOUR_HOTKEY \
  --subtensor.network finney \
  --netuid 85 \
  --axon.port 8091" \
  --name sn85-miner

# 5. Monitor
pm2 logs sn85-miner --lines 100 --follow
```

## Post-Launch Validation

| Check | Command | Expected Output |
|-------|---------|-----------------|
| Miner registered | `btcli w overview --wallet.name YOUR_WALLET` | Shows netuid 85 |
| Services running | `pm2 status` | All processes "online" |
| GPU utilized | `nvidia-smi` | Video2X or ffmpeg processes |
| Validator jobs | `pm2 logs sn85-miner --lines 50` | "Received job" entries |

## First 24 Hours

- [ ] Monitor logs for errors
- [ ] Check validator scores (when available)
- [ ] Verify compression ratios reported
- [ ] Confirm quality metrics in logs
- [ ] Adjust CQ values if VMAF consistently off-target

## Emergency Contacts

| Resource | Location |
|----------|----------|
| Miner docs | `docs/miner_setup.md` |
| Competitive config | `COMPETITIVE_CONFIG.md` |
| Troubleshooting | `miner_monitor.py --help` |
| Scoring sim | `scoring_simulator.py --analyze` |
