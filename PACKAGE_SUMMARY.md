# SN85 Miner Deployment Package Summary

**Generated**: 2026-03-23
**Version**: Step 9 - Production Ready
**Target Hardware**: Ubuntu 24.04 + RTX 4090

## Package Contents

### Core Optimizations (Modified)

| File | Lines | Optimization |
|------|-------|--------------|
| `services/compress/encoder.py` | 707-728 | NVENC-aware CQ adjustment (+1 for GPU encoders) |
| `services/compress/server.py` | 140-198 | Runtime GPU detection + codec mapping |
| `services/upscaling/server.py` | 116-136 | Auto av1_nvenc/libx265 selection |
| `requirements.txt` | - | Dependencies pinned for stability |

### Operational Tools (New)

| Tool | Purpose |
|------|---------|
| `start_miner.sh` | Unified service management (check/start/stop/status/benchmark) |
| `miner_monitor.py` | Health monitoring + scoring prediction |
| `scoring_simulator.py` | Pre-deployment validator score prediction |
| `benchmark_compression.py` | Throughput + quality benchmarking |

### Documentation

| Document | Purpose |
|----------|---------|
| `DEPLOYMENT.md` | Complete production deployment runbook |
| `COMPETITIVE_CONFIG.md` | RTX 4090 tuning guide with CQ tables |
| `PACKAGE_SUMMARY.md` | This file - deployment checklist |

## Quick Start

```bash
# 1. Transfer to production server
rsync -avz vidaio-subnet/ user@rtx4090-server:/opt/vidaio-subnet/

# 2. On production server, install dependencies
sudo apt update && sudo apt install -y ffmpeg redis python3-pip
wget https://github.com/k4yt3x/video2x/releases/download/6.4.0/video2x-6.4.0-linux-amd64.deb
sudo dpkg -i video2x-6.4.0-linux-amd64.deb

# 3. Configure environment
cp .env.example .env
# Edit: WALLET_NAME, HOTKEY_NAME, S3 credentials

# 4. Verify + Start
python miner_monitor.py
./start_miner.sh start
```

## Expected Performance (RTX 4090)

| Metric | Conservative | Competitive | Aggressive |
|--------|--------------|-------------|------------|
| Compression | 15x | 20x | 25x |
| VMAF Target | 91 | 92 | 90 |
| Expected Score | 0.79 | 0.83 | 0.85 |
| Risk Level | Low | Medium | High |

## Post-Deployment Checklist

- [ ] First 24h: Monitor VMAF scores per scene type
- [ ] First 48h: Adjust `nvenc_cq_adjustment` if VMAF consistently above threshold
- [ ] Week 1: Benchmark throughput vs duration targets
- [ ] Week 1: Analyze validator score patterns
- [ ] Week 2: Consider scaling to multiple GPUs if profitable

## Support

- SN85 Documentation: `neurons/validator.py` (reference)
- Scoring Logic: `services/scoring/scoring_function.py`
- CQ Tables: `services/compress/config_optimized_rtx4090.json`
