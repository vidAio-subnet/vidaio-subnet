# SN85 (Vidaio) Miner Deployment Guide

## Production Deployment Checklist

### 1. Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | RTX 3090 | RTX 4090 (24GB) |
| RAM | 32GB | 64GB DDR5 |
| Storage | 500GB SSD | 1TB NVMe Gen4 |
| Network | 100 Mbps | 1 Gbps |
| OS | Ubuntu 22.04 | Ubuntu 24.04 LTS |

### 2. System Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install core dependencies
sudo apt install -y \
    ffmpeg \
    redis-server \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    wget \
    cuda-toolkit-12-4

# Install NVIDIA drivers (for RTX 4090)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y nvidia-driver-550

# Install Video2X
wget https://github.com/k4yt3x/video2x/releases/download/6.4.0/video2x-linux-ubuntu2404-amd64.deb
sudo dpkg -i video2x-linux-ubuntu2404-amd64.deb
sudo apt-get install -f

# Verify installations
ffmpeg -version | head -1
video2x --version
nvidia-smi
```

### 3. Environment Setup

```bash
# Clone repository
git clone <vidaio-subnet-repo> ~/vidaio-subnet
cd ~/vidaio-subnet

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 4. Configuration

Create `.env` file:

```bash
# Required: Bittensor wallet
BT_WALLET_NAME="your_wallet_name"
BT_WALLET_HOTKEY="your_hotkey_name"

# Required: Storage (Backblaze B2 recommended)
BUCKET_COMPATIBLE_ENDPOINT="https://s3.us-west-002.backblazeb2.com"
BUCKET_COMPATIBLE_ACCESS_KEY="your_key"
BUCKET_COMPATIBLE_SECRET_KEY="your_secret"

# Required: Video source for organic jobs
PEXELS_API_KEY="your_pexels_key"

# Optional: Logging
LOG_LEVEL="INFO"

# Optional: Performance tuning
VIDEO2X_BINARY="video2x"
```

### 5. Wallet Setup

```bash
# Create new wallet (if needed)
btcli wallet create --wallet.name $BT_WALLET_NAME

# Or regenerate from seed
btcli wallet regen_coldkey --wallet.name $BT_WALLET_NAME
btcli wallet regen_hotkey --wallet.name $BT_WALLET_NAME --wallet.hotkey $BT_WALLET_HOTKEY

# Verify wallet
btcli wallet overview --wallet.name $BT_WALLET_NAME

# Register on subnet 85
btcli subnet register --netuid 85 --wallet.name $BT_WALLET_NAME --wallet.hotkey $BT_WALLET_HOTKEY
```

### 6. Service Startup

Using PM2 for process management:

```bash
# Install PM2
sudo npm install -g pm2

# Start Redis
sudo systemctl enable redis
sudo systemctl start redis

# Start compression service
pm2 start "python services/compress/server.py" \
    --name sn85-compressor \
    --log logs/compressor.log \
    --error logs/compressor-error.log

# Start upscaling service
pm2 start "python services/upscaling/server.py" \
    --name sn85-upscaler \
    --log logs/upscaler.log \
    --error logs/upscaler-error.log

# Start file deletion service
pm2 start "python services/miner_utilities/file_deletion_server.py" \
    --name sn85-deleter \
    --log logs/deleter.log

# Start miner
pm2 start "python neurons/miner.py \
    --wallet.name $BT_WALLET_NAME \
    --wallet.hotkey $BT_WALLET_HOTKEY \
    --netuid 85 \
    --subtensor.network finney \
    --logging.info" \
    --name sn85-miner \
    --log logs/miner.log \
    --error logs/miner-error.log

# Save PM2 config
pm2 save
pm2 startup
```

### 7. Health Verification

```bash
# Check all services
pm2 status

# Check miner logs
pm2 logs sn85-miner --lines 50

# Check GPU utilization
watch -n 1 nvidia-smi

# Check service endpoints
curl http://localhost:29115/health  # Upscaler
curl http://localhost:29116/health  # Compressor

# Run scoring simulator
python scoring_simulator.py --table --threshold 89
```

### 8. Monitoring Dashboard

```bash
# Enable PM2 monitoring
pm2 monitor

# Or use web dashboard
pm2 plus
```

## Troubleshooting

### Issue: Video2X fails with Vulkan error
**Solution**: Video2X requires GPU. For production, use RTX 4090 with proper NVIDIA drivers.

### Issue: Compression service won't start
**Solution**: Check if ports 29115/29116 are available:
```bash
sudo lsof -i :29115
sudo lsof -i :29116
```

### Issue: Wallet not registered
**Solution**: Ensure sufficient TAO for registration:
```bash
btcli wallet balance --wallet.name $BT_WALLET_NAME
btcli subnet register --netuid 85 --wallet.name $BT_WALLET_NAME --wallet.hotkey $BT_WALLET_HOTKEY
```

### Issue: Low scores on validator
**Solution**: Check CQ tuning and codec selection:
```bash
# Verify NVENC is being used
python -c "import torch; print(torch.cuda.is_available())"
ffmpeg -encoders 2>/dev/null | grep nvenc

# Run benchmark
python benchmark_compression.py --input test_video.mp4 --quality Medium
```

## Backup and Recovery

### Backup wallet
```bash
cp -r ~/.bittensor/wallets/$BT_WALLET_NAME ~/wallet-backup-$(date +%Y%m%d)
```

### Backup PM2 config
```bash
pm2 save
pm2 dump
```

### Recovery
```bash
# Restore wallet
cp -r ~/wallet-backup-YYYYMMDD ~/.bittensor/wallets/$BT_WALLET_NAME

# Restore services
pm2 resurrect
```

## Security Best Practices

1. **Never commit `.env` files** - Store secrets in environment variables
2. **Restrict SSH access** - Use key-based auth, disable root login
3. **Firewall rules**:
```bash
sudo ufw enable
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 8091/tcp  # Bittensor (if needed)
sudo ufw deny 29115/tcp  # Internal services (localhost only)
sudo ufw deny 29116/tcp
```
4. **Regular updates**:
```bash
sudo apt update && sudo apt upgrade -y
pm2 update
```

## Performance Tuning

### For RTX 4090 Optimization

The miner auto-detects RTX 4090 and applies these optimizations:
- `av1_nvenc` codec for best compression
- Adaptive CQ adjustment (+1 for GPU encoders)
- NVENC-specific encoding parameters

To verify optimal settings:
```bash
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')
"
```

### Expected Performance

| Metric | Target | Notes |
|--------|--------|-------|
| Upscale (2x) | ~2-4x real-time | Depends on input resolution |
| Compression | ~1-2x real-time | AV1 NVENC significantly faster |
| VMAF Score | ≥ threshold + 2 | 89 → 91+ for competitive scores |
| Compression Ratio | 15-20x | Sweet spot for scoring |

## Upgrade Procedure

```bash
# 1. Stop miner (keep services running)
pm2 stop sn85-miner

# 2. Pull updates
git pull origin main

# 3. Update dependencies
pip install -r requirements.txt

# 4. Restart services
pm2 restart sn85-compressor
pm2 restart sn85-upscaler
pm2 restart sn85-miner

# 5. Verify
pm2 logs sn85-miner --lines 20
```
