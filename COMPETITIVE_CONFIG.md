# SN85 (Vidaio) Competitive Configuration Guide

## Executive Summary

This guide documents the production-optimized configuration for competitive SN85 mining. These settings maximize validator scores by optimizing the compression/quality tradeoff.

**Target Performance:**
- Compression ratio: 15-20x
- VMAF: ≥ threshold + 2 VMAF points
- Expected score: 0.75-0.80 (normalized)

## Hardware Requirements

| Component | Minimum | Competitive |
|-----------|---------|-------------|
| GPU | RTX 3090 | RTX 4090 (Ada architecture) |
| VRAM | 12 GB | 24 GB |
| System RAM | 32 GB | 64 GB DDR5 |
| Storage | 500 GB SSD | 1 TB NVMe Gen4 |
| Codec Support | HEVC NVENC | AV1 NVENC (Ada's 8th gen encoder) |

## Optimization Areas

### 1. Codec Selection

**Auto-Detection Logic:** (services/compress/server.py:154-195)

```python
# Priority order for RTX 4090:
'av1': ('av1_nvenc', True),    # Best compression ratio
'hevc': ('hevc_nvenc', True),  # Fallback compatibility

# CPU fallback:
'av1': ('libsvtav1', False),   # Best CPU AV1
'hevc': ('libx265', False),    # Fallback
```

**Why AV1 NVENC on RTX 4090:**
- 8th generation NVENC (Ada) has dedicated AV1 hardware
- Better compression than HEVC at same quality
- Faster encoding than CPU AV1

### 2. CQ (Constant Quality) Tuning

**CQ Lookup Tables:** (services/compress/encoder.py:63-97)

| Scene Type | High (93) | Medium (89) | Low (85) |
|------------|-----------|-------------|----------|
| animation | 22 | 28 | 31 |
| low-action | 23 | 26 | 29 |
| medium-action | 21 | 24 | 27 |
| high-action | 19 | 22 | 25 |

**GPU Adjustment:** (services/compress/encoder.py:707-725)

```python
if is_gpu_encoder:  # NVENC detected
    aggressive_adjustment = 1  # +1 CQ for better compression
    conservative_cq_adjustment = min(base + 1, 3)
    # Result: CQ can be 1 point higher than CPU for same VMAF
```

**Rationale:** NVENC at same CQ produces slightly higher bitrate than CPU encoders. +1 adjustment compensates while maintaining quality.

### 3. Upscaler Output Codec

**Configuration:** (services/upscaling/server.py:116-153)

```python
if torch.cuda.is_available():
    codec = "av1_nvenc"
    codec_params = [
        "-e", "preset=p2",        # p2 = balanced on Ada
        "-e", "qp=22",             # Visual quality ~ CRF 20
        "-e", "tile-columns=2",    # Parallel encoding for 4K
        "-e", "tile-rows=1",
    ]
else:
    codec = "libx265"
    codec_params = ["-e", "preset=slow", "-e", "crf=20"]
```

**Preset Selection:**
- AV1 NVENC: presets p1 (fastest) through p7 (slowest)
- p2 provides good quality/speed balance for RTX 4090
- NVENC quality is largely determined by QP value, not preset

### 4. Scoring Optimization

**Validator Scoring Formula:**
```
Score = (0.7 × compression_component + 0.3 × quality_component) / 1.12

Compression (ratio r):
  r ≤ 20: ((r - 1.25) / 18.75) ^ 0.9
  r > 20: 1.0 + 0.1 × ln(r / 20)

Quality (VMAF v, threshold t):
  v ≥ t: 0.7 + 0.3 × (v - t) / (100 - t)
  v < t: quadratic penalty

Score = 0 if: VMAF < (t - 5) OR compression_rate ≥ 0.80
```

**Sweet Spots:**

| Target | Compression | VMAF | Expected Score |
|--------|-------------|------|----------------|
| Threshold=89 | 15x | 89.0 | 0.71 |
| Threshold=89 | 15x | 91.0 | 0.75 |
| Threshold=89 | 20x | 89.0 | 0.74 |
| Threshold=89 | 20x | 91.0 | 0.78 |
| Threshold=93 | 15x | 93.0 | 0.71 |
| Threshold=93 | 15x | 95.0 | 0.75 |

**Strategy:** Target 15-20x compression at VMAF = threshold + 2

## Configuration Files

### 1. Environment (.env)

```bash
# Required
BT_WALLET_NAME="your_wallet"
BT_WALLET_HOTKEY="your_hotkey"
BUCKET_COMPATIBLE_ENDPOINT="https://s3.us-west-002.backblazeb2.com"
BUCKET_COMPATIBLE_ACCESS_KEY="xxx"
BUCKET_COMPATIBLE_SECRET_KEY="xxx"
PEXELS_API_KEY="xxx"

# Performance
VIDEO2X_BINARY="video2x"
LOG_LEVEL="INFO"

# Optional tuning
CONSERVATIVE_CQ_ADJUSTMENT=2
NVENC_CQ_ADJUSTMENT=1
```

### 2. Service Config (config.json)

Key parameters:

```json
{
  "video_processing": {
    "target_quality": "Medium",
    "target_codec": "av1_nvenc",
    "codec_mode": "CRF",
    "conservative_cq_adjustment": 2,
    "max_encoding_retries": 2,
    "size_increase_protection": true,
    "basic_cq_lookup_by_quality": {
      "High": {
        "animation": 22,
        "low-action": 23,
        "medium-action": 21,
        "high-action": 19,
        "default": 21
      },
      "Medium": {
        "animation": 28,
        "low-action": 26,
        "medium-action": 24,
        "high-action": 22,
        "default": 24
      },
      "Low": {
        "animation": 31,
        "low-action": 29,
        "medium-action": 27,
        "high-action": 25,
        "default": 27
      }
    }
  },
  "vmaf_calculation": {
    "calculate_full_video_vmaf": true,
    "vmaf_use_sampling": true,
    "vmaf_num_clips": 3,
    "vmaf_clip_duration": 2
  }
}
```

## Performance Benchmarks

### Expected Throughput (RTX 4090)

| Resolution | Task | Time |
|------------|------|------|
| 1080p → 4K | Upscale (2x) | ~0.5x real-time |
| 1080p | Compress AV1 NVENC | ~2-4x real-time |
| 4K | Compress AV1 NVENC | ~1-2x real-time |

### Quality Validation Pipeline

1. **Scene Classification** - AI model categorizes content
2. **CQ Selection** - Lookup table maps scene type → CQ value
3. **GPU Adjustments** - NVENC gets +1 CQ
4. **Encode** - Single-pass with size protection
5. **VMAF Validation** - Quality check against original

## Monitoring Commands

### Health Check
```bash
python miner_monitor.py
```

### Scoring Simulation
```bash
# Single scenario
python scoring_simulator.py --vmaf 91 --ratio 18 --threshold 89

# Full table
python scoring_simulator.py --table --threshold 89

# Find sweet spot
python scoring_simulator.py --analyze --threshold 93
```

### Service Logs
```bash
# Monitor all services
pm2 logs

# Specific service
pm2 logs sn85-miner --lines 100 --follow

# GPU utilization
watch -n 1 nvidia-smi
```

## Troubleshooting Performance Issues

### Low Compression Ratios
1. Check CQ values in config
2. Verify NVENC is being used: `ffmpeg -encoders | grep nvenc`
3. Review scene classification output

### Low VMAF Scores
1. Reduce CQ values by 1-2 points
2. Check if scene type classification is correct
3. Verify input video quality

### Slow Throughput
1. Check GPU utilization: `nvidia-smi dmon`
2. Verify AV1 NVENC (not CPU): check service logs
3. Monitor disk I/O: `iostat -x 1`

## Upgrade Checklist

When upgrading to RTX 4090 from older GPU:

- [ ] Update NVIDIA drivers to 550+
- [ ] Verify AV1 NVENC: `ffmpeg -encoders | grep av1_nvenc`
- [ ] Update CQ adjustment config (+1 for NVENC)
- [ ] Test with benchmark: `python benchmark_compression.py`
- [ ] Monitor scores for 24h to validate

## Validation Results

Run these to confirm optimal configuration:

```bash
# 1. Health check
python miner_monitor.py --save

# 2. Compression benchmark
python services/compress/server.py  # Run in background
curl -X POST http://localhost:29116/compress-video \
  -H "Content-Type: application/json" \
  -d '{"payload_url": "test_video.mp4", "vmaf_threshold": 89, "target_codec": "av1"}'

# 3. Scoring validation
python scoring_simulator.py --analyze --threshold 89
```

## References

- Files modified for optimization:
  - `services/upscaling/server.py` - Auto codec selection
  - `services/compress/server.py` - GPU detection & codec mapping
  - `services/compress/encoder.py` - NVENC CQ adjustment
  - Tools created:
    - `scoring_simulator.py` - Score prediction
    - `miner_monitor.py` - Health monitoring
    - `benchmark_compression.py` - Performance testing
