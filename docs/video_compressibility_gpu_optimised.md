# Video Compressibility Test Dataset — GPU-Optimised Pipeline

> **Target hardware**: NVIDIA A6000 / A100 (data-centre GPUs with NVENC but **no NVDEC**)
>
> This document focuses on maximising throughput using NVENC-only encoding with CPU-side decoding and filtering.

---

## GPU Architecture Context

### NVENC vs NVDEC on Data-Centre GPUs

| GPU | NVENC Sessions | NVDEC | Compute | VRAM |
|-----|---------------|-------|---------|------|
| A100 | ✅ Unlimited* | ❌ None | 6912 CUDA cores | 40/80 GB |
| A6000 | ✅ Unlimited* | ❌ None | 10752 CUDA cores | 48 GB |
| A40 | ✅ Unlimited* | ❌ None | 10752 CUDA cores | 48 GB |
| Consumer (e.g. RTX 4090) | ⚠️ 8 max | ✅ Yes | 16384 CUDA cores | 24 GB |

\* Data-centre GPUs have **no artificial NVENC session limit** (consumer cards cap at 3-8 concurrent sessions unless patched).

> [!IMPORTANT]
> Since A6000/A100 have **no NVDEC**, the pipeline is always: **CPU decode → CPU filter → GPU encode (NVENC)**. Never use `-hwaccel cuda` or `-hwaccel cuvid` — these require NVDEC and will fail.

### What This Means for the Pipeline

```
Input file → [CPU: demux + decode] → [CPU: filter graph] → [GPU: NVENC encode] → Output file
                  FFmpeg                    FFmpeg                NVENC HW
                 libavcodec               libavfilter            on-chip encoder
```

The CPU handles all decoding and filtering. The GPU's NVENC ASIC handles only the final encode step. This means:
- **CPU is the bottleneck** for complex filter chains (noise, scale, denoise)
- **NVENC is the bottleneck** only when filter chains are trivial (trim + encode)
- The key optimisation is **parallelism**: run many FFmpeg processes to saturate both CPU cores and NVENC sessions

---

## Core FFmpeg Commands (NVENC-Only, No NVDEC)

### Base Pattern — CPU Decode, NVENC Encode

```bash
# H.264 via NVENC — NO hwaccel flag (CPU decode)
ffmpeg -i input.mp4 \
  -c:v h264_nvenc -preset p4 -rc constqp -qp 18 \
  -c:a copy output.mp4

# H.265 via NVENC
ffmpeg -i input.mp4 \
  -c:v hevc_nvenc -preset p4 -rc constqp -qp 20 \
  -c:a copy output.mp4
```

> [!CAUTION]
> Do **not** use `-hwaccel cuda`, `-hwaccel cuvid`, or `-hwaccel nvdec`. These require NVDEC hardware which A6000/A100 lack. FFmpeg will error with `No decoder surfaces left` or `HW accel not available`.

### NVENC Preset Reference

NVENC has its own preset system (different from libx264 presets):

| NVENC Preset | Speed | Quality | Use Case |
|-------------|-------|---------|----------|
| `p1` | Fastest | Lowest | Real-time streaming, bulk test data |
| `p2` | Very fast | Low | High-throughput batch processing |
| `p4` | Balanced | Good | **Recommended default for test data** |
| `p5` | Slow | High | Quality-sensitive test variants |
| `p7` | Slowest | Highest | Reference quality (rarely needed) |

### NVENC Rate Control Modes

```bash
# Constant QP (fastest, predictable quality — BEST FOR TEST DATA)
-rc constqp -qp 18

# CRF-like via CQ mode (good quality/size balance)
-rc vbr -cq 23 -b:v 0

# Target bitrate (useful for testing specific bitrate scenarios)
-rc cbr -b:v 5M

# Two-pass VBR (slow but best quality — rarely needed for test data)
-rc vbr -2pass 1 -b:v 5M
```

> [!TIP]
> For test data generation, always use `-rc constqp` with `-qp`. It's the fastest mode and gives deterministic quality. Save VBR/CBR modes for when you specifically need bitrate-controlled variants.

---

## Compressibility Factor Transformations (GPU-Optimised)

### Resolution Scaling

```bash
# Scale with CPU filter → NVENC encode
# Use scale_npp if available for GPU-side scaling (faster)
ffmpeg -i input.mp4 \
  -vf "scale=1920:1080:flags=lanczos" \
  -c:v h264_nvenc -preset p4 -rc constqp -qp 18 \
  out_1080p.mp4

ffmpeg -i input.mp4 \
  -vf "scale=1280:720:flags=lanczos" \
  -c:v h264_nvenc -preset p4 -rc constqp -qp 18 \
  out_720p.mp4

ffmpeg -i input.mp4 \
  -vf "scale=854:480:flags=lanczos" \
  -c:v h264_nvenc -preset p4 -rc constqp -qp 18 \
  out_480p.mp4
```

#### GPU-Side Scaling with scale_npp (If Available)

If your FFmpeg is compiled with `--enable-libnpp` (NVIDIA Performance Primitives), you can offload scaling to the GPU:

```bash
# Upload frames to GPU, scale on GPU, encode on GPU
# Avoids CPU→GPU transfer for raw frames
ffmpeg -i input.mp4 \
  -vf "hwupload_cuda,scale_npp=1280:720:interp_algo=lanczos" \
  -c:v h264_nvenc -preset p4 -rc constqp -qp 18 \
  out_720p.mp4
```

> [!NOTE]
> `scale_npp` requires frames to be in GPU memory. The `hwupload_cuda` filter transfers CPU-decoded frames to the GPU. This is beneficial when scaling is the dominant operation, but adds overhead for small files.

### Frame Rate Conversion

```bash
# FPS change is a CPU filter, then NVENC encodes the result
ffmpeg -i input.mp4 \
  -vf "fps=24" \
  -c:v h264_nvenc -preset p4 -rc constqp -qp 18 \
  out_24fps.mp4

ffmpeg -i input.mp4 \
  -vf "fps=60" \
  -c:v h264_nvenc -preset p4 -rc constqp -qp 18 \
  out_60fps.mp4
```

### Noise Injection (Spatial Complexity)

```bash
# Noise filter runs on CPU, encode on GPU
ffmpeg -i input.mp4 \
  -vf "noise=alls=20:allf=t" \
  -c:v h264_nvenc -preset p4 -rc constqp -qp 18 \
  out_noisy.mp4

ffmpeg -i input.mp4 \
  -vf "noise=alls=50:allf=t+u" \
  -c:v h264_nvenc -preset p1 -rc constqp -qp 18 \
  out_very_noisy.mp4
```

### Denoising (CPU-Bound — Slowest Transform)

```bash
# nlmeans is VERY CPU-intensive — this is the bottleneck transform
ffmpeg -i input.mp4 \
  -vf "nlmeans=s=6:p=7:r=15" \
  -c:v h264_nvenc -preset p4 -rc constqp -qp 18 \
  out_denoised.mp4

# FASTER alternative: hqdn3d (lower quality but 5-10x faster)
ffmpeg -i input.mp4 \
  -vf "hqdn3d=4:4:6:6" \
  -c:v h264_nvenc -preset p4 -rc constqp -qp 18 \
  out_denoised_fast.mp4
```

> [!WARNING]
> `nlmeans` is the slowest filter in the pipeline (~2-5 fps on 1080p). For bulk generation, use `hqdn3d` instead — it's 5-10x faster with acceptable quality for test data purposes.

### Blur / Sharpen

```bash
# Blur (more compressible output)
ffmpeg -i input.mp4 \
  -vf "gblur=sigma=2" \
  -c:v h264_nvenc -preset p4 -rc constqp -qp 18 \
  out_blurred.mp4

# Sharpen (less compressible output)
ffmpeg -i input.mp4 \
  -vf "unsharp=5:5:1.5" \
  -c:v h264_nvenc -preset p4 -rc constqp -qp 18 \
  out_sharp.mp4
```

### Speed Changes

```bash
# 2x speed (drops frames, doubles temporal delta)
ffmpeg -i input.mp4 \
  -vf "setpts=0.5*PTS" -an \
  -c:v h264_nvenc -preset p4 -rc constqp -qp 18 \
  out_2x.mp4

# 0.5x speed (duplicates frames, halves temporal delta)
ffmpeg -i input.mp4 \
  -vf "setpts=2.0*PTS" -an \
  -c:v h264_nvenc -preset p4 -rc constqp -qp 18 \
  out_half.mp4
```

### Quality Level Variations (Pre-compression)

```bash
# Near-lossless
ffmpeg -i input.mp4 -c:v h264_nvenc -preset p4 -rc constqp -qp 8 out_qp8.mp4

# Medium quality
ffmpeg -i input.mp4 -c:v h264_nvenc -preset p4 -rc constqp -qp 23 out_qp23.mp4

# Low quality (heavy artefacts)
ffmpeg -i input.mp4 -c:v h264_nvenc -preset p4 -rc constqp -qp 38 out_qp38.mp4
```

### Combined Filters (Single Pass — Most Efficient)

```bash
# Chain all CPU filters THEN encode once on GPU
ffmpeg -i input.mp4 \
  -vf "scale=1280:720:flags=lanczos,noise=alls=20:allf=t,fps=24" \
  -c:v h264_nvenc -preset p4 -rc constqp -qp 18 \
  out_720p_noisy_24fps.mp4
```

> [!IMPORTANT]
> Always chain filters into a single `-vf` string. Running separate FFmpeg passes means decode→encode→decode→encode, wasting both CPU and GPU time. A single pass with chained filters is 2-4x faster.

---

## Stream Copy Optimisation (Bypass GPU Entirely)

For operations that don't modify pixel data, skip both CPU decode and GPU encode:

```bash
# Trim a clip (instant — no decode/encode)
ffmpeg -ss 10 -i input.mp4 -t 30 -c copy -avoid_negative_ts make_start trimmed.mp4

# Extract multiple clips from one video (very fast)
ffmpeg -i input.mp4 \
  -ss 0 -t 30 -c copy -avoid_negative_ts make_start clip1.mp4 \
  -ss 60 -t 30 -c copy -avoid_negative_ts make_start clip2.mp4 \
  -ss 120 -t 30 -c copy -avoid_negative_ts make_start clip3.mp4
```

---

## Codec Variations via NVENC

### H.264 Profiles

```bash
# Baseline (most compatible, no B-frames — different compression characteristics)
ffmpeg -i input.mp4 -c:v h264_nvenc -profile:v baseline -preset p4 -rc constqp -qp 23 out_h264_baseline.mp4

# Main (B-frames enabled)
ffmpeg -i input.mp4 -c:v h264_nvenc -profile:v main -preset p4 -rc constqp -qp 23 out_h264_main.mp4

# High (most compression tools available)
ffmpeg -i input.mp4 -c:v h264_nvenc -profile:v high -preset p4 -rc constqp -qp 23 out_h264_high.mp4
```

### H.265 via NVENC

```bash
# Main profile, 8-bit
ffmpeg -i input.mp4 -c:v hevc_nvenc -profile:v main -preset p4 -rc constqp -qp 24 out_h265.mp4

# Main10 profile, 10-bit (larger file, finer gradients)
ffmpeg -i input.mp4 -c:v hevc_nvenc -profile:v main10 -preset p4 -rc constqp -qp 24 \
  -pix_fmt p010le out_h265_10bit.mp4
```

### VP9 and AV1 (CPU-Only Codecs — Parallel Strategy)

These codecs have no NVENC support, so they must use CPU encoders. Run them in parallel with NVENC jobs to utilise both CPU and GPU simultaneously:

```bash
# VP9 (CPU only — run alongside NVENC jobs)
ffmpeg -i input.mp4 -c:v libvpx-vp9 -crf 30 -b:v 0 -row-mt 1 -threads 4 out_vp9.webm

# AV1 via SVT-AV1 (CPU only — fastest AV1 encoder)
ffmpeg -i input.mp4 -c:v libsvtav1 -crf 30 -preset 8 out_av1.mp4
```

---

## Parallelism Strategy for A6000/A100

### Key Insight: NVENC is a Dedicated ASIC

NVENC is a **separate hardware block** from the CUDA cores. This means:
- NVENC encodes don't use CUDA cores at all
- You can run NVENC encoding + CUDA workloads simultaneously
- Multiple NVENC sessions run in parallel (unlimited on data-centre cards)
- The limiting factor is **memory bandwidth** and **encoder throughput**, not sessions

### Optimal Parallelism

```bash
# Determine optimal parallel FFmpeg processes
CPU_CORES=$(nproc)
# Rule of thumb: NVENC saturates at ~8-12 concurrent 1080p encodes
# CPU decode is the bottleneck, so match to CPU capability
NVENC_JOBS=$((CPU_CORES / 2))   # Each FFmpeg job uses ~2 CPU threads for decode
MAX_NVENC=12                     # NVENC throughput ceiling
PARALLEL=$(( NVENC_JOBS < MAX_NVENC ? NVENC_JOBS : MAX_NVENC ))

echo "Running $PARALLEL parallel FFmpeg+NVENC jobs"
```

### Recommended Parallelism by GPU

| GPU | CPU Cores Available | Recommended Parallel Jobs | Expected Throughput (1080p) |
|-----|-------------------|--------------------------|---------------------------|
| A100 (DGX / cloud) | 16-64 | 8-12 NVENC + 4-8 CPU-only | ~400-800 fps total |
| A6000 (workstation) | 16-32 | 6-10 NVENC + 2-4 CPU-only | ~300-600 fps total |
| A40 (cloud) | 8-16 | 4-8 NVENC + 2-4 CPU-only | ~200-400 fps total |

### GNU Parallel Integration

```bash
#!/bin/bash
# generate_variants.sh — parallel NVENC pipeline

SEEDS_DIR="seeds"
OUTPUT_DIR="dataset"
GPU_ID=0

mkdir -p "$OUTPUT_DIR"

# Define transformations as functions
generate_variant() {
    local input="$1"
    local basename=$(basename "$input" .mp4)
    local gpu="$2"

    # Resolution variants (NVENC)
    for res in "1920:1080" "1280:720" "854:480"; do
        local label=$(echo $res | tr ':' 'x')
        ffmpeg -loglevel warning -i "$input" \
            -vf "scale=${res}:flags=lanczos" \
            -c:v h264_nvenc -gpu "$gpu" -preset p4 -rc constqp -qp 18 \
            -c:a copy -y \
            "${OUTPUT_DIR}/${basename}_${label}.mp4"
    done

    # Noise variants (NVENC)
    for noise in 0 20 50; do
        local filter=""
        if [ "$noise" -gt 0 ]; then
            filter="-vf noise=alls=${noise}:allf=t"
        fi
        ffmpeg -loglevel warning -i "$input" \
            $filter \
            -c:v h264_nvenc -gpu "$gpu" -preset p4 -rc constqp -qp 18 \
            -c:a copy -y \
            "${OUTPUT_DIR}/${basename}_noise${noise}.mp4"
    done

    # Quality variants (NVENC)
    for qp in 8 18 28 38; do
        ffmpeg -loglevel warning -i "$input" \
            -c:v h264_nvenc -gpu "$gpu" -preset p4 -rc constqp -qp "$qp" \
            -c:a copy -y \
            "${OUTPUT_DIR}/${basename}_qp${qp}.mp4"
    done

    # H.265 variant (NVENC)
    ffmpeg -loglevel warning -i "$input" \
        -c:v hevc_nvenc -gpu "$gpu" -preset p4 -rc constqp -qp 20 \
        -c:a copy -y \
        "${OUTPUT_DIR}/${basename}_h265.mp4"

    echo "Done: $basename"
}

export -f generate_variant
export OUTPUT_DIR

# Run in parallel — adjust -j based on GPU/CPU capacity
find "$SEEDS_DIR" -name "*.mp4" | \
    parallel -j 8 generate_variant {} "$GPU_ID"
```

### Multi-GPU Support

If the machine has multiple GPUs (common in DGX or multi-GPU cloud instances):

```bash
# Distribute jobs across GPUs round-robin
NUM_GPUS=$(nvidia-smi -L | wc -l)

find seeds/ -name "*.mp4" | \
    parallel -j $((8 * NUM_GPUS)) \
    'GPU_ID=$(( {#} % '"$NUM_GPUS"' )); generate_variant {} $GPU_ID'
```

### Hybrid CPU+GPU Pipeline

Since VP9/AV1 are CPU-only, run them in parallel with NVENC jobs to utilise all hardware:

```bash
#!/bin/bash
# hybrid_pipeline.sh — saturate both CPU and GPU

NVENC_PIDS=()
CPU_PIDS=()

# Phase 1: NVENC jobs (H.264 + H.265 for all videos)
for f in seeds/*.mp4; do
    base=$(basename "$f" .mp4)

    # H.264 via NVENC (GPU)
    ffmpeg -loglevel warning -i "$f" \
        -c:v h264_nvenc -preset p4 -rc constqp -qp 18 \
        -y "dataset/${base}_h264.mp4" &
    NVENC_PIDS+=($!)

    # H.265 via NVENC (GPU)
    ffmpeg -loglevel warning -i "$f" \
        -c:v hevc_nvenc -preset p4 -rc constqp -qp 20 \
        -y "dataset/${base}_h265.mp4" &
    NVENC_PIDS+=($!)

    # VP9 via CPU (runs alongside NVENC, uses different hardware)
    ffmpeg -loglevel warning -i "$f" \
        -c:v libvpx-vp9 -crf 30 -b:v 0 -row-mt 1 -threads 4 \
        -y "dataset/${base}_vp9.webm" &
    CPU_PIDS+=($!)

    # Throttle: max 8 NVENC + 4 CPU concurrent
    if [ ${#NVENC_PIDS[@]} -ge 8 ]; then
        wait "${NVENC_PIDS[0]}"
        NVENC_PIDS=("${NVENC_PIDS[@]:1}")
    fi
    if [ ${#CPU_PIDS[@]} -ge 4 ]; then
        wait "${CPU_PIDS[0]}"
        CPU_PIDS=("${CPU_PIDS[@]:1}")
    fi
done

wait  # Wait for all remaining jobs

# Phase 2: AV1 (CPU-only, run after NVENC jobs free up CPU)
find seeds/ -name "*.mp4" | \
    parallel -j 4 'base=$(basename {} .mp4); \
    ffmpeg -loglevel warning -i {} \
        -c:v libsvtav1 -crf 30 -preset 8 \
        -y "dataset/${base}_av1.mp4"'
```

---

## Throughput Benchmarks (Expected)

### NVENC Encoding Speed (A6000, 1080p, No Filters)

| Preset | H.264 (h264_nvenc) | H.265 (hevc_nvenc) |
|--------|--------------------|--------------------|
| p1 | ~800-1000 fps | ~500-700 fps |
| p4 | ~400-600 fps | ~250-400 fps |
| p7 | ~150-250 fps | ~100-180 fps |

### Effective Pipeline Speed (CPU Decode + Filter + NVENC Encode)

| Pipeline | Per-Process FPS | 8 Parallel | Total FPS |
|----------|----------------|------------|-----------|
| Trim only (`-c copy`) | ∞ (I/O bound) | 8 | ~10,000+ fps |
| Decode + NVENC (no filter) | ~400 fps | 8 | ~2,000 fps |
| Scale 4K→1080p + NVENC | ~150 fps | 8 | ~800 fps |
| Noise + Scale + NVENC | ~80 fps | 8 | ~500 fps |
| hqdn3d + NVENC | ~40 fps | 6 | ~200 fps |
| nlmeans + NVENC | ~5 fps | 4 | ~20 fps |

### Estimated Total Pipeline Time (40 Seeds → 200-500 Variants)

| Scenario | Time |
|----------|------|
| All H.264, basic transforms, NVENC p4 | ~30-60 min |
| + H.265 variants | ~60-90 min |
| + VP9/AV1 subset (CPU parallel) | ~2-4 hours |
| + nlmeans denoised variants | +2-3 hours |
| **Total (everything)** | **~4-8 hours** |

---

## FFmpeg Build Requirements

Ensure FFmpeg is compiled with NVENC support (no NVDEC needed):

```bash
# Check NVENC availability
ffmpeg -encoders 2>/dev/null | grep nvenc
# Should show: h264_nvenc, hevc_nvenc (possibly av1_nvenc on newer GPUs)

# Check that NVDEC is NOT required (don't use -hwaccel)
ffmpeg -i input.mp4 -c:v h264_nvenc -preset p4 -f null - 2>&1 | head -5
```

### Minimal Build Flags for NVENC-Only

```bash
./configure \
  --enable-nonfree \
  --enable-cuda-nvcc \
  --enable-nvenc \
  --enable-libx264 \        # CPU H.264 fallback
  --enable-libx265 \        # CPU H.265 fallback
  --enable-libvpx \         # VP9 CPU encoder
  --enable-libsvtav1 \      # AV1 CPU encoder
  --enable-gpl \
  --extra-cflags="-I/usr/local/cuda/include" \
  --extra-ldflags="-L/usr/local/cuda/lib64"
# Note: --enable-nvdec and --enable-cuvid are NOT needed
```

---

## Checklist: GPU-Optimised Dataset Generation

- [ ] Verify NVENC works: `ffmpeg -i test.mp4 -c:v h264_nvenc -f null -`
- [ ] Confirm no NVDEC: do NOT use `-hwaccel cuda`
- [ ] Download ~40 seed videos via yt-dlp
- [ ] Run trim passes first (`-c copy` — instant)
- [ ] Run NVENC H.264/H.265 transform variants (parallel -j 8)
- [ ] Run CPU VP9/AV1 variants in parallel with NVENC jobs
- [ ] Generate synthetic edge cases (black, noise, static)
- [ ] Create metadata manifest (JSON/CSV)
- [ ] Validate output: spot-check 10 random videos with `ffprobe`
