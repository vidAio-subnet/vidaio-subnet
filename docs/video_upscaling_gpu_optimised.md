# Video Upscaling Test Dataset — GPU-Optimised Pipeline

> **Target hardware**: NVIDIA A6000 / A100 (data-centre GPUs with NVENC but **no NVDEC**)
>
> This document focuses on maximising throughput when generating paired (input, reference) upscaling test data using NVENC-only encoding with CPU-side decoding and filtering.

---

## GPU Architecture Recap

| GPU | NVENC | NVDEC | Sessions | VRAM |
|-----|-------|-------|----------|------|
| A100 | ✅ Unlimited | ❌ None | No cap | 40/80 GB |
| A6000 | ✅ Unlimited | ❌ None | No cap | 48 GB |
| A40 | ✅ Unlimited | ❌ None | No cap | 48 GB |

**Pipeline for these GPUs is always:**
```
Input file → [CPU: demux + decode] → [CPU: filter/scale] → [GPU: NVENC encode] → Output
```

> [!CAUTION]
> Never use `-hwaccel cuda`, `-hwaccel cuvid`, or `-hwaccel nvdec`. These require NVDEC hardware and will fail on A6000/A100 with `No decoder surfaces left` or similar errors.

---

## Upscaling Test Data Structure

Unlike compression testing, upscaling requires **paired data**:

```
dataset/
├── refs/           ← High-res ground truth (4K preferred)
│   ├── seed01_ref.mp4
│   └── seed02_ref.mp4
├── inputs/         ← Downscaled test inputs (what the upscaler receives)
│   ├── seed01_2x_lanczos_h264_qp10.mp4
│   ├── seed01_4x_bicubic_h265_qp20.mp4
│   └── ...
└── manifest.json   ← Metadata mapping inputs → references
```

**The generation bottleneck** is creating the `inputs/` directory — each reference gets downscaled into many variants across scale factors, downscale methods, codecs, and quality levels.

---

## Phase 1: Download & Trim (Zero GPU Cost)

### Download Seeds at Maximum Quality
```bash
# Always grab highest res — 4K references enable 2×, 3×, 4× test pairs
yt-dlp -a urls.txt --concurrent-fragments 4 \
  -f "bestvideo[height>=2160]+bestaudio/bestvideo[height>=1080]+bestaudio/best" \
  --merge-output-format mp4 \
  -o "seeds/%(id)s.%(ext)s"
```

### Trim Reference Clips (Stream Copy — Instant)
```bash
# -c copy = no decode/encode, just packet slicing. Essentially free.
for f in seeds/*.mp4; do
  base=$(basename "$f" .mp4)
  for offset in 10 60 120; do
    ffmpeg -loglevel warning -ss $offset -i "$f" -t 30 \
      -c copy -avoid_negative_ts make_start \
      -y "refs/${base}_t${offset}.mp4"
  done
done
```
**Speed**: ~10,000+ fps equivalent. Not a bottleneck.

---

## Phase 2: Downscale Matrix (GPU-Accelerated)

### Core NVENC Pattern for Downscaling

```bash
# CPU decode → CPU scale filter → NVENC encode
# This is the optimal pattern for A6000/A100
ffmpeg -i "refs/video_ref.mp4" \
  -vf "scale=iw/2:ih/2:flags=lanczos" \
  -c:v h264_nvenc -preset p4 -rc constqp -qp 10 \
  -c:a copy \
  "inputs/video_2x_lanczos.mp4"
```

### NVENC Preset Selection for Test Data

| Preset | Speed | Quality | Recommendation |
|--------|-------|---------|----------------|
| `p1` | Fastest | Lowest | Bulk generation, quality doesn't matter |
| `p2` | Very fast | Low | High-throughput batch |
| **`p4`** | **Balanced** | **Good** | **Default — good enough for test inputs** |
| `p5` | Slow | High | When testing "clean source" variants |
| `p7` | Slowest | Highest | Near-lossless reference-grade encoding |

### NVENC Rate Control for Upscaling Inputs

```bash
# constqp — fastest, deterministic quality (RECOMMENDED)
-rc constqp -qp 10    # near-lossless
-rc constqp -qp 20    # good quality
-rc constqp -qp 30    # medium (visible artefacts)
-rc constqp -qp 40    # poor (heavy artefacts — tests artefact amplification)

# VBR with CQ target — when testing specific bitrate scenarios
-rc vbr -cq 23 -b:v 0
```

> [!TIP]
> Always use `-rc constqp` for test data generation. It's the fastest NVENC rate control mode and provides consistent quality across content types.

---

## Phase 2A: Scale Factor Variants (NVENC)

```bash
generate_scale_variants() {
    local ref="$1"
    local gpu="$2"
    local base=$(basename "$ref" .mp4)

    # 2× downscale
    ffmpeg -loglevel warning -i "$ref" \
      -vf "scale=iw/2:ih/2:flags=lanczos" \
      -c:v h264_nvenc -gpu "$gpu" -preset p4 -rc constqp -qp 10 \
      -y "inputs/${base}_2x.mp4"

    # 3× downscale
    ffmpeg -loglevel warning -i "$ref" \
      -vf "scale=iw/3:ih/3:flags=lanczos" \
      -c:v h264_nvenc -gpu "$gpu" -preset p4 -rc constqp -qp 10 \
      -y "inputs/${base}_3x.mp4"

    # 4× downscale
    ffmpeg -loglevel warning -i "$ref" \
      -vf "scale=iw/4:ih/4:flags=lanczos" \
      -c:v h264_nvenc -gpu "$gpu" -preset p4 -rc constqp -qp 10 \
      -y "inputs/${base}_4x.mp4"
}
```

## Phase 2B: Downscale Method Variants (NVENC)

```bash
generate_method_variants() {
    local ref="$1"
    local gpu="$2"
    local base=$(basename "$ref" .mp4)

    for method in lanczos bicubic bilinear area neighbor; do
        ffmpeg -loglevel warning -i "$ref" \
          -vf "scale=iw/2:ih/2:flags=${method}" \
          -c:v h264_nvenc -gpu "$gpu" -preset p4 -rc constqp -qp 10 \
          -y "inputs/${base}_2x_${method}.mp4"
    done

    # Gaussian blur + subsample (optical blur simulation)
    ffmpeg -loglevel warning -i "$ref" \
      -vf "gblur=sigma=1.5,scale=iw/2:ih/2:flags=bilinear" \
      -c:v h264_nvenc -gpu "$gpu" -preset p4 -rc constqp -qp 10 \
      -y "inputs/${base}_2x_gaussian.mp4"
}
```

## Phase 2C: Source Quality Degradation Variants (NVENC)

These simulate real-world compressed low-res inputs:

```bash
generate_quality_variants() {
    local ref="$1"
    local gpu="$2"
    local base=$(basename "$ref" .mp4)

    for qp in 5 15 25 35 45; do
        ffmpeg -loglevel warning -i "$ref" \
          -vf "scale=iw/2:ih/2:flags=lanczos" \
          -c:v h264_nvenc -gpu "$gpu" -preset p4 -rc constqp -qp "$qp" \
          -y "inputs/${base}_2x_qp${qp}.mp4"
    done
}
```

## Phase 2D: Noise Injection Variants (CPU Filter → NVENC)

```bash
generate_noise_variants() {
    local ref="$1"
    local gpu="$2"
    local base=$(basename "$ref" .mp4)

    # Light noise + downscale
    ffmpeg -loglevel warning -i "$ref" \
      -vf "noise=alls=10:allf=t,scale=iw/2:ih/2:flags=lanczos" \
      -c:v h264_nvenc -gpu "$gpu" -preset p4 -rc constqp -qp 10 \
      -y "inputs/${base}_2x_noise10.mp4"

    # Heavy noise + downscale
    ffmpeg -loglevel warning -i "$ref" \
      -vf "noise=alls=30:allf=t+u,scale=iw/2:ih/2:flags=lanczos" \
      -c:v h264_nvenc -gpu "$gpu" -preset p4 -rc constqp -qp 10 \
      -y "inputs/${base}_2x_noise30.mp4"
}
```

> [!NOTE]
> Noise filters run on CPU. For heavy noise (`alls=50`), the CPU filter becomes the bottleneck, not NVENC. See the throughput section below for per-filter speed data.

## Phase 2E: Combined Filter Chains (Single Pass — Critical Optimisation)

Instead of running separate FFmpeg calls for scale + noise + FPS change, **chain everything into one pass**:

```bash
# BAD: 3 separate passes (3× decode + 3× encode)
ffmpeg -i ref.mp4 -vf "scale=iw/2:ih/2:flags=lanczos" -c:v h264_nvenc ... temp1.mp4
ffmpeg -i temp1.mp4 -vf "noise=alls=20:allf=t" -c:v h264_nvenc ... temp2.mp4
ffmpeg -i temp2.mp4 -vf "fps=24" -c:v h264_nvenc ... output.mp4

# GOOD: single pass (1× decode + 1× encode)
ffmpeg -i ref.mp4 \
  -vf "scale=iw/2:ih/2:flags=lanczos,noise=alls=20:allf=t,fps=24" \
  -c:v h264_nvenc -preset p4 -rc constqp -qp 10 \
  output.mp4
```

**Single pass is 2-3× faster** because you avoid redundant decode/encode cycles.

---

## Phase 2F: GPU-Side Scaling with scale_npp

If your FFmpeg build includes `--enable-libnpp` (NVIDIA Performance Primitives), scaling can run on the GPU instead of CPU:

```bash
# Upload CPU-decoded frames to GPU, scale on GPU via NPP, encode on GPU via NVENC
ffmpeg -i ref.mp4 \
  -vf "hwupload_cuda,scale_npp=iw/2:ih/2:interp_algo=lanczos" \
  -c:v h264_nvenc -preset p4 -rc constqp -qp 10 \
  output.mp4
```

### When scale_npp Helps vs Hurts

| Scenario | scale_npp Benefit |
|----------|------------------|
| Scale-only, no other filters | ✅ **15-30% faster** — avoids CPU→GPU frame copy |
| Scale + CPU filter (noise, blur) | ❌ Hurts — must download back to CPU for filter, then upload again |
| 4K→540p (large reduction) | ✅ Good — scaling is compute-heavy |
| 1080p→540p (small reduction) | ⚠️ Marginal — hwupload overhead may negate benefit |

> [!IMPORTANT]
> Only use `scale_npp` when scaling is the **only** filter operation. If you're chaining CPU filters (noise, blur, denoise), keep scaling on CPU to avoid redundant GPU↔CPU transfers.

### Checking scale_npp Availability
```bash
ffmpeg -filters 2>/dev/null | grep scale_npp
# If present: "scale_npp   V->V   NVIDIA NPP video scaling"
# If absent: compile FFmpeg with --enable-libnpp --enable-cuda-nvcc
```

---

## Phase 3: Codec Variants (Hybrid CPU + GPU)

### NVENC Codecs (GPU — Fast)

```bash
# H.264 via NVENC
ffmpeg -i ref.mp4 -vf "scale=iw/2:ih/2:flags=lanczos" \
  -c:v h264_nvenc -preset p4 -rc constqp -qp 18 out_h264.mp4

# H.264 profiles (different artefact signatures)
ffmpeg -i ref.mp4 -vf "scale=iw/2:ih/2:flags=lanczos" \
  -c:v h264_nvenc -profile:v baseline -preset p4 -rc constqp -qp 25 out_h264_baseline.mp4

ffmpeg -i ref.mp4 -vf "scale=iw/2:ih/2:flags=lanczos" \
  -c:v h264_nvenc -profile:v high -preset p4 -rc constqp -qp 25 out_h264_high.mp4

# H.265 via NVENC (smoother artefacts at same bitrate)
ffmpeg -i ref.mp4 -vf "scale=iw/2:ih/2:flags=lanczos" \
  -c:v hevc_nvenc -preset p4 -rc constqp -qp 22 out_h265.mp4

# H.265 10-bit via NVENC
ffmpeg -i ref.mp4 -vf "scale=iw/2:ih/2:flags=lanczos" \
  -c:v hevc_nvenc -profile:v main10 -preset p4 -rc constqp -qp 22 \
  -pix_fmt p010le out_h265_10bit.mp4
```

### CPU-Only Codecs (Run In Parallel with NVENC)

Since NVENC is a dedicated ASIC separate from CUDA cores and CPU, CPU-only encoders can run **simultaneously** without competing for the same hardware:

```bash
# VP9 (CPU only — use alongside NVENC jobs)
ffmpeg -i ref.mp4 -vf "scale=iw/2:ih/2:flags=lanczos" \
  -c:v libvpx-vp9 -crf 30 -b:v 0 -row-mt 1 -threads 4 out_vp9.webm

# AV1 via SVT-AV1 (fastest AV1 encoder — CPU only)
ffmpeg -i ref.mp4 -vf "scale=iw/2:ih/2:flags=lanczos" \
  -c:v libsvtav1 -crf 30 -preset 8 out_av1.mp4

# MPEG-2 (legacy artefact signatures — CPU only, very fast)
ffmpeg -i ref.mp4 -vf "scale=iw/2:ih/2:flags=lanczos" \
  -c:v mpeg2video -b:v 3M out_mpeg2.mpg

# MJPEG (per-frame artefacts, no temporal prediction — CPU only)
ffmpeg -i ref.mp4 -vf "scale=iw/2:ih/2:flags=lanczos" \
  -c:v mjpeg -q:v 5 out_mjpeg.mp4
```

### Codec × Quality Matrix for Upscaling Inputs

Different codecs need different parameter values to produce equivalent perceptual quality:

| Perceived Quality | H.264 NVENC (qp) | H.265 NVENC (qp) | VP9 (crf) | AV1 SVT (crf) | MPEG-2 (bitrate) |
|------------------|-------------------|-------------------|-----------|----------------|-------------------|
| Near-lossless | 5 | 8 | 10 | 10 | 15M |
| Good | 15 | 18 | 25 | 25 | 8M |
| Medium | 25 | 28 | 35 | 35 | 3M |
| Poor | 35 | 38 | 45 | 45 | 1M |
| Terrible | 45 | 48 | 55 | 55 | 500K |

---

## Parallelism Strategy

### Key Insight: Three Independent Hardware Resources

```
┌─────────────────────────────────────────────────┐
│                   System                         │
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │   CPU    │  │  NVENC   │  │   Disk I/O    │  │
│  │ Cores    │  │  ASIC    │  │               │  │
│  │          │  │          │  │               │  │
│  │ Decode   │  │ Encode   │  │ Read/Write    │  │
│  │ Filter   │  │ H.264    │  │ Source/Output │  │
│  │ VP9 enc  │  │ H.265    │  │               │  │
│  │ AV1 enc  │  │          │  │               │  │
│  └──────────┘  └──────────┘  └───────────────┘  │
│                                                  │
│  All three can run simultaneously!               │
└─────────────────────────────────────────────────┘
```

### Optimal Process Count

```bash
CPU_CORES=$(nproc)

# NVENC jobs: each needs ~2 CPU threads for decode + filter
NVENC_JOBS=$((CPU_CORES / 2))
MAX_NVENC=12              # NVENC throughput ceiling (~12 concurrent 1080p streams)
NVENC_PARALLEL=$(( NVENC_JOBS < MAX_NVENC ? NVENC_JOBS : MAX_NVENC ))

# CPU-only codec jobs: use remaining CPU headroom
# These run alongside NVENC jobs since NVENC doesn't use CPU
CPU_ONLY_JOBS=$((CPU_CORES / 4))  # VP9/AV1 are hungry — 4 threads each

echo "NVENC parallel: $NVENC_PARALLEL"
echo "CPU codec parallel: $CPU_ONLY_JOBS"
```

### Recommended Parallelism by Machine

| Machine | CPU Cores | NVENC Jobs | CPU-Only Jobs | Total Concurrent |
|---------|-----------|------------|---------------|-----------------|
| A100 (DGX) | 64 | 12 | 8 | 20 |
| A100 (cloud 8-core) | 8 | 4 | 2 | 6 |
| A6000 (workstation) | 32 | 10 | 4 | 14 |
| A40 (cloud 16-core) | 16 | 8 | 3 | 11 |

---

## Complete Pipeline Script

```bash
#!/bin/bash
# generate_upscale_dataset.sh
# GPU-optimised pipeline for A6000/A100 (NVENC, no NVDEC)

set -euo pipefail

SEEDS_DIR="seeds"
REFS_DIR="refs"
INPUTS_DIR="inputs"
GPU_ID=0
NVENC_PRESET="p4"

mkdir -p "$REFS_DIR" "$INPUTS_DIR"

# ─────────────────────────────────────────────
# Phase 1: Trim references (stream copy — instant)
# ─────────────────────────────────────────────
echo "=== Phase 1: Creating reference clips ==="
for f in "$SEEDS_DIR"/*.mp4; do
    base=$(basename "$f" .mp4)
    ffmpeg -loglevel warning -ss 10 -i "$f" -t 30 \
      -c copy -avoid_negative_ts make_start \
      -y "${REFS_DIR}/${base}_ref.mp4"
done

# ─────────────────────────────────────────────
# Phase 2: NVENC variants (scale × method × quality)
# ─────────────────────────────────────────────
generate_nvenc_variants() {
    local ref="$1"
    local gpu="$2"
    local preset="$3"
    local base=$(basename "$ref" .mp4)

    # ── Scale factor variants (lanczos, near-lossless) ──
    for scale in 2 3 4; do
        ffmpeg -loglevel warning -i "$ref" \
          -vf "scale=iw/${scale}:ih/${scale}:flags=lanczos" \
          -c:v h264_nvenc -gpu "$gpu" -preset "$preset" -rc constqp -qp 10 \
          -c:a copy -y "${INPUTS_DIR}/${base}_${scale}x_lanczos_qp10.mp4"
    done

    # ── Downscale method variants (2× at near-lossless) ──
    for method in lanczos bicubic bilinear area neighbor; do
        ffmpeg -loglevel warning -i "$ref" \
          -vf "scale=iw/2:ih/2:flags=${method}" \
          -c:v h264_nvenc -gpu "$gpu" -preset "$preset" -rc constqp -qp 10 \
          -c:a copy -y "${INPUTS_DIR}/${base}_2x_${method}_qp10.mp4"
    done

    # ── Quality degradation variants (2× lanczos, varying qp) ──
    for qp in 5 15 25 35 45; do
        ffmpeg -loglevel warning -i "$ref" \
          -vf "scale=iw/2:ih/2:flags=lanczos" \
          -c:v h264_nvenc -gpu "$gpu" -preset "$preset" -rc constqp -qp "$qp" \
          -c:a copy -y "${INPUTS_DIR}/${base}_2x_lanczos_qp${qp}.mp4"
    done

    # ── H.265 variant via NVENC ──
    ffmpeg -loglevel warning -i "$ref" \
      -vf "scale=iw/2:ih/2:flags=lanczos" \
      -c:v hevc_nvenc -gpu "$gpu" -preset "$preset" -rc constqp -qp 20 \
      -c:a copy -y "${INPUTS_DIR}/${base}_2x_h265_qp20.mp4"

    # ── H.265 10-bit variant via NVENC ──
    ffmpeg -loglevel warning -i "$ref" \
      -vf "scale=iw/2:ih/2:flags=lanczos" \
      -c:v hevc_nvenc -gpu "$gpu" -profile:v main10 -preset "$preset" \
      -rc constqp -qp 20 -pix_fmt p010le \
      -c:a copy -y "${INPUTS_DIR}/${base}_2x_h265_10bit_qp20.mp4"

    # ── Noise + downscale (combined filter chain) ──
    for noise in 10 30; do
        ffmpeg -loglevel warning -i "$ref" \
          -vf "noise=alls=${noise}:allf=t,scale=iw/2:ih/2:flags=lanczos" \
          -c:v h264_nvenc -gpu "$gpu" -preset "$preset" -rc constqp -qp 10 \
          -c:a copy -y "${INPUTS_DIR}/${base}_2x_noise${noise}_qp10.mp4"
    done

    # ── FPS variants ──
    for fps in 24 60; do
        ffmpeg -loglevel warning -i "$ref" \
          -vf "fps=${fps},scale=iw/2:ih/2:flags=lanczos" \
          -c:v h264_nvenc -gpu "$gpu" -preset "$preset" -rc constqp -qp 10 \
          -c:a copy -y "${INPUTS_DIR}/${base}_2x_${fps}fps_qp10.mp4"
    done

    # ── Blur + downscale (tests blurry source upscaling) ──
    ffmpeg -loglevel warning -i "$ref" \
      -vf "gblur=sigma=2,scale=iw/2:ih/2:flags=lanczos" \
      -c:v h264_nvenc -gpu "$gpu" -preset "$preset" -rc constqp -qp 10 \
      -c:a copy -y "${INPUTS_DIR}/${base}_2x_blurred_qp10.mp4"

    echo "NVENC done: $base"
}

export -f generate_nvenc_variants
export INPUTS_DIR

echo "=== Phase 2: NVENC variant generation ==="
find "$REFS_DIR" -name "*.mp4" | \
    parallel -j 8 generate_nvenc_variants {} "$GPU_ID" "$NVENC_PRESET"

# ─────────────────────────────────────────────
# Phase 3: CPU-only codec variants (VP9, AV1, MPEG-2)
# Run AFTER Phase 2, or simultaneously if enough CPU headroom
# ─────────────────────────────────────────────
generate_cpu_codec_variants() {
    local ref="$1"
    local base=$(basename "$ref" .mp4)

    # VP9
    ffmpeg -loglevel warning -i "$ref" \
      -vf "scale=iw/2:ih/2:flags=lanczos" \
      -c:v libvpx-vp9 -crf 30 -b:v 0 -row-mt 1 -threads 4 \
      -y "${INPUTS_DIR}/${base}_2x_vp9_crf30.webm"

    # AV1 (SVT — fast preset)
    ffmpeg -loglevel warning -i "$ref" \
      -vf "scale=iw/2:ih/2:flags=lanczos" \
      -c:v libsvtav1 -crf 30 -preset 8 \
      -y "${INPUTS_DIR}/${base}_2x_av1_crf30.mp4"

    # MPEG-2 (legacy artefacts)
    ffmpeg -loglevel warning -i "$ref" \
      -vf "scale=iw/2:ih/2:flags=lanczos" \
      -c:v mpeg2video -b:v 3M \
      -y "${INPUTS_DIR}/${base}_2x_mpeg2_3M.mpg"

    echo "CPU codecs done: $base"
}

export -f generate_cpu_codec_variants

echo "=== Phase 3: CPU codec variants (VP9, AV1, MPEG-2) ==="
find "$REFS_DIR" -name "*.mp4" | \
    parallel -j 4 generate_cpu_codec_variants {}

# ─────────────────────────────────────────────
# Phase 4: Synthetic test patterns (fast, few files)
# ─────────────────────────────────────────────
echo "=== Phase 4: Synthetic test patterns ==="

# SMPTE test bars
ffmpeg -loglevel warning -f lavfi -i "smptebars=s=3840x2160:r=30:d=10" \
  -c:v h264_nvenc -gpu "$GPU_ID" -preset p4 -rc constqp -qp 5 \
  -y "${REFS_DIR}/smpte_bars_4k.mp4"
ffmpeg -loglevel warning -i "${REFS_DIR}/smpte_bars_4k.mp4" \
  -vf "scale=960:540:flags=lanczos" \
  -c:v h264_nvenc -gpu "$GPU_ID" -preset p4 -rc constqp -qp 10 \
  -y "${INPUTS_DIR}/smpte_bars_4x.mp4"

# Resolution test chart
ffmpeg -loglevel warning -f lavfi -i "testsrc2=s=3840x2160:r=30:d=10" \
  -c:v h264_nvenc -gpu "$GPU_ID" -preset p4 -rc constqp -qp 5 \
  -y "${REFS_DIR}/testsrc_4k.mp4"
ffmpeg -loglevel warning -i "${REFS_DIR}/testsrc_4k.mp4" \
  -vf "scale=960:540:flags=lanczos" \
  -c:v h264_nvenc -gpu "$GPU_ID" -preset p4 -rc constqp -qp 10 \
  -y "${INPUTS_DIR}/testsrc_4x.mp4"

# Random noise (should NOT improve after upscaling)
ffmpeg -loglevel warning -f lavfi \
  -i "nullsrc=s=960x540:r=30:d=5,geq=random(1)*255:128:128" \
  -c:v h264_nvenc -gpu "$GPU_ID" -preset p4 -rc constqp -qp 10 \
  -y "${INPUTS_DIR}/random_noise.mp4"

# Pure black (maximum compressibility edge case)
ffmpeg -loglevel warning -f lavfi -i "color=c=black:s=3840x2160:r=30:d=10" \
  -c:v h264_nvenc -gpu "$GPU_ID" -preset p4 -rc constqp -qp 5 \
  -y "${REFS_DIR}/black_4k.mp4"
ffmpeg -loglevel warning -i "${REFS_DIR}/black_4k.mp4" \
  -vf "scale=960:540:flags=lanczos" \
  -c:v h264_nvenc -gpu "$GPU_ID" -preset p4 -rc constqp -qp 10 \
  -y "${INPUTS_DIR}/black_4x.mp4"

echo "=== Phase 4 complete ==="

# ─────────────────────────────────────────────
# Phase 5: Generate manifest
# ─────────────────────────────────────────────
echo "=== Phase 5: Generating manifest ==="
python3 -c "
import os, json, subprocess

manifest = []
for f in sorted(os.listdir('inputs')):
    path = os.path.join('inputs', f)
    # Extract reference name (strip transformation suffixes)
    parts = f.rsplit('_', 3)
    ref_base = parts[0] if len(parts) > 1 else f.replace('.mp4','')

    # Get video info
    probe = subprocess.run(
        ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', path],
        capture_output=True, text=True
    )
    try:
        info = json.loads(probe.stdout)
        vs = next(s for s in info['streams'] if s['codec_type'] == 'video')
        w, h = int(vs['width']), int(vs['height'])
        codec = vs['codec_name']
    except:
        w, h, codec = 0, 0, 'unknown'

    manifest.append({
        'input_path': path,
        'reference_pattern': f'refs/{ref_base}*',
        'width': w, 'height': h,
        'codec': codec,
        'filename': f
    })

with open('manifest.json', 'w') as mf:
    json.dump(manifest, mf, indent=2)
print(f'Manifest: {len(manifest)} entries')
"
echo "=== Done! ==="
```

---

## Multi-GPU Support

For machines with multiple GPUs (DGX, multi-GPU cloud):

```bash
NUM_GPUS=$(nvidia-smi -L | wc -l)

# Round-robin GPU assignment
find "$REFS_DIR" -name "*.mp4" | \
    parallel -j $((8 * NUM_GPUS)) \
    'GPU_ID=$(( {#} % '"$NUM_GPUS"' )); generate_nvenc_variants {} $GPU_ID '"$NVENC_PRESET"
```

### Multi-GPU Throughput

| Config | NVENC Streams | Expected Total FPS (1080p) |
|--------|--------------|---------------------------|
| 1× A100 | 8-12 | ~2,000-4,000 |
| 2× A100 | 16-24 | ~4,000-8,000 |
| 4× A100 (DGX) | 32-48 | ~8,000-16,000 |
| 8× A100 (DGX A100) | 64-96 | ~16,000-30,000 |

---

## Throughput Benchmarks

### Per-Process Speed (Single FFmpeg, 1080p Output, A6000)

| Pipeline | FPS | Bottleneck |
|----------|-----|-----------|
| Stream copy (`-c copy`) | ∞ (I/O) | Disk |
| Decode → NVENC (no filter) | ~400-600 | CPU decode |
| Scale 4K→1080p + NVENC | ~150-250 | CPU scale filter |
| Scale 4K→540p + NVENC | ~200-350 | CPU scale filter |
| Noise + Scale + NVENC | ~80-150 | CPU noise filter |
| Blur + Scale + NVENC | ~100-200 | CPU blur filter |
| hqdn3d + Scale + NVENC | ~30-60 | CPU denoise |
| nlmeans + Scale + NVENC | ~3-8 | CPU nlmeans (avoid!) |
| `scale_npp` only + NVENC | ~300-500 | NVENC |

### Aggregate Throughput (8 Parallel Jobs, A6000)

| Pipeline | Total FPS | Time for 80 refs × 30s |
|----------|-----------|----------------------|
| Scale-only variants (NVENC) | ~1,500 | ~3 min |
| Full NVENC matrix (all transforms) | ~600 | ~10 min |
| CPU codecs (VP9 + AV1 + MPEG-2) | ~40 | ~90 min |
| **Everything combined** | — | **~2 hours** |

### Estimated Total Pipeline Time

| Phase | A100 (64-core) | A6000 (32-core) | A40 (16-core) |
|-------|---------------|-----------------|---------------|
| Download 40 seeds | ~20 min | ~20 min | ~20 min |
| Trim references | ~1 min | ~1 min | ~1 min |
| NVENC variants (H.264 + H.265) | ~15 min | ~30 min | ~60 min |
| CPU codecs (VP9 + AV1) | ~45 min | ~90 min | ~3 hours |
| Synthetic patterns | ~2 min | ~2 min | ~2 min |
| **Total** | **~1.5 hours** | **~2.5 hours** | **~4.5 hours** |

---

## Avoiding Common Pitfalls

### ❌ Don't: Use `-hwaccel cuda` on A6000/A100
```bash
# WRONG — will fail, no NVDEC hardware
ffmpeg -hwaccel cuda -i input.mp4 -c:v h264_nvenc output.mp4
```

### ✅ Do: CPU decode, GPU encode only
```bash
# CORRECT — no hwaccel flag
ffmpeg -i input.mp4 -c:v h264_nvenc output.mp4
```

### ❌ Don't: Run multiple passes for combined transforms
```bash
# WRONG — 3× decode/encode overhead
ffmpeg -i ref.mp4 -vf "scale=iw/2:ih/2" out1.mp4
ffmpeg -i out1.mp4 -vf "noise=alls=20" out2.mp4
ffmpeg -i out2.mp4 -c:v h264_nvenc final.mp4
```

### ✅ Do: Chain filters in a single pass
```bash
# CORRECT — 1× decode, 1× encode
ffmpeg -i ref.mp4 -vf "scale=iw/2:ih/2:flags=lanczos,noise=alls=20:allf=t" \
  -c:v h264_nvenc -preset p4 -rc constqp -qp 10 final.mp4
```

### ❌ Don't: Use `nlmeans` for bulk test data generation
```bash
# WRONG — 3-8 fps, will take hours per video
ffmpeg -i ref.mp4 -vf "nlmeans=s=6:p=7:r=15,scale=iw/2:ih/2" ...
```

### ✅ Do: Use `hqdn3d` as a fast denoise substitute
```bash
# CORRECT — 30-60 fps, good enough for test data
ffmpeg -i ref.mp4 -vf "hqdn3d=4:4:6:6,scale=iw/2:ih/2:flags=lanczos" \
  -c:v h264_nvenc -preset p4 -rc constqp -qp 10 out.mp4
```

### ❌ Don't: Use libaom-av1 for test data
```bash
# WRONG — 0.5-2 fps
ffmpeg -i ref.mp4 -c:v libaom-av1 -crf 30 -cpu-used 4 out.mp4
```

### ✅ Do: Use libsvtav1 with fast preset
```bash
# CORRECT — 20-50 fps
ffmpeg -i ref.mp4 -c:v libsvtav1 -crf 30 -preset 10 out.mp4
```

---

## Checklist

- [ ] Verify NVENC: `ffmpeg -i test.mp4 -c:v h264_nvenc -preset p4 -f null -`
- [ ] Verify hevc_nvenc: `ffmpeg -i test.mp4 -c:v hevc_nvenc -preset p4 -f null -`
- [ ] Confirm NO `-hwaccel` flags in any command
- [ ] Check scale_npp availability: `ffmpeg -filters | grep scale_npp`
- [ ] Download ~40 seed videos at max resolution via yt-dlp
- [ ] Trim reference clips (`-c copy`)
- [ ] Run NVENC variant generation (`parallel -j 8`)
- [ ] Run CPU codec variants (`parallel -j 4`) simultaneously
- [ ] Generate synthetic edge cases
- [ ] Create manifest.json
- [ ] Validate: `ffprobe` spot-check 10 random (input, reference) pairs
- [ ] Verify paired alignment: check that inputs have correct resolution relative to refs
