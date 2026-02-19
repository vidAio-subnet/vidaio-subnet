# Video Compressibility: Factors, Test Dataset Design & Generation Pipeline

## Part 1 — Factors That Determine Video Compressibility

Video compression exploits redundancies. The more redundancy a video contains, the more compressible it is. Every factor below directly controls one or more types of redundancy.

---

### 1. Spatial Complexity (Intra-Frame)

| Factor | Low Compressibility | High Compressibility |
|--------|---------------------|----------------------|
| **Texture detail** | Grass, foliage, fabric weave, hair | Flat walls, sky, solid-colour surfaces |
| **Edge density** | Dense cityscapes, text overlays | Soft gradients, out-of-focus backgrounds |
| **Noise / film grain** | High ISO footage, analog film scans | Clean studio lighting, denoised footage |
| **Colour variance** | Many distinct hues per frame | Monochromatic / limited palette |
| **Spatial frequency** | High-frequency patterns (e.g. brick walls) | Low-frequency smooth regions |

> [!NOTE]
> Spatial complexity is the single biggest predictor for I-frame (keyframe) size, and sets a floor on achievable bitrate.

---

### 2. Temporal Complexity (Inter-Frame)

| Factor | Low Compressibility | High Compressibility |
|--------|---------------------|----------------------|
| **Motion magnitude** | Fast sports, shaky handheld camera | Tripod-locked, slow-moving subjects |
| **Motion irregularity** | Chaotic particle effects, rain, confetti | Linear pans, predictable object paths |
| **Scene cuts / transitions** | Rapid montage, music video edits | Long uninterrupted takes |
| **Object occlusion/reveal** | Objects constantly entering/leaving frame | Static scene with minor movement |
| **Flicker / strobe** | Concert lighting, welding, lightning | Consistent illumination |

> [!IMPORTANT]
> Temporal redundancy is what P-frames and B-frames exploit. A video with zero motion compresses to nearly nothing; one with chaotic motion per pixel essentially defeats inter-prediction.

---

### 3. Resolution & Frame Dimensions

- Higher resolution = more pixels per frame = larger data before compression
- But also = larger block sizes available (e.g., 64×64 CTUs in HEVC) = potentially better compression ratios on smooth content
- Non-standard resolutions can harm encoder alignment (padding, sub-optimal tiling)

| Resolution | Typical Use |
|------------|-------------|
| 360p (640×360) | Low-bandwidth mobile |
| 480p (854×480) | SD web content |
| 720p (1280×720) | HD streaming baseline |
| 1080p (1920×1080) | Full HD — most common benchmark |
| 1440p (2560×1440) | 2K gaming/streaming |
| 4K (3840×2160) | Ultra HD — stress test |

---

### 4. Frame Rate (FPS)

- Higher FPS → more frames → more total data, BUT each consecutive frame is more similar (smaller temporal delta)
- Net effect: higher FPS generally *increases* bitrate but *decreases* per-frame delta cost
- Variable frame rate (VFR) can confuse some encoders and cause timestamp issues

| Frame Rate | Character |
|------------|-----------|
| 24 fps | Cinema (larger inter-frame differences) |
| 30 fps | Standard broadcast |
| 60 fps | Gaming / sports (good temporal redundancy) |
| 120+ fps | Slow-motion source / high-speed captures |

---

### 5. Bit Depth & Colour Space

| Property | Values | Impact |
|----------|--------|--------|
| **Bit depth** | 8-bit vs 10-bit vs 12-bit | Higher = more precision = larger raw data, but also smoother gradients = fewer banding artifacts to encode |
| **Chroma subsampling** | 4:4:4, 4:2:2, 4:2:0 | 4:2:0 discards 75% of chroma → very compressible; 4:4:4 retains all colour info |
| **Colour space / gamut** | BT.709, BT.2020, HDR (PQ/HLG) | Wider gamut + HDR metadata adds complexity & data |

---

### 6. Content Type / Semantic Category

Different content genres have fundamentally different compression profiles:

| Category | Spatial | Temporal | Overall |
|----------|---------|----------|---------|
| **Screencast / slides** | Low (flat) | Very low (static) | Extremely compressible |
| **Talking head** | Medium | Low | Highly compressible |
| **Nature / landscape** | High (texture) | Low–Medium | Medium |
| **Sports / action** | Medium | Very high | Hard to compress |
| **Gaming** | Variable | High (fast movement + particle FX) | Hard to compress |
| **Music video / montage** | High | Very high (rapid cuts) | Very hard |
| **Noise / grain-heavy** | Very high | Medium–High | Extremely hard |
| **Animation / cartoon** | Low (flat fills) | Medium | Very compressible |
| **Underwater / smoke** | Medium (turbulence) | High (irregular flow) | Hard |

---

### 7. Encoding Parameters (Pre-existing Compression)

If the source is already compressed, these matter:

| Parameter | Effect |
|-----------|--------|
| **Existing bitrate** | Already-compressed video has fewer bits to squeeze; re-encoding a 500 kbps source won't gain much |
| **GOP structure** | Long GOPs (lots of B-frames) → already exploited temporal redundancy |
| **Quantization level** | Heavily quantized source has blocking artefacts that are *hard* to re-compress cleanly |
| **Codec generation** | H.264 source → re-encoding to H.265 can still gain 30-50%; H.265 → AV1 gains ~20-30% |

---

### 8. Audio Track Properties

Often overlooked, but relevant for total file compression:

- Codec (AAC, Opus, FLAC)
- Bitrate (64 kbps vs 320 kbps)
- Channels (mono, stereo, 5.1 surround)
- Silence ratio

---

### 9. Duration

- Directly multiplies all other factors
- Longer videos amortise keyframe overhead better
- Short clips (< 5s) are disproportionately large per-second due to keyframe density

---

### 10. Container & Metadata Overhead

- Container format (MP4, MKV, WebM) adds headers, index tables
- Subtitle tracks, chapter markers, HDR metadata, ICC profiles
- Negligible for long videos, noticeable for very short clips

---

## How to Make Test Data Maximally Thorough

### Dimension Coverage Matrix

To be thorough, the test dataset should be a **factorial design** across the key dimensions:

```
Dimensions to vary:
├── Spatial complexity:     [low, medium, high]
├── Temporal complexity:    [low, medium, high]
├── Resolution:             [480p, 720p, 1080p, 4K]
├── Frame rate:             [24, 30, 60]
├── Bit depth:              [8-bit, 10-bit]
├── Chroma subsampling:     [4:2:0, 4:2:2]
├── Content type:           [screencast, talking_head, nature, sports, animation, grain_heavy]
├── Duration:               [5s, 30s, 2min, 10min]
├── Pre-compression level:  [uncompressed/lossless, light CRF, heavy CRF]
└── Codec (see Part 3):     [H.264, H.265, VP9, AV1]
```

A full factorial is 3×3×4×3×2×2×6×4×3×4 = **124,416 combinations** — obviously impractical.

### Practical Strategy: Latin Hypercube Sampling

Instead of full factorial, use a **stratified sampling** approach:

1. **Identify the 3 most important axes**: spatial complexity, temporal complexity, and resolution
2. **Create a core matrix** of 3×3×4 = 36 combinations for these
3. **For each cell**, vary secondary dimensions (FPS, bit depth, duration) across the 36 cells so that every secondary value appears roughly equally
4. **Target: 100–200 test videos** that cover the space well

### Edge Cases to Include

- **Pure black / pure white** video (theoretical max compression)
- **Pure random noise** per pixel per frame (theoretical min compression — incompressible)
- **Single static frame** looped for duration (tests temporal-only compression)
- **Extremely rapid scene cuts** (1 cut per second for 60 seconds)
- **Screen recording with scrolling text** (high spatial, moderate temporal)
- **Slow zoom on high-detail image** (tests inter-prediction on scale changes)
- **Variable frame rate (VFR)** source
- **Interlaced content** (legacy but important for robustness)
- **Videos with burned-in subtitles vs clean**
- **HDR content** (BT.2020 + PQ transfer)

---

## Part 2 — Creating the Dataset from YouTube Videos + FFmpeg

### Thinking Out Loud: The Pipeline

**The Problem**: We need 100–200+ diverse test videos spanning many compressibility profiles, but we're starting from a limited set of YouTube downloads.

**Key Insight**: A single YouTube video can be *transformed* via FFmpeg to produce many variants along different compressibility axes. So we need:
1. A **diverse seed set** of YouTube downloads (maybe 30–50 videos)
2. An **FFmpeg transformation pipeline** that multiplies each seed into variants

#### Step 1: Curate the Seed Set via yt-dlp

Target ~40 YouTube videos across content types:

| Category | Search Terms | Count |
|----------|-------------|-------|
| Static/simple | "security camera feed", "webcam timelapse", "ASMR desk setup" | 5 |
| Talking head | "podcast episode", "news anchor", "lecture recording" | 5 |
| Nature/landscape | "4K nature drone", "underwater coral reef", "aurora borealis" | 5 |
| Sports/action | "football highlights", "Formula 1 onboard", "skateboarding tricks" | 5 |
| Animation/cartoon | "anime scene", "Pixar short film", "motion graphics loop" | 5 |
| Gaming | "gameplay 4K 60fps", "Minecraft speedrun", "retro game capture" | 5 |
| Music video | "music video 4K", "concert footage", "EDM visualizer" | 3 |
| Grain/noise | "8mm film transfer", "VHS recording", "low light surveillance" | 3 |
| Screencast | "coding tutorial", "software demo", "Google Slides presentation" | 4 |

Download command pattern:
```bash
# Download best quality up to 4K, prefer mp4
yt-dlp -f "bestvideo[height<=2160]+bestaudio/best[height<=2160]" \
  --merge-output-format mp4 \
  -o "seeds/%(id)s_%(title).50s.%(ext)s" \
  "<URL>"
```

#### Step 2: FFmpeg Transformation Matrix

From each seed video, generate variants by manipulating compressibility factors:

##### A. Temporal Subclipping (Duration Variation)
```bash
# Extract a 30-second clip starting at a random offset
ffmpeg -ss $OFFSET -i input.mp4 -t 30 -c copy clip_30s.mp4

# Extract a 5-second clip (high keyframe density)
ffmpeg -ss $OFFSET -i input.mp4 -t 5 -c copy clip_5s.mp4
```

##### B. Resolution Scaling
```bash
# Scale to specific resolutions
ffmpeg -i input.mp4 -vf "scale=1920:1080:flags=lanczos" -c:v libx264 -crf 18 out_1080p.mp4
ffmpeg -i input.mp4 -vf "scale=1280:720:flags=lanczos"  -c:v libx264 -crf 18 out_720p.mp4
ffmpeg -i input.mp4 -vf "scale=854:480:flags=lanczos"   -c:v libx264 -crf 18 out_480p.mp4
```

##### C. Frame Rate Conversion
```bash
# Change to 24fps (larger inter-frame deltas)
ffmpeg -i input.mp4 -r 24 -c:v libx264 -crf 18 out_24fps.mp4

# Change to 60fps (frame interpolation — creates smoother content)
ffmpeg -i input.mp4 -r 60 -c:v libx264 -crf 18 out_60fps.mp4
```

##### D. Noise Injection (Increase Spatial Complexity)
```bash
# Add synthetic grain to make content harder to compress
ffmpeg -i input.mp4 -vf "noise=alls=20:allf=t" -c:v libx264 -crf 18 out_noisy.mp4

# Heavy noise (simulates very noisy camera)
ffmpeg -i input.mp4 -vf "noise=alls=50:allf=t+u" -c:v libx264 -crf 18 out_very_noisy.mp4
```

##### E. Denoising (Decrease Spatial Complexity)
```bash
# Denoise to make content easier to compress
ffmpeg -i input.mp4 -vf "nlmeans=s=6:p=7:r=15" -c:v libx264 -crf 18 out_denoised.mp4
```

##### F. Blur / Sharpen (Spatial Frequency Manipulation)
```bash
# Gaussian blur (reduces high-frequency detail → more compressible)
ffmpeg -i input.mp4 -vf "gblur=sigma=2" -c:v libx264 -crf 18 out_blurred.mp4

# Sharpen (increases high-frequency content → less compressible)
ffmpeg -i input.mp4 -vf "unsharp=5:5:1.5" -c:v libx264 -crf 18 out_sharp.mp4
```

##### G. Speed Changes (Temporal Density Manipulation)
```bash
# 2x speed (doubles temporal change rate → harder to compress temporally)
ffmpeg -i input.mp4 -vf "setpts=0.5*PTS" -an -c:v libx264 -crf 18 out_2x.mp4

# 0.5x speed (halves temporal change → easier temporal compression)
ffmpeg -i input.mp4 -vf "setpts=2.0*PTS" -an -c:v libx264 -crf 18 out_half.mp4
```

##### H. Bit Depth Conversion
```bash
# Convert to 10-bit (using x265 which natively supports it)
ffmpeg -i input.mp4 -c:v libx265 -crf 18 -pix_fmt yuv420p10le out_10bit.mp4
```

##### I. Pre-compression Level (Quality Degradation)
```bash
# Light compression (nearly lossless)
ffmpeg -i input.mp4 -c:v libx264 -crf 8 out_crf8.mp4

# Medium compression
ffmpeg -i input.mp4 -c:v libx264 -crf 23 out_crf23.mp4

# Heavy compression (visible artefacts — tests re-compression)
ffmpeg -i input.mp4 -c:v libx264 -crf 35 out_crf35.mp4
```

##### J. Synthetic Edge Cases
```bash
# Pure black (maximally compressible)
ffmpeg -f lavfi -i "color=c=black:s=1920x1080:r=30:d=30" -c:v libx264 -crf 18 black_30s.mp4

# Random noise (minimally compressible)
ffmpeg -f lavfi -i "nullsrc=s=1920x1080:r=30:d=10,geq=random(1)*255:128:128" \
  -c:v libx264 -crf 18 noise_10s.mp4

# Static frame looped (tests pure temporal compression)
ffmpeg -loop 1 -i frame.png -t 30 -c:v libx264 -crf 18 -pix_fmt yuv420p static_30s.mp4

# Rapid scene cuts (1 fps slideshow of diverse images)
ffmpeg -f lavfi -i "testsrc2=s=1920x1080:r=1:d=60" -r 30 -c:v libx264 -crf 18 slideshow.mp4
```

#### Step 3: Labelling & Metadata

Create a manifest CSV/JSON for each generated video:

```json
{
  "id": "seed03_720p_noisy_24fps",
  "source_youtube_id": "dQw4w9WgXcQ",
  "content_type": "music_video",
  "resolution": "1280x720",
  "fps": 24,
  "duration_s": 30,
  "bit_depth": 8,
  "chroma": "4:2:0",
  "spatial_complexity": "high",
  "temporal_complexity": "high",
  "transformations": ["scale_720p", "noise_20", "fps_24"],
  "source_codec": "h264",
  "source_crf": 18,
  "file_size_bytes": 12345678,
  "bitrate_kbps": 3200
}
```

---

### Speed Optimisations

#### 1. Parallelise with GNU Parallel or xargs
```bash
# Process all seeds in parallel (use N = number of CPU cores)
find seeds/ -name "*.mp4" | parallel -j $(nproc) ./transform.sh {}
```

#### 2. Use Stream Copy Where Possible
When only trimming (not re-encoding), use `-c copy` to avoid the decode/encode cycle entirely:
```bash
ffmpeg -ss 10 -i input.mp4 -t 30 -c copy -avoid_negative_ts 1 trimmed.mp4
```
This is **100–1000x faster** than re-encoding.

#### 3. Hardware-Accelerated Encoding
```bash
# NVIDIA NVENC (H.264)
ffmpeg -hwaccel cuda -i input.mp4 -c:v h264_nvenc -preset p4 -cq 18 out.mp4

# NVIDIA NVENC (H.265)
ffmpeg -hwaccel cuda -i input.mp4 -c:v hevc_nvenc -preset p4 -cq 18 out.mp4

# Apple VideoToolbox (macOS)
ffmpeg -i input.mp4 -c:v h264_videotoolbox -q:v 50 out.mp4

# Intel QSV
ffmpeg -hwaccel qsv -i input.mp4 -c:v h264_qsv -global_quality 18 out.mp4
```

> [!TIP]
> NVENC is **5–10x faster** than `libx264` on a modern GPU, with only ~5-10% quality loss at equivalent bitrates. Use it for generating bulk test data where perfect quality isn't critical.

#### 4. Combine Filters into a Single Pass
Instead of running FFmpeg multiple times (scale → noise → encode), chain filters:
```bash
# Single-pass: scale + add noise + set fps in one command
ffmpeg -i input.mp4 \
  -vf "scale=1280:720:flags=lanczos,noise=alls=20:allf=t,fps=24" \
  -c:v libx264 -crf 18 out.mp4
```

#### 5. Use `-preset ultrafast` for Test Data
Since the generated videos are *inputs* to your compression algorithm (not the final output), use the fastest encoding preset:
```bash
ffmpeg -i input.mp4 -c:v libx264 -preset ultrafast -crf 18 out.mp4
```
This is **4–6x faster** than `-preset medium` (default). The files will be larger, but that's fine — they're test inputs.

#### 6. Batch yt-dlp Downloads
```bash
# Download all URLs from a file in parallel
yt-dlp -a urls.txt --concurrent-fragments 4 -f "bestvideo[height<=2160]+bestaudio" \
  --merge-output-format mp4 -o "seeds/%(id)s.%(ext)s"
```

#### 7. Pipeline Architecture
```
yt-dlp (download)  →  trim (stream copy)  →  transform (GPU encode)  →  label (metadata)
     ↓                      ↓                       ↓                       ↓
  [parallel]            [parallel]              [parallel]              [parallel]
  ~5 at once            instant                 GPU-bound               instant
```

Use a task queue (even a simple bash script with `&` and `wait`) to keep the GPU saturated.

---

## Part 3 — Codec Variation

### Why Codec Diversity Matters

Different codecs exploit different mathematical models for compression. A robust compression algorithm should handle all major codecs because:

1. **Different block partitioning** — H.264 uses 16×16 macroblocks, H.265 uses up to 64×64 CTUs, AV1 uses up to 128×128 superblocks
2. **Different transform bases** — DCT everywhere, but AV1 adds ADST, identity transforms
3. **Different prediction modes** — AV1 has 56 intra prediction modes vs H.264's 9
4. **Different entropy coding** — CABAC (H.264/H.265), ANS (AV1), VP9 (multi-symbol bool coder)
5. **Different loop filters** — deblocking, SAO (H.265), CDEF + loop restoration (AV1)

### Codecs to Include

| Codec | FFmpeg Encoder | Container | Notes |
|-------|---------------|-----------|-------|
| **H.264 / AVC** | `libx264` | MP4 | Universal baseline — most deployed codec |
| **H.265 / HEVC** | `libx265` | MP4 | 30–50% better than H.264, widely used in streaming |
| **VP9** | `libvpx-vp9` | WebM | Google's open codec, YouTube default for a long time |
| **AV1** | `libsvtav1` | MP4/WebM | State-of-the-art, 30%+ better than H.265, slow to encode |
| **VP8** | `libvpx` | WebM | Legacy, but tests older codec handling |
| **MPEG-2** | `mpeg2video` | MPEG-TS | Legacy broadcast — very different compression model |

### FFmpeg Encoding Commands per Codec

```bash
# H.264 (baseline, main, high profiles)
ffmpeg -i input.mp4 -c:v libx264 -profile:v baseline -crf 23 out_h264_baseline.mp4
ffmpeg -i input.mp4 -c:v libx264 -profile:v main -crf 23 out_h264_main.mp4
ffmpeg -i input.mp4 -c:v libx264 -profile:v high -crf 23 out_h264_high.mp4

# H.265 (main, main10)
ffmpeg -i input.mp4 -c:v libx265 -crf 28 out_h265.mp4
ffmpeg -i input.mp4 -c:v libx265 -crf 28 -pix_fmt yuv420p10le out_h265_10bit.mp4

# VP9
ffmpeg -i input.mp4 -c:v libvpx-vp9 -crf 30 -b:v 0 -row-mt 1 out_vp9.webm

# AV1 (SVT-AV1 — fast)
ffmpeg -i input.mp4 -c:v libsvtav1 -crf 30 -preset 6 out_av1.mp4

# AV1 (libaom — slow but reference quality)
ffmpeg -i input.mp4 -c:v libaom-av1 -crf 30 -cpu-used 4 -row-mt 1 out_av1_ref.mp4

# VP8
ffmpeg -i input.mp4 -c:v libvpx -crf 10 -b:v 1M out_vp8.webm

# MPEG-2
ffmpeg -i input.mp4 -c:v mpeg2video -b:v 8M out_mpeg2.mpg
```

### Codec-Specific Speed Considerations

| Codec | Relative Encode Speed | GPU Acceleration |
|-------|----------------------|------------------|
| H.264 (libx264) | 1x (baseline) | NVENC, QSV, VideoToolbox |
| H.265 (libx265) | 0.3–0.5x | NVENC, QSV, VideoToolbox |
| VP9 (libvpx-vp9) | 0.1–0.2x | *None* (CPU only) |
| AV1 (libsvtav1) | 0.2–0.4x | *None widely* (CPU only, very parallelisable) |
| AV1 (libaom) | 0.02–0.05x | *None* |

> [!WARNING]
> VP9 and AV1 encoding is **extremely slow** compared to H.264. For test dataset generation, use SVT-AV1 (not libaom) and consider encoding only a subset of videos in these codecs, or use lower quality presets.

### Codec Coverage Strategy

Rather than encoding every video in every codec, use a **stratified approach**:

1. **All videos** → H.264 (fast, universal baseline)
2. **50% of videos** → H.265 (moderate speed, important codec)
3. **25% of videos** → VP9 + AV1 (slow, but necessary for coverage)
4. **5% of videos** → MPEG-2 / VP8 (legacy edge cases)

This gives good coverage while keeping generation time manageable.

### Codec Profile Variation

Within each codec, vary the encoding profiles and quality levels:

```
H.264:
├── Profile: baseline, main, high
├── CRF: 8 (near-lossless), 18 (high quality), 28 (medium), 38 (low quality)
└── Preset: ultrafast, medium, slow

H.265:
├── Profile: main, main10
├── CRF: 18, 28, 38
└── Preset: ultrafast, medium, slow

AV1 (SVT):
├── CRF: 20, 30, 40, 50
└── Preset: 4 (slow/quality), 8 (fast), 12 (ultrafast)
```

---

## Summary: Recommended Test Dataset Composition

| Dimension | Values | Count |
|-----------|--------|-------|
| Seed videos (from YouTube) | Diverse content types | ~40 |
| × Resolutions | 480p, 720p, 1080p, 4K | 4 |
| × Frame rates | 24, 30, 60 | 3 |
| × Transformations | clean, noisy, blurred, sharp, speed-changed | 5 |
| × Codecs | H.264, H.265, VP9, AV1 | 4 (stratified) |
| × Quality levels | high, medium, low | 3 |
| + Synthetic edge cases | black, noise, static, slideshow, interlaced | ~10 |

**Estimated total: 200–500 unique test videos** (with stratified codec/quality sampling)

**Estimated storage: 50–200 GB** (depending on resolution/duration distribution)

**Estimated generation time:**
- With GPU (NVENC): ~4–8 hours
- CPU only (all codecs): ~24–72 hours
- Hybrid (H.264/H.265 on GPU, VP9/AV1 on CPU in parallel): ~8–16 hours
