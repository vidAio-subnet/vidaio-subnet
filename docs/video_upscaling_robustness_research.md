# Video Upscaling Robustness: Factors, Test Dataset Design & Generation Pipeline

## Part 1 — Factors That Determine Upscaling Difficulty

Video upscaling (super-resolution) reconstructs high-resolution detail from low-resolution input. The difficulty and quality of upscaling depends on how much recoverable information exists in the source and how challenging the content is to hallucinate plausibly. Every factor below directly affects upscaling performance.

---

### 1. Source Resolution & Scale Factor

The most fundamental variable — how much resolution must be invented.

| Scale Factor | Input → Output | Difficulty | Notes |
|-------------|----------------|-----------|-------|
| 2× | 540p → 1080p | Moderate | Most algorithms optimised for this |
| 3× | 360p → 1080p | Hard | Significant detail hallucination required |
| 4× | 270p → 1080p | Very hard | Most lost information, highest failure rate |
| Non-integer | 480p → 1080p (2.25×) | Tricky | Sub-pixel alignment issues, ringing artefacts |
| Asymmetric | Different X/Y scale | Unusual | Anamorphic content, tests edge handling |

> [!IMPORTANT]
> Scale factor is the single biggest determinant of upscaling difficulty. A 4× upscale must invent 15 out of every 16 pixels — the algorithm is mostly hallucinating.

---

### 2. Spatial Complexity & Texture Detail

How much high-frequency information the algorithm must reconstruct:

| Content Type | Upscaling Difficulty | Why |
|-------------|---------------------|-----|
| **Fine text / subtitles** | Extremely hard | Requires pixel-perfect edge reconstruction; errors are immediately obvious |
| **Hair / fur** | Very hard | Dense, irregular high-frequency patterns |
| **Fabric weave / mesh** | Very hard | Aliasing-prone repetitive microstructures |
| **Foliage / grass** | Hard | Fractal-like detail at all scales |
| **Brick / tile patterns** | Hard | Regular patterns susceptible to moiré |
| **Skin / faces** | Medium-Hard | Perceptually critical; uncanny valley risk |
| **Smooth gradients / sky** | Easy | Low-frequency, little to reconstruct |
| **Solid colours / flat fills** | Very easy | Trivial interpolation |

---

### 3. Temporal Complexity & Motion

Motion interacts with upscaling in critical ways:

| Factor | Low Difficulty | High Difficulty |
|--------|---------------|-----------------|
| **Motion magnitude** | Static tripod shot | Fast panning, handheld shake |
| **Motion type** | Smooth linear pan | Chaotic / random motion |
| **Object speed** | Slow-moving subjects | Fast sports, vehicles |
| **Motion blur** | Sharp frames (high shutter speed) | Heavy motion blur (1/24s shutter) |
| **Temporal consistency** | — | Flickering / shimmer between upscaled frames |
| **Scene cuts** | Long takes | Rapid cuts (algorithm must quickly adapt) |

> [!NOTE]
> **Temporal consistency** is a uniquely challenging upscaling metric. Even if each frame looks good individually, flickering or shimmer between frames destroys perceived quality. Fast motion exposes temporal inconsistency most aggressively.

---

### 4. Compression Artefacts in Source

Upscaling algorithms must handle already-degraded inputs:

| Artefact Type | Source | Impact on Upscaling |
|--------------|--------|---------------------|
| **Block artefacts** | Heavy H.264/H.265 quantization | Algorithm may amplify block boundaries |
| **Ringing / Gibbs** | Oversharpened or heavily compressed edges | Creates halos that get magnified |
| **Banding** | Low bit depth + gradient regions | Staircase patterns become more visible at higher res |
| **Mosquito noise** | Lossy compression near edges | Algorithm may hallucinate false detail from noise |
| **Macroblocking** | Very low bitrate H.264 | Large visible blocks that upscaler must smooth |
| **Colour bleeding** | Heavy chroma subsampling (4:2:0) | Chroma channel is already 2× downscaled, must be 4× recovered |
| **Clean source** | Lossless or very high bitrate | Easiest case — tests pure upscaling ability |

> [!WARNING]
> Real-world upscaling inputs are almost always compressed. An algorithm that only works on clean sources is useless in practice. Testing with CRF 35-45 sources is essential.

---

### 5. Downscaling Method (How the Low-Res Was Created)

The downscaling filter used to produce the low-resolution input fundamentally changes what information is preserved:

| Downscale Filter | FFmpeg Flag | Information Preserved | Upscaling Difficulty |
|-----------------|-------------|----------------------|---------------------|
| **Nearest-neighbour** | `neighbor` | Blocky, aliased | Easy (sharp edges) but alien-looking |
| **Bilinear** | `bilinear` | Blurry, moderate aliasing | Medium |
| **Bicubic** | `bicubic` | Good balance, slight ringing | Medium |
| **Lanczos** | `lanczos` | Best detail, slight ringing | Medium-Hard |
| **Area averaging** | `area` | Good for large scale reductions, smooth | Hard (most information lost smoothly) |
| **Gaussian blur + subsample** | Manual | Very blurry, no aliasing | Hard (minimal high-freq residue) |

> [!IMPORTANT]
> The downscale method is a **hidden variable** in real-world upscaling. Your test dataset should include multiple downscale methods to test robustness. Most YouTube videos are downscaled using Lanczos or bicubic by their transcoder.

---

### 6. Bit Depth & Colour Space

| Property | Impact on Upscaling |
|----------|---------------------|
| **8-bit** | Standard; banding in gradients becomes visible at higher res |
| **10-bit** | Smoother gradients; easier to upscale without banding |
| **4:2:0 chroma** | Chroma is already at half resolution — upscaler must super-resolve luma AND chroma |
| **4:2:2 chroma** | Better chroma detail preserved |
| **4:4:4 chroma** | Full colour info — tests luma-only vs full upscaling |
| **BT.709** | Standard colour space |
| **BT.2020 / HDR** | Wider gamut — tests colour handling during upscaling |

---

### 7. Frame Rate

| Frame Rate | Impact |
|------------|--------|
| **24 fps** | Larger inter-frame differences; temporal consistency harder |
| **30 fps** | Standard; moderate temporal redundancy |
| **60 fps** | More temporal info available; easier temporal consistency but 2× the processing |
| **Variable FPS** | Irregular frame timing; tests temporal model robustness |

---

### 8. Content Semantic Categories

Certain content types are perceptually critical — errors are immediately visible:

| Category | Key Challenge | Perceptual Sensitivity |
|----------|--------------|----------------------|
| **Faces / portraits** | Uncanny valley; must preserve facial features exactly | 🔴 Critical |
| **Text / titles / UI** | Must be sharp and readable after upscale | 🔴 Critical |
| **Animation / anime** | Clean lines + flat fills; aliasing on edges very visible | 🔴 Critical |
| **Nature / landscapes** | Organic textures; errors less obvious but detail important | 🟡 High |
| **Sports / action** | Fast motion + crowd detail | 🟡 High |
| **Gaming / CG** | Mix of clean geometry + complex textures | 🟡 High |
| **Surveillance / CCTV** | Very low quality source; real-world use case | 🟠 Medium |
| **Underwater / fog** | Soft focus, low contrast; upscaling adds little value | 🟢 Low |
| **Screencasts / slides** | Sharp edges, text, flat regions | 🔴 Critical |

---

### 9. Noise & Grain in Source

| Type | Impact |
|------|--------|
| **Sensor noise** (high ISO) | Upscaler may amplify noise as detail, or denoise and lose real detail |
| **Film grain** | Perceptually expected; algorithm should preserve or plausibly regenerate |
| **Compression noise** | Should be cleaned, not amplified |
| **Clean source** | Baseline test — pure upscaling ability |
| **Mixed** (grain + compression) | Most realistic scenario |

---

### 10. Edge Cases & Adversarial Content

| Case | What It Tests |
|------|--------------|
| **Pure black / pure white** | Does the algorithm add noise to flat regions? |
| **Single-pixel-wide lines** | Aliasing handling on sub-pixel features |
| **Moiré-prone patterns** (stripes, grids) | Frequency aliasing during upscaling |
| **Interlaced content** | Must deinterlace before or during upscale |
| **Letterboxed / pillarboxed** | Black bars should remain clean; content area correctly identified |
| **Aspect ratio change** | 4:3 content in 16:9 container |
| **Very short clips** (< 2s) | Temporal model initialisation |
| **Synthetic test patterns** | Colour bars, resolution charts — objective quality measurement |

---

## How to Make Test Data Maximally Thorough

### Dimension Coverage Matrix

```
Dimensions to vary:
├── Scale factor:           [2×, 3×, 4×]
├── Source resolution:      [240p, 360p, 480p, 540p, 720p]
├── Spatial complexity:     [low, medium, high, extreme]
├── Temporal complexity:    [static, low, medium, high]
├── Downscale method:       [nearest, bilinear, bicubic, lanczos, area]
├── Source compression:     [clean/lossless, light CRF, medium CRF, heavy CRF]
├── Content type:           [faces, text, animation, nature, sports, gaming, screencast]
├── Noise level:            [clean, light noise, heavy noise, grain]
├── Frame rate:             [24, 30, 60]
├── Bit depth:              [8-bit, 10-bit]
├── Codec:                  [H.264, H.265, VP9, AV1]
└── Duration:               [2s, 10s, 30s, 2min]
```

Full factorial: 3×5×4×4×5×4×7×4×3×2×4×4 = **32,256,000** — obviously impractical.

### Practical Strategy: Stratified Sampling + Critical Combinations

**Tier 1 — Core Matrix (must cover)**: Scale × Spatial × Temporal × Downscale
- 3 × 4 × 4 × 5 = **240 essential combinations**

**Tier 2 — Important Modifiers**: Overlay compression level and content type across Tier 1
- Distribute 4 CRF levels across the 240 cells → each cell gets one CRF variant
- 7 content types sampled across cells → ~35 per content type

**Tier 3 — Secondary Dimensions**: FPS, bit depth, codec spread across Tier 2
- Ensure each value appears ≥10 times

**Target: 200–400 test videos** that cover the space with no blind spots.

### Mandatory Edge Cases (Always Include)

| # | Test Case | Purpose |
|---|-----------|---------|
| 1 | Resolution test chart (EBU/SMPTE) downscaled + upscaled | Objective spatial frequency response |
| 2 | Scrolling text at various speeds | Text sharpness under motion |
| 3 | Face close-up → 4× upscale from 240p | Worst-case perceptual test |
| 4 | Anime with thin line art → 4× upscale | Aliasing / line integrity |
| 5 | High-ISO night footage → 2× upscale | Noise amplification test |
| 6 | CRF 45 source → 2× upscale | Artefact amplification test |
| 7 | Interlaced broadcast footage | Deinterlace + upscale pipeline |
| 8 | Letterboxed 4:3 in 16:9 | Edge detection / padding handling |
| 9 | Pure random noise → upscale | Should not hallucinate structure |
| 10 | Static frame → 30s (check temporal consistency) | No shimmer on static content |

---

## Part 2 — Creating the Dataset from YouTube Videos + FFmpeg

### Thinking Out Loud: The Pipeline

**The challenge**: Upscaling test data is fundamentally different from compression test data. We need:
1. **High-resolution reference videos** (ground truth) — the higher the better
2. **Synthetically downscaled versions** — our test inputs
3. The **quality metric** is then: upscale(downscaled) vs original reference

This means we should download the **highest available quality** from YouTube, then downscale programmatically to create (input, reference) pairs.

**Key realisation**: The downscaling method itself is a test variable. We shouldn't just downscale once — we should create variants using different downscale algorithms, as this changes what the upscaler receives.

#### Step 1: Curate Seed Set via yt-dlp (Download at Max Quality)

Target ~40 YouTube videos at the highest possible resolution:

| Category | Search Terms | Count | Why |
|----------|-------------|-------|-----|
| Face close-ups | "interview 4K", "portrait video 4K", "makeup tutorial 4K" | 5 | Perceptually critical |
| Text-heavy | "coding tutorial 1080p", "presentation recording", "news crawl" | 4 | Sharp edge reconstruction |
| Anime / animation | "anime 4K upscale", "animated short film", "motion graphics" | 4 | Line art, flat fills |
| Nature / landscape | "4K nature documentary", "drone footage 4K", "macro photography" | 5 | Organic texture detail |
| Sports / action | "4K football highlights", "F1 onboard 4K", "skateboarding 4K" | 4 | Fast motion + detail |
| Gaming | "4K 60fps gameplay", "Minecraft 4K", "retro game capture" | 4 | CG textures + UI elements |
| Screencast | "VS Code tutorial", "Figma walkthrough", "terminal demo" | 3 | Text + UI + flat regions |
| Film grain | "8mm film scan", "35mm film transfer", "VHS recording" | 3 | Noise handling |
| Low-light | "night city 4K", "astrophotography timelapse", "concert footage" | 3 | High ISO noise |
| General / diverse | "street photography 4K", "cooking show", "travel vlog 4K" | 5 | Broad coverage |

```bash
# Download at maximum available quality (prefer 4K, fallback to best)
yt-dlp -f "bestvideo[height>=2160]+bestaudio/bestvideo[height>=1080]+bestaudio/best" \
  --merge-output-format mp4 \
  -o "seeds/%(id)s_%(title).50s.%(ext)s" \
  "<URL>"

# Batch download from URL list
yt-dlp -a urls.txt --concurrent-fragments 4 \
  -f "bestvideo[height>=2160]+bestaudio/bestvideo[height>=1080]+bestaudio/best" \
  --merge-output-format mp4 \
  -o "seeds/%(id)s.%(ext)s"
```

> [!TIP]
> Always download the highest resolution available. A 4K source gives usable (input, reference) pairs for 2× (1080p→4K), 4× (540p→4K), and even 720p→4K (non-integer). A 1080p source only supports up to 2× (540p→1080p).

#### Step 2: Create Reference Clips (Ground Truth)

Trim seeds into manageable reference clips:

```bash
# Extract clean reference clips (stream copy — no re-encode)
ffmpeg -ss 10 -i "seeds/video.mp4" -t 30 -c copy \
  -avoid_negative_ts make_start "refs/video_ref_30s.mp4"

# For very long videos, extract multiple clips at different timestamps
for offset in 10 60 120 300; do
  ffmpeg -ss $offset -i "seeds/video.mp4" -t 10 -c copy \
    -avoid_negative_ts make_start "refs/video_ref_${offset}.mp4"
done
```

#### Step 3: Downscale to Create Test Inputs

This is where the test dataset is actually constructed. Each reference clip gets downscaled in multiple ways:

##### A. Scale Factor Variations
```bash
REF="refs/video_ref.mp4"

# 2× downscale (1080p from 4K reference, or 540p from 1080p)
ffmpeg -i "$REF" -vf "scale=iw/2:ih/2:flags=lanczos" -c:v libx264 -crf 10 "inputs/${base}_2x_lanczos.mp4"

# 3× downscale
ffmpeg -i "$REF" -vf "scale=iw/3:ih/3:flags=lanczos" -c:v libx264 -crf 10 "inputs/${base}_3x_lanczos.mp4"

# 4× downscale
ffmpeg -i "$REF" -vf "scale=iw/4:ih/4:flags=lanczos" -c:v libx264 -crf 10 "inputs/${base}_4x_lanczos.mp4"
```

##### B. Downscale Method Variations
```bash
# Same scale factor (2×) but different downscale algorithms
for method in "neighbor" "bilinear" "bicubic" "lanczos" "area" "spline"; do
  ffmpeg -i "$REF" \
    -vf "scale=iw/2:ih/2:flags=$method" \
    -c:v libx264 -crf 10 \
    "inputs/${base}_2x_${method}.mp4"
done

# Gaussian blur + subsample (simulates optical blur before sampling)
ffmpeg -i "$REF" \
  -vf "gblur=sigma=1.5,scale=iw/2:ih/2:flags=bilinear" \
  -c:v libx264 -crf 10 \
  "inputs/${base}_2x_gaussian.mp4"
```

##### C. Source Compression Level Variations
After downscaling, re-encode at different quality levels to simulate real-world degraded inputs:

```bash
# Near-lossless input (tests pure upscaling)
ffmpeg -i "$REF" -vf "scale=iw/2:ih/2:flags=lanczos" \
  -c:v libx264 -crf 5 "inputs/${base}_2x_crf5.mp4"

# Light compression (typical good-quality web video)
ffmpeg -i "$REF" -vf "scale=iw/2:ih/2:flags=lanczos" \
  -c:v libx264 -crf 23 "inputs/${base}_2x_crf23.mp4"

# Medium compression (typical streaming)
ffmpeg -i "$REF" -vf "scale=iw/2:ih/2:flags=lanczos" \
  -c:v libx264 -crf 30 "inputs/${base}_2x_crf30.mp4"

# Heavy compression (worst-case: visible artefacts)
ffmpeg -i "$REF" -vf "scale=iw/2:ih/2:flags=lanczos" \
  -c:v libx264 -crf 40 "inputs/${base}_2x_crf40.mp4"

# Extremely heavy (adversarial — tests artefact amplification)
ffmpeg -i "$REF" -vf "scale=iw/2:ih/2:flags=lanczos" \
  -c:v libx264 -crf 48 "inputs/${base}_2x_crf48.mp4"
```

##### D. Noise Injection (Before Downscale)
Add noise to the reference before downscaling to test noise robustness:

```bash
# Light sensor noise
ffmpeg -i "$REF" \
  -vf "noise=alls=10:allf=t,scale=iw/2:ih/2:flags=lanczos" \
  -c:v libx264 -crf 10 "inputs/${base}_2x_noise10.mp4"

# Heavy sensor noise (high ISO simulation)
ffmpeg -i "$REF" \
  -vf "noise=alls=30:allf=t+u,scale=iw/2:ih/2:flags=lanczos" \
  -c:v libx264 -crf 10 "inputs/${base}_2x_noise30.mp4"

# Film grain simulation
ffmpeg -i "$REF" \
  -vf "noise=alls=15:allf=t,scale=iw/2:ih/2:flags=lanczos" \
  -c:v libx264 -crf 10 "inputs/${base}_2x_grain.mp4"
```

##### E. Frame Rate Variations
```bash
# Convert to 24fps before downscaling (larger temporal deltas)
ffmpeg -i "$REF" \
  -vf "fps=24,scale=iw/2:ih/2:flags=lanczos" \
  -c:v libx264 -crf 10 "inputs/${base}_2x_24fps.mp4"

# Convert to 60fps (more temporal redundancy)
ffmpeg -i "$REF" \
  -vf "fps=60,scale=iw/2:ih/2:flags=lanczos" \
  -c:v libx264 -crf 10 "inputs/${base}_2x_60fps.mp4"
```

##### F. Bit Depth Conversion
```bash
# 10-bit source (smoother gradients, tests bit-depth aware upscaling)
ffmpeg -i "$REF" \
  -vf "scale=iw/2:ih/2:flags=lanczos" \
  -c:v libx265 -crf 10 -pix_fmt yuv420p10le \
  "inputs/${base}_2x_10bit.mp4"
```

##### G. Synthetic Test Patterns
```bash
# EBU colour bars (standard test pattern)
ffmpeg -f lavfi -i "smptebars=s=3840x2160:r=30:d=10" \
  -c:v libx264 -crf 5 "refs/smpte_bars_4k.mp4"
ffmpeg -i "refs/smpte_bars_4k.mp4" \
  -vf "scale=960:540:flags=lanczos" \
  -c:v libx264 -crf 10 "inputs/smpte_bars_4x.mp4"

# Resolution test pattern (fine lines at various frequencies)
ffmpeg -f lavfi -i "testsrc2=s=3840x2160:r=30:d=10" \
  -c:v libx264 -crf 5 "refs/testsrc_4k.mp4"
ffmpeg -i "refs/testsrc_4k.mp4" \
  -vf "scale=960:540:flags=lanczos" \
  -c:v libx264 -crf 10 "inputs/testsrc_4x.mp4"

# Scrolling text (tests temporal consistency + text clarity)
ffmpeg -f lavfi -i "color=c=white:s=1920x1080:r=30:d=30,drawtext=\
  fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:\
  text='The quick brown fox jumps over the lazy dog 0123456789':\
  fontsize=36:fontcolor=black:x=(w-text_w)/2:y=h-50*t" \
  -c:v libx264 -crf 5 "refs/scrolling_text.mp4"

# Moiré-prone pattern (striped/grid)
ffmpeg -f lavfi -i "testsrc=s=3840x2160:r=30:d=10" \
  -c:v libx264 -crf 5 "refs/moire_test_4k.mp4"

# Pure random noise (should NOT improve after upscaling)
ffmpeg -f lavfi -i "nullsrc=s=960x540:r=30:d=5,geq=random(1)*255:128:128" \
  -c:v libx264 -crf 10 "inputs/random_noise.mp4"

# Static single frame for 30s (temporal consistency check)
ffmpeg -loop 1 -i "refs/sample_frame.png" -t 30 \
  -vf "scale=960:540:flags=lanczos" \
  -c:v libx264 -crf 10 -pix_fmt yuv420p \
  "inputs/static_frame_30s.mp4"
```

#### Step 4: Labelling & Metadata

Create a manifest for the paired dataset:

```json
{
  "id": "seed03_2x_lanczos_crf23",
  "reference_path": "refs/seed03_ref.mp4",
  "input_path": "inputs/seed03_2x_lanczos_crf23.mp4",
  "source_youtube_id": "dQw4w9WgXcQ",
  "content_type": "faces",
  "reference_resolution": "3840x2160",
  "input_resolution": "1920x1080",
  "scale_factor": 2.0,
  "downscale_method": "lanczos",
  "source_crf": 23,
  "fps": 30,
  "bit_depth": 8,
  "chroma": "4:2:0",
  "spatial_complexity": "medium",
  "temporal_complexity": "low",
  "noise_level": "clean",
  "duration_s": 30,
  "codec": "h264",
  "transformations": ["scale_2x_lanczos", "crf_23"]
}
```

---

### Speed Optimisations

#### 1. Stream Copy for Trimming (Zero Cost)
```bash
# Trim reference clips without re-encoding
ffmpeg -ss 10 -i input.mp4 -t 30 -c copy trimmed.mp4
# Speed: essentially instant (100-1000x faster than re-encode)
```

#### 2. Single-Pass Filter Chains
Instead of downscaling and then re-encoding with degradation separately, combine into one pass:
```bash
# Bad: two passes (decode→scale→encode + decode→encode)
ffmpeg -i ref.mp4 -vf "scale=iw/2:ih/2:flags=lanczos" -crf 5 temp.mp4
ffmpeg -i temp.mp4 -c:v libx264 -crf 35 input_crf35.mp4

# Good: single pass with quality control
ffmpeg -i ref.mp4 \
  -vf "scale=iw/2:ih/2:flags=lanczos" \
  -c:v libx264 -crf 35 input_crf35.mp4
```

However, note that for some test variants you **do** want two passes — e.g., when testing how the upscaler handles double-compressed content:
```bash
# Intentional two-pass: downscale at high quality, THEN degrade
ffmpeg -i ref.mp4 -vf "scale=iw/2:ih/2:flags=lanczos" -crf 5 clean_lr.mp4
ffmpeg -i clean_lr.mp4 -c:v libx264 -crf 40 degraded_lr.mp4
# This simulates a real-world scenario where someone compressed an already low-res video
```

#### 3. Hardware-Accelerated Encoding (NVENC for A6000/A100)
```bash
# CPU decode → CPU filter → NVENC encode (NO -hwaccel, no NVDEC on these GPUs)
ffmpeg -i ref.mp4 \
  -vf "scale=iw/2:ih/2:flags=lanczos" \
  -c:v h264_nvenc -preset p4 -rc constqp -qp 10 \
  input_2x.mp4

# HEVC via NVENC
ffmpeg -i ref.mp4 \
  -vf "scale=iw/2:ih/2:flags=lanczos" \
  -c:v hevc_nvenc -preset p4 -rc constqp -qp 12 \
  input_2x_h265.mp4
```

> [!CAUTION]
> Do **not** use `-hwaccel cuda` or `-hwaccel cuvid` on A6000/A100 — they lack NVDEC decode hardware. Pipeline is always CPU decode → CPU filter → GPU encode.

#### 4. Parallelise with GNU Parallel
```bash
#!/bin/bash
# generate_upscale_dataset.sh

generate_variants() {
    local ref="$1"
    local base=$(basename "$ref" .mp4)

    # Scale factor variants
    for scale in 2 3 4; do
        for method in lanczos bicubic bilinear area; do
            ffmpeg -loglevel warning -i "$ref" \
                -vf "scale=iw/${scale}:ih/${scale}:flags=${method}" \
                -c:v h264_nvenc -preset p4 -rc constqp -qp 10 \
                -y "inputs/${base}_${scale}x_${method}.mp4"
        done
    done

    # Compression level variants (2× lanczos base)
    for crf in 5 18 28 38 48; do
        ffmpeg -loglevel warning -i "$ref" \
            -vf "scale=iw/2:ih/2:flags=lanczos" \
            -c:v libx264 -crf $crf \
            -y "inputs/${base}_2x_crf${crf}.mp4"
    done

    echo "Done: $base"
}

export -f generate_variants
find refs/ -name "*.mp4" | parallel -j 8 generate_variants {}
```

#### 5. Batch Processing: One Source, Many Outputs
FFmpeg can write multiple outputs from a single decode:
```bash
# Decode once, produce three downscale variants simultaneously
ffmpeg -i ref.mp4 \
  -vf "scale=iw/2:ih/2:flags=lanczos" -c:v h264_nvenc -preset p4 -qp 10 "inputs/${base}_2x.mp4" \
  -vf "scale=iw/3:ih/3:flags=lanczos" -c:v h264_nvenc -preset p4 -qp 10 "inputs/${base}_3x.mp4" \
  -vf "scale=iw/4:ih/4:flags=lanczos" -c:v h264_nvenc -preset p4 -qp 10 "inputs/${base}_4x.mp4"
```

#### 6. Use `-preset ultrafast` for libx264 Inputs
Since these are test inputs (not deliverables), encoding speed trumps efficiency:
```bash
ffmpeg -i ref.mp4 \
  -vf "scale=iw/2:ih/2:flags=lanczos" \
  -c:v libx264 -preset ultrafast -crf 23 \
  input_fast.mp4
# 4-6x faster than -preset medium
```

#### 7. Pipeline Architecture
```
yt-dlp (download 4K)
       │
       ▼
  [trim clips] ──── -c copy (instant)
       │
       ▼
  [reference clips] ──── stored as ground truth
       │
       ▼
  [downscale matrix] ──── parallel (GNU Parallel × NVENC)
       │                    ├── 2×, 3×, 4× scales
       │                    ├── lanczos, bicubic, bilinear, area methods
       │                    └── CRF 5, 18, 28, 38, 48 quality levels
       ▼
  [inputs/] + [refs/] ──── paired dataset with metadata manifest
```

---

## Part 3 — Codec Variation for Upscaling Test Data

### Why Codec Matters for Upscaling

Different codecs introduce **different artefact signatures** that the upscaler must handle:

| Codec | Artefact Character | Impact on Upscaling |
|-------|-------------------|---------------------|
| **H.264** | 16×16 macroblock artefacts, deblocking | Most common; upscaler MUST handle well |
| **H.265** | Finer 8×8 to 64×64 CTU artefacts, SAO filtering | Smoother artefacts, less obvious blocking |
| **VP9** | Similar to H.265 but different deblocking | Tests codec-agnostic robustness |
| **AV1** | CDEF + loop restoration, very smooth even at low bitrate | Cleanest low-bitrate source; tests if upscaler benefits from cleaner input |
| **MPEG-2** | Very obvious block artefacts, no in-loop deblocking | Legacy content; tests handling of severe blocking |
| **VP8** | Similar to H.264 artefacts but different encoder decisions | Legacy web content |
| **MJPEG** | Per-frame JPEG artefacts (no temporal filtering) | Tests frame-by-frame artefact handling |

### Codecs as Downscale Source Encoder

The codec used to encode the low-res input affects what the upscaler sees:

```bash
REF="refs/video_ref.mp4"
BASE="video_ref"

# H.264 (universal baseline)
ffmpeg -i "$REF" -vf "scale=iw/2:ih/2:flags=lanczos" \
  -c:v libx264 -crf 28 "inputs/${BASE}_2x_h264_crf28.mp4"

# H.265 (smoother artefacts at same perceptual quality)
ffmpeg -i "$REF" -vf "scale=iw/2:ih/2:flags=lanczos" \
  -c:v libx265 -crf 32 "inputs/${BASE}_2x_h265_crf32.mp4"

# VP9 (YouTube's common codec)
ffmpeg -i "$REF" -vf "scale=iw/2:ih/2:flags=lanczos" \
  -c:v libvpx-vp9 -crf 35 -b:v 0 -row-mt 1 "inputs/${BASE}_2x_vp9.webm"

# AV1 (state-of-the-art — cleanest at low bitrate)
ffmpeg -i "$REF" -vf "scale=iw/2:ih/2:flags=lanczos" \
  -c:v libsvtav1 -crf 35 -preset 8 "inputs/${BASE}_2x_av1.mp4"

# MPEG-2 (legacy broadcast — worst artefacts)
ffmpeg -i "$REF" -vf "scale=iw/2:ih/2:flags=lanczos" \
  -c:v mpeg2video -b:v 3M "inputs/${BASE}_2x_mpeg2.mpg"

# MJPEG (per-frame JPEG, no temporal prediction)
ffmpeg -i "$REF" -vf "scale=iw/2:ih/2:flags=lanczos" \
  -c:v mjpeg -q:v 5 "inputs/${BASE}_2x_mjpeg.mp4"
```

### Codec × Quality Matrix

For thorough testing, cross codec with quality level:

| Codec | Near-Lossless | Good | Medium | Poor | Terrible |
|-------|--------------|------|--------|------|----------|
| H.264 | CRF 5 | CRF 18 | CRF 28 | CRF 38 | CRF 48 |
| H.265 | CRF 8 | CRF 22 | CRF 32 | CRF 42 | CRF 50 |
| VP9 | CRF 10 | CRF 25 | CRF 35 | CRF 45 | CRF 55 |
| AV1 | CRF 10 | CRF 25 | CRF 35 | CRF 45 | CRF 55 |
| MPEG-2 | 15M bps | 8M bps | 3M bps | 1M bps | 500K bps |

### Codec Coverage Strategy (Speed-Aware)

Since VP9 and AV1 are very slow to encode:

1. **All videos** → H.264 (fast, universal, most common real-world source)
2. **50% of videos** → H.265 (important, moderate speed)
3. **25% of videos** → VP9 + AV1 (slow but necessary — YouTube uses these)
4. **10% of videos** → MPEG-2 + MJPEG (legacy / adversarial)

### Speed Tips for Codec Generation

```bash
# H.264/H.265: Use NVENC (5-10x faster than CPU)
ffmpeg -i ref.mp4 -vf "scale=iw/2:ih/2:flags=lanczos" \
  -c:v h264_nvenc -preset p1 -rc constqp -qp 18 out.mp4

# VP9: Use row-mt and threads
ffmpeg -i ref.mp4 -vf "scale=iw/2:ih/2:flags=lanczos" \
  -c:v libvpx-vp9 -crf 30 -b:v 0 -row-mt 1 -threads 8 out.webm

# AV1: Use SVT-AV1 with fast preset (not libaom)
ffmpeg -i ref.mp4 -vf "scale=iw/2:ih/2:flags=lanczos" \
  -c:v libsvtav1 -crf 30 -preset 10 out.mp4
# preset 10-12 is fast; preset 4-6 is quality; avoid libaom entirely

# Run CPU codecs (VP9/AV1) in parallel with NVENC (H.264/H.265)
# since NVENC uses dedicated hardware, not CPU cores
```

---

## Summary: Recommended Upscaling Test Dataset Composition

| Dimension | Values | Count |
|-----------|--------|-------|
| Seed videos (from YouTube, 4K preferred) | Diverse content types | ~40 |
| Reference clips (trimmed ground truth) | 10-30s each | ~80 |
| × Scale factors | 2×, 3×, 4× | 3 |
| × Downscale methods | lanczos, bicubic, bilinear, area, gaussian | 5 |
| × Source compression | CRF 5, 18, 28, 38, 48 | 5 (stratified) |
| × Codecs | H.264, H.265, VP9, AV1 | 4 (stratified) |
| + Noise variants | clean, light, heavy, grain | 4 (subset) |
| + Frame rate variants | 24, 30, 60 | 3 (subset) |
| + Synthetic test patterns | colour bars, test chart, scrolling text, moiré, noise, static | ~10 |

**Estimated total: 300–600 (input, reference) pairs**

**Estimated storage: 100–400 GB** (4K references dominate storage)

**Estimated generation time (A6000/A100):**
- Downloading seeds: ~30 min
- Trimming references: ~2 min (stream copy)
- H.264/H.265 downscale variants (NVENC): ~1-2 hours
- VP9/AV1 variants (CPU, parallel): ~3-6 hours
- Synthetic test patterns: ~5 min
- **Total: ~4-8 hours**

---

## Key Differences from Compression Test Data

| Aspect | Compression Testing | Upscaling Testing |
|--------|-------------------|-------------------|
| **Input quality** | Any quality (testing encoder) | High quality reference required |
| **Data structure** | Single video per test | Paired (input, reference) |
| **Ground truth** | Original file | High-res reference |
| **Key variable** | Content complexity | Scale factor + downscale method |
| **Quality metric** | VMAF, SSIM, PSNR vs original | VMAF, SSIM, PSNR of upscaled vs reference |
| **Artefact concern** | Compression artefacts in output | Amplified artefacts + hallucinated detail |
| **Temporal concern** | Bitrate consistency | Temporal consistency (shimmer/flicker) |
| **Critical content** | High-motion, noisy scenes | Faces, text, fine lines, edges |
