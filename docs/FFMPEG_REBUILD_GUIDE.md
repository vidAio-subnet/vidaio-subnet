# FFmpeg Complete Rebuild Guide
## With dav1d, SVT-AV1, NVIDIA NVENC/NVDEC Support

### Quick Start (Automated)

```bash
# 0. OPTIONAL: Clean up old FFmpeg installation first
cd /root/vidaio-subnet
chmod +x cleanup_old_ffmpeg.sh
./cleanup_old_ffmpeg.sh

# 1. Make the rebuild script executable
chmod +x rebuild_ffmpeg_complete.sh

# 2. Run the rebuild (takes 15-30 minutes)
./rebuild_ffmpeg_complete.sh

# 3. Verify installation
ffmpeg -version
ffmpeg -decoders | grep av1
ffmpeg -encoders | grep av1
ffmpeg -encoders | grep nvenc

# 4. Restart scoring server
pm2 restart scoring-server
```

### What This Installs

| Component | Type | Purpose | Speed |
|-----------|------|---------|-------|
| **dav1d** | Decoder | Software AV1 decoder (best compatibility) | Fast |
| **libsvtav1** | Encoder | CPU-based AV1 encoder | Medium |
| **av1_cuvid** | Decoder | NVIDIA GPU AV1 decoder | Very Fast |
| **av1_nvenc** | Encoder | NVIDIA GPU AV1 encoder | Very Fast |
| **h264_nvenc** | Encoder | NVIDIA GPU H.264 encoder | Very Fast |
| **hevc_nvenc** | Encoder | NVIDIA GPU H.265 encoder | Very Fast |
| **libvmaf** | Filter | Netflix VMAF quality metric | N/A |
| **libx264** | Encoder | CPU H.264 encoder | Medium |
| **libx265** | Encoder | CPU H.265 encoder | Slow |

### Manual Steps (If Needed)

#### 1. Install Build Dependencies

```bash
sudo apt-get update
sudo apt-get install -y \
    autoconf automake build-essential cmake git-core \
    libass-dev libfreetype6-dev libgnutls28-dev \
    libmp3lame-dev libopus-dev libsdl2-dev libtool \
    libva-dev libvdpau-dev libvorbis-dev \
    libxcb1-dev libxcb-shm0-dev libxcb-xfixes0-dev \
    meson ninja-build pkg-config texinfo wget yasm zlib1g-dev \
    nasm libx264-dev libx265-dev libnuma-dev libvpx-dev \
    libfdk-aac-dev libtheora-dev
```

#### 2. Build dav1d (AV1 Decoder)

```bash
cd /root/ffmpeg_sources
git clone --depth 1 https://code.videolan.org/videolan/dav1d.git
cd dav1d
mkdir build && cd build
meson setup -Denable_tools=false -Denable_tests=false \
    --default-library=static .. \
    --prefix="/root/ffmpeg_build" \
    --libdir="/root/ffmpeg_build/lib"
ninja
ninja install
```

#### 3. Build SVT-AV1 (AV1 Encoder)

```bash
cd /root/ffmpeg_sources
git clone --depth 1 https://gitlab.com/AOMediaCodec/SVT-AV1.git
cd SVT-AV1/Build
cmake .. -G"Unix Makefiles" \
    -DCMAKE_INSTALL_PREFIX="/root/ffmpeg_build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_DEC=OFF \
    -DBUILD_SHARED_LIBS=OFF
make -j$(nproc)
make install
```

#### 4. Build libvmaf (Quality Metrics)

```bash
cd /root/ffmpeg_sources
git clone --depth 1 https://github.com/Netflix/vmaf.git
cd vmaf/libvmaf
meson setup build --buildtype=release \
    --prefix="/root/ffmpeg_build" \
    --default-library=static
ninja -C build
ninja -C build install
```

#### 5. Install NVIDIA Video Codec SDK Headers

```bash
cd /root/ffmpeg_sources
git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git
cd nv-codec-headers
make install PREFIX="/root/ffmpeg_build"
```

#### 6. Build FFmpeg with Everything

```bash
cd /root/ffmpeg_sources
git clone --depth 1 --branch n7.1 https://git.ffmpeg.org/ffmpeg.git
cd ffmpeg

PKG_CONFIG_PATH="/root/ffmpeg_build/lib/pkgconfig" ./configure \
  --prefix="/root/ffmpeg_build" \
  --pkg-config-flags="--static" \
  --extra-cflags="-I/root/ffmpeg_build/include -I/usr/local/cuda/include" \
  --extra-ldflags="-L/root/ffmpeg_build/lib -L/usr/local/cuda/lib64" \
  --extra-libs="-lpthread -lm" \
  --ld="g++" \
  --bindir="/usr/local/bin" \
  --enable-gpl \
  --enable-gnutls \
  --enable-libass \
  --enable-libfdk-aac \
  --enable-libfreetype \
  --enable-libmp3lame \
  --enable-libopus \
  --enable-libvorbis \
  --enable-libvpx \
  --enable-libx264 \
  --enable-libx265 \
  --enable-nonfree \
  --enable-libdav1d \
  --enable-libsvtav1 \
  --enable-libvmaf \
  --enable-cuda-nvcc \
  --enable-cuvid \
  --enable-nvenc \
  --enable-nvdec \
  --enable-libnpp \
  --enable-libtheora \
  --enable-version3

make -j$(nproc)  # Takes 10-20 minutes
sudo make install
sudo ldconfig
```

### Verification

```bash
# 1. Check FFmpeg version
ffmpeg -version

# 2. List all AV1 decoders
ffmpeg -decoders 2>&1 | grep -i av1
# Expected:
#   V....D av1          Alliance for Open Media AV1
#   V....D libdav1d     dav1d AV1 decoder
#   V..... av1_cuvid    Nvidia CUVID AV1 decoder

# 3. List all AV1 encoders
ffmpeg -encoders 2>&1 | grep -i av1
# Expected:
#   V..... libsvtav1    SVT-AV1 encoder
#   V..... av1_nvenc    NVIDIA NVENC AV1 encoder

# 4. Check NVIDIA encoders
ffmpeg -encoders 2>&1 | grep nvenc
# Expected:
#   V..... h264_nvenc   NVIDIA NVENC H.264 encoder
#   V..... hevc_nvenc   NVIDIA NVENC H.265/HEVC encoder
#   V..... av1_nvenc    NVIDIA NVENC AV1 encoder

# 5. Check NVIDIA decoders
ffmpeg -decoders 2>&1 | grep cuvid
# Expected:
#   V..... h264_cuvid   Nvidia CUVID H.264 decoder
#   V..... hevc_cuvid   Nvidia CUVID HEVC decoder
#   V..... av1_cuvid    Nvidia CUVID AV1 decoder

# 6. Verify VMAF filter
ffmpeg -filters 2>&1 | grep vmaf
# Expected: vmaf filter listed
```

### Testing AV1 Decoding

```bash
# Test 1: Check if you can decode AV1 (no encoding)
ffmpeg -i /path/to/av1_video.mp4 -f null -
# Should complete without "Missing Sequence Header" errors

# Test 2: Convert AV1 to H.264 using dav1d
ffmpeg -i av1_input.mp4 -c:v libx264 -crf 20 h264_output.mp4

# Test 3: Convert AV1 to H.264 using NVIDIA GPU
ffmpeg -hwaccel cuda -c:v av1_cuvid -i av1_input.mp4 \
       -c:v h264_nvenc -preset p1 -crf 20 h264_output.mp4

# Test 4: Encode to AV1 with SVT-AV1
ffmpeg -i input.mp4 -c:v libsvtav1 -crf 30 output_av1.mp4

# Test 5: Encode to AV1 with NVIDIA
ffmpeg -i input.mp4 -c:v av1_nvenc -preset p1 -crf 30 output_av1.mp4
```

### Troubleshooting

#### Issue: "ERROR: libdav1d >= 1.0.0 not found"
```bash
# Verify dav1d is installed
pkg-config --modversion dav1d

# If not found, rebuild dav1d and ensure PREFIX is correct
export PKG_CONFIG_PATH="/root/ffmpeg_build/lib/pkgconfig:$PKG_CONFIG_PATH"
```

#### Issue: "NVENC encoder not found"
```bash
# Check NVIDIA driver
nvidia-smi

# Verify CUDA installation
nvcc --version
ls /usr/local/cuda/include/ffnvcodec/

# Reinstall nv-codec-headers if needed
cd /root/ffmpeg_sources/nv-codec-headers
sudo make install
```

#### Issue: "Cannot load libcuda.so"
```bash
# Add CUDA to library path
echo "/usr/local/cuda/lib64" | sudo tee /etc/ld.so.conf.d/cuda.conf
sudo ldconfig
```

#### Issue: FFmpeg still shows old version
```bash
# Check which ffmpeg is being used
which ffmpeg
# Should be: /usr/local/bin/ffmpeg

# If wrong path, update PATH
export PATH="/usr/local/bin:$PATH"
echo 'export PATH="/usr/local/bin:$PATH"' >> ~/.bashrc
```

### Performance Comparison

| Decoder | 4K AV1 Video (10 sec) | CPU Usage | GPU Usage |
|---------|----------------------|-----------|-----------|
| Old (broken) | ❌ Failed | N/A | N/A |
| **libdav1d** | ✅ ~3-5 sec | ~80% | 0% |
| **av1_cuvid** | ✅ ~0.5-1 sec | ~10% | ~40% |

| Encoder | 1080p Video (10 sec) | Time | Quality |
|---------|---------------------|------|---------|
| **libx264** (CPU) | ~15 sec | Good | Excellent |
| **h264_nvenc** (GPU) | ~2 sec | Excellent | Very Good |
| **libsvtav1** (CPU) | ~45 sec | Poor | Excellent |
| **av1_nvenc** (GPU) | ~5 sec | Good | Very Good |

### After Installation

1. **Restart Scoring Server**:
   ```bash
   pm2 restart scoring-server
   ```

2. **Monitor Logs**:
   ```bash
   pm2 logs scoring-server | grep -E "AV1|conversion|codec"
   ```

3. **Expected Behavior**:
   - AV1 videos now convert successfully using `libdav1d`
   - Conversion time: 2-5 seconds (vs failing before)
   - VMAF scores calculated correctly
   - No more "Missing Sequence Header" errors

### Build Time Estimates

- **dav1d**: ~1-2 minutes
- **SVT-AV1**: ~3-5 minutes
- **libvmaf**: ~2-3 minutes
- **FFmpeg**: ~10-20 minutes
- **Total**: ~15-30 minutes

### Disk Space Required

- Source files: ~500 MB
- Build artifacts: ~1 GB
- Final installation: ~200 MB
- **Total**: ~1.7 GB

### Cleanup After Installation

```bash
# Optional: Remove source files to save space
rm -rf /root/ffmpeg_sources

# Keep build directory for future rebuilds
# rm -rf /root/ffmpeg_build  # Don't delete this!
```

### References

- dav1d: https://code.videolan.org/videolan/dav1d
- SVT-AV1: https://gitlab.com/AOMediaCodec/SVT-AV1
- FFmpeg Compilation Guide: https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu
- NVIDIA Video Codec SDK: https://developer.nvidia.com/video-codec-sdk
