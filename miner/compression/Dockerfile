FROM nvidia/cuda:13.0.0-devel-ubuntu22.04

# Build dependencies for ffmpeg + NVENC/NVDEC + codec libraries
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential ninja-build nasm perl g++ cmake wget unzip \
    nvidia-cuda-dev nvidia-cuda-toolkit pkg-config \
    libx264-dev libx265-dev libvpx-dev \
    git python3-pip python3-venv curl \
    && rm -rf /var/lib/apt/lists/*

# Install meson (needed for dav1d build)
RUN pip3 install --no-cache-dir meson

# nv-codec-headers (NVENC/NVDEC API headers)
RUN wget https://github.com/FFmpeg/nv-codec-headers/archive/876af32a202d0de83bd1d36fe74ee0f7fcf86b0d.zip \
    && unzip 876af32a202d0de83bd1d36fe74ee0f7fcf86b0d.zip \
    && cd nv-codec-headers-876af32a202d0de83bd1d36fe74ee0f7fcf86b0d \
    && make && make install && cd .. \
    && rm -rf nv-codec-headers-* 876af32a202d0de83bd1d36fe74ee0f7fcf86b0d.zip

# dav1d (AV1 decoder)
RUN git clone --depth 1 --branch 1.4.2 https://code.videolan.org/videolan/dav1d.git \
    && cd dav1d && mkdir build && cd build \
    && meson setup .. --default-library=static --buildtype release \
    && ninja && ninja install && ldconfig \
    && cd ../.. && rm -rf dav1d

# SVT-AV1 (AV1 encoder for libsvtav1)
RUN git clone --depth 1 --branch v4.1.0 https://gitlab.com/AOMediaCodec/SVT-AV1.git \
    && cd SVT-AV1 && mkdir build && cd build \
    && cmake .. -G"Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
    && make -j$(nproc) && make install && ldconfig \
    && cd ../.. && rm -rf SVT-AV1

# Build ffmpeg (pinned commit 33b215d) with CUDA + x264 + x265 + dav1d + svtav1
RUN export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:/usr/local/lib/x86_64-linux-gnu/pkgconfig:$PKG_CONFIG_PATH" \
    && git init ffmpeg && cd ffmpeg \
    && git remote add origin https://github.com/FFmpeg/FFmpeg.git \
    && git fetch --depth 1 origin 33b215d1554a14e87416a24f8e6034312e629af7 \
    && git checkout FETCH_HEAD \
    && ./configure \
        --enable-gpl \
        --enable-libx264 \
        --enable-libx265 \
        --enable-libvpx \
        --enable-libdav1d \
        --enable-libsvtav1 \
        --enable-nonfree \
        --enable-cuda-nvcc \
        --enable-nvenc \
        --enable-nvdec \
        --extra-cflags="-I/usr/local/cuda/include" \
        --extra-ldflags="-L/usr/local/cuda/lib64" \
        --disable-static \
        --enable-shared \
        --nvccflags="-gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 -O2" \
    && make -j$(nproc) && make install && ldconfig \
    && cd .. && rm -rf ffmpeg

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8004
CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8004"]