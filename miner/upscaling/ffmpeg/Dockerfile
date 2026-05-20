FROM python:3.12-slim

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    software-properties-common curl build-essential cmake ninja-build nasm g++ pkg-config git wget unzip perl \
    libx264-dev libx265-dev libvpx-dev nvidia-cuda-dev nvidia-cuda-toolkit \
    && rm -rf /var/lib/apt/lists/*

# Install nv-codec-headers (pinned commit)
RUN wget -q https://github.com/FFmpeg/nv-codec-headers/archive/876af32a202d0de83bd1d36fe74ee0f7fcf86b0d.zip \
    && unzip -q 876af32a202d0de83bd1d36fe74ee0f7fcf86b0d.zip \
    && cd nv-codec-headers-876af32a202d0de83bd1d36fe74ee0f7fcf86b0d \
    && make && make install \
    && cd .. && rm -rf nv-codec-headers-* 876af32a202d0de83bd1d36fe74ee0f7fcf86b0d.zip

# Build dav1d v1.4.2
RUN pip install --no-cache-dir meson==1.5.1 \
    && git clone --branch 1.4.2 --depth 1 https://code.videolan.org/videolan/dav1d.git \
    && cd dav1d && mkdir build && cd build \
    && meson setup .. --default-library=static --buildtype release \
    && ninja && ninja install && ldconfig \
    && cd ../.. && rm -rf dav1d

# Build ffmpeg (pinned commit 33b215d) with CUDA + NVENC + NVDEC + x264 + x265 + dav1d
RUN export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:/usr/local/lib/x86_64-linux-gnu/pkgconfig:$PKG_CONFIG_PATH" \
    && git init ffmpeg && cd ffmpeg \
    && git remote add origin https://github.com/FFmpeg/FFmpeg.git \
    && git fetch --depth 1 origin 33b215d1554a14e87416a24f8e6034312e629af7 \
    && git checkout FETCH_HEAD \
    && ./configure --enable-gpl --enable-libx264 --enable-libx265 --enable-libvpx --enable-libdav1d --enable-nonfree --enable-cuda-nvcc --enable-nvenc --enable-nvdec --extra-cflags="-I/usr/local/cuda/include" --extra-ldflags="-L/usr/local/cuda/lib64" --disable-static --enable-shared --nvccflags="-gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 -O2" \
    && make -j$(nproc) && make install && ldconfig \
    && cd .. && rm -rf ffmpeg

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
RUN mkdir -p /tmp/organic-proxy

EXPOSE 8003
CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8003"]
