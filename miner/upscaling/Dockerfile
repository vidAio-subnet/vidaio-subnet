FROM ubuntu:24.04 AS video2x-builder

ARG VIDEO2X_REF=6.4.0
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    git \
    glslang-tools \
    libavcodec-dev \
    libavdevice-dev \
    libavfilter-dev \
    libavformat-dev \
    libavutil-dev \
    libboost-program-options-dev \
    libomp-dev \
    libswscale-dev \
    libvulkan-dev \
    ninja-build \
    pkg-config \
    && git clone --depth 1 --branch "${VIDEO2X_REF}" --recurse-submodules --shallow-submodules \
       https://github.com/k4yt3x/video2x.git /src/video2x \
    && cd /src/video2x \
    && cmake -G Ninja -B build -S . \
       -DVIDEO2X_USE_EXTERNAL_NCNN=OFF \
       -DVIDEO2X_USE_EXTERNAL_SPDLOG=OFF \
       -DCMAKE_CXX_COMPILER=g++ \
       -DCMAKE_BUILD_TYPE=Release \
       -DCMAKE_INSTALL_PREFIX=video2x-linux-ubuntu-amd64/usr \
    && cmake --build build --config Release --target install --parallel \
    && mkdir -p video2x-linux-ubuntu-amd64/DEBIAN \
    && cp packaging/debian/control.ubuntu2404 video2x-linux-ubuntu-amd64/DEBIAN/control \
    && dpkg-deb --build video2x-linux-ubuntu-amd64 \
    && rm -rf /var/lib/apt/lists/*


FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,video,compute,utility
ENV PATH="/venv/bin:${PATH}"

COPY --from=video2x-builder /src/video2x/video2x-linux-ubuntu-amd64.deb /tmp/video2x.deb

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    ffmpeg \
    mesa-vulkan-drivers \
    python3 \
    python3-venv \
    vulkan-tools \
    /tmp/video2x.deb \
    && rm -f /tmp/video2x.deb \
    && python3 -m venv /venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

RUN mkdir -p /tmp/organic-proxy

EXPOSE 8003
CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8003"]
