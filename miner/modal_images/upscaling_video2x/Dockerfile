FROM ubuntu:24.04 AS video2x-builder

ARG VIDEO2X_VERSION=6.4.0
ARG VIDEO2X_COMMIT=a96bda9b4d79616cc6b71b94e6945146b5b4d509
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
    && git clone --depth 1 --branch "${VIDEO2X_VERSION}" --recurse-submodules --shallow-submodules \
       https://github.com/k4yt3x/video2x.git /src/video2x \
    && cd /src/video2x \
    && test "$(git rev-parse HEAD)" = "${VIDEO2X_COMMIT}" \
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
ENV VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
ENV PATH="/venv/bin:${PATH}"

COPY --from=video2x-builder /src/video2x/video2x-linux-ubuntu-amd64.deb /tmp/video2x.deb

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    ffmpeg \
    libegl1 \
    python-is-python3 \
    python3 \
    python3-venv \
    vulkan-tools \
    /tmp/video2x.deb \
    && rm -f /tmp/video2x.deb \
    && python3 -m venv /venv \
    && mkdir -p /usr/share/vulkan/icd.d \
    && printf '%s\n' \
       '{' \
       '    "file_format_version": "1.0.0",' \
       '    "ICD": {' \
       '        "library_path": "libEGL_nvidia.so.0",' \
       '        "api_version": "1.3.0"' \
       '    }' \
       '}' > /usr/share/vulkan/icd.d/nvidia_icd.json \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN mkdir -p /tmp/organic-proxy
