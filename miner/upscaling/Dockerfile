FROM ubuntu:24.04 AS video2x-builder

ARG VIDEO2X_REF=6.4.0
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.cargo/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cargo \
    git \
    && cargo install just \
    && git clone --depth 1 --branch "${VIDEO2X_REF}" --recurse-submodules --shallow-submodules \
       https://github.com/k4yt3x/video2x.git /src/video2x \
    && cd /src/video2x \
    && just ubuntu2404 \
    && rm -rf /var/lib/apt/lists/*


FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=video,compute,utility
ENV VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json:/usr/share/vulkan/icd.d/radeon_icd.x86_64.json:/usr/share/vulkan/icd.d/intel_icd.x86_64.json:/usr/share/vulkan/icd.d/lvp_icd.x86_64.json
ENV PATH="/venv/bin:${PATH}"

COPY --from=video2x-builder /src/video2x/video2x-linux-ubuntu-amd64.deb /tmp/video2x.deb

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    ffmpeg \
    mesa-vulkan-drivers \
    python3 \
    python3-venv \
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
