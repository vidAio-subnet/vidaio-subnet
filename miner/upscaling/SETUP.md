```sh
# nvidia-container-toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Build and start the Video2X worker
docker compose -f miner/docker-compose.yml --profile upscaling-video2x up --build upscaling-video2x
```

The Video2X worker image builds Video2X from source and installs it into the
same container as the FastAPI service. It no longer requires a pre-pulled
upstream Video2X release image or access to the Docker daemon socket.

The default encoder is `av1_nvenc`. The service passes
`VIDEO2X_COMMON_ENCODER_ARGS=--pix-fmt=yuv420p,--max-b-frames=0` as Video2X
encoder arguments and `VIDEO2X_ENCODER_OPTIONS=preset=p4,cq={cq},profile=main`
as codec-specific FFmpeg AVOptions. Request-supplied codecs are ignored unless
`VIDEO2X_ALLOW_REQUEST_CODEC=true`.
