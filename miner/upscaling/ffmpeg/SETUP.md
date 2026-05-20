# FFmpeg Upscaling Setup

This variant uses FFmpeg directly (CUDA/NVENC capable build) for video upscaling.

## Notes
- The Dockerfile pins `nv-codec-headers`, `dav1d`, and FFmpeg commit `33b215d1554a14e87416a24f8e6034312e629af7`.
- Requires NVIDIA GPU runtime on the host.

## Run
```bash
docker compose --profile upscaling-ffmpeg up -d upscaling-ffmpeg
```
