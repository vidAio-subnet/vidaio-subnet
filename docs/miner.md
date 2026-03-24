# Miner Guide

This guide covers everything needed to run a miner on the Vidaio subnet (SN85). Miners process video compression and upscaling tasks dispatched by validators, using [Chutes](https://chutes.ai) for GPU-accelerated execution.

## Architecture Overview

```
Validator                        Miner Node                        Chutes (GPU)
   |                                |                                  |
   |-- VideoCompression/Upscaling ->|                                  |
   |   Protocol (Bittensor Axon)    |                                  |
   |                                |-- presigned PUT URL (S3/B2) ---->|
   |                                |-- POST /process ---------------->|
   |                                |   (video_url, params, upload_url)|
   |                                |                                  |-- download video
   |                                |                                  |-- miner.process_video()
   |                                |                                  |-- upload result to PUT URL
   |                                |<--------- { success, url } ------|
   |                                |                                  |
   |                                |-- presigned GET URL              |
   |<-- optimized_video_url --------|                                  |
```

1. Validator sends a synapse (compression or upscaling request) to the miner via Bittensor.
2. Miner generates a presigned PUT URL from its S3-compatible bucket.
3. Miner calls its Chute's `/process` endpoint, passing the video URL and upload URL.
4. The Chute downloads the video, runs `Miner.process_video()`, and uploads the result.
5. Miner returns a presigned GET URL to the validator so it can download and score the result.

## Prerequisites

- Python 3.10+
- A registered Bittensor wallet on subnet 85
- A [Chutes](https://chutes.ai) account with API key
- A [Hugging Face](https://huggingface.co) account (to host your miner code)
- An S3-compatible storage bucket (Backblaze B2, AWS S3, Cloudflare R2, or Hippius)

## Quick Start

### 1. Install Dependencies

```bash
git clone https://github.com/score-technologies/vidaio-subnet.git
cd vidaio-subnet
pip install -e .
```

### 2. Configure Environment

The miner reads configuration from environment variables. Storage config uses `BUCKET_*` env vars loaded via `dotenv`, and Chutes config uses pydantic-settings with `__` as the nested delimiter.

```bash
# --- Chutes ---
# These map to CONFIG.chutes.* via pydantic-settings (env_nested_delimiter="__")
export CHUTES__API_KEY="your-chutes-api-key"
export CHUTES__COMPRESSION_SLUG="your-compression-chute-slug"
export CHUTES__UPSCALING_SLUG="your-upscaling-chute-slug"

# --- S3-Compatible Storage ---
# These are read directly via os.getenv() in StorageConfig
export BUCKET_TYPE="backblaze"               # backblaze | amazon_s3 | cloudflare | hippius
export BUCKET_COMPATIBLE_ENDPOINT="s3.us-west-004.backblazeb2.com"
export BUCKET_COMPATIBLE_ACCESS_KEY="your-access-key"
export BUCKET_COMPATIBLE_SECRET_KEY="your-secret-key"
export BUCKET_NAME="your-bucket-name"
```

### 3. Run the Miner

```bash
python neurons/miner.py \
    --netuid 85 \
    --wallet.name <your-coldkey> \
    --wallet.hotkey <your-hotkey> \
    --subtensor.network finney \
    --axon.port 8091
```

The miner will register its axon on the network and begin listening for validator requests.

## Task Types

Miners declare which task they handle via the `TaskWarrantProtocol`. Set the `warrant_task` variable in `neurons/miner.py`:

| Value | Task | Description |
|-------|------|-------------|
| `TaskType.COMPRESSION` | Compression | Reduce file size while preserving quality (VMAF-aware) |
| `TaskType.UPSCALING` | Upscaling | Increase resolution (SD->HD, HD->4K, etc.) |

Miners also declare their max content length via `LengthCheckProtocol` (currently 5 or 10 seconds).

## Chute Setup

Each task type requires a separate Chute deployed on the Chutes platform. Your Chute code lives in a Hugging Face repo.

### Required Files in Your HF Repo

| File | Purpose |
|------|---------|
| `miner.py` | Your `Miner` class with `process_video()` logic |
| `chute_config.yml` | Machine specs, image config, scaling parameters |

> All code must be self-contained in `miner.py`. Network access is disabled inside Chutes -- you cannot download weights or external assets at runtime. Bundle everything in the repo.

### Miner Class Interface

Your `miner.py` must export a class named `Miner` with the following interface:

**Compression:**
```python
class Miner:
    def __init__(self, path_hf_repo: Path) -> None:
        """path_hf_repo is the local path to the downloaded HF repo."""
        ...

    def process_video(
        self,
        input_path: Path,
        vmaf_threshold: float,    # Target VMAF quality (0-100)
        target_codec: str,        # "av1", "hevc", "h264", "vp9"
        codec_mode: str,          # "CRF", "CBR", "VBR"
        target_bitrate: float,    # Mbps (used for CBR/VBR modes)
    ) -> Path:
        """Return the path to the compressed output file."""
        ...
```

**Upscaling:**
```python
class Miner:
    def __init__(self, path_hf_repo: Path) -> None:
        ...

    def process_video(self, input_path: Path, task_type: str) -> Path:
        """
        task_type is one of: "HD24K", "SD2HD", "SD24K", "4K28K"
        Return the path to the upscaled output file.
        """
        ...
```

### Example `chute_config.yml`

**Compression:**
```yaml
Image:
  from_base: parachutes/python:3.12
  run_command:
    - pip install --upgrade setuptools wheel
    - pip install huggingface_hub==0.19.4
    - apt-get update && apt-get install -y ffmpeg
  set_workdir: /app

NodeSelector:
  gpu_count: 1
  min_vram_gb_per_gpu: 16

Chute:
  shutdown_after_seconds: 600
  concurrency: 2
  max_instances: 5
  scaling_threshold: 0.5
```

**Upscaling** (more resource-intensive, lower concurrency):
```yaml
Image:
  from_base: parachutes/python:3.12
  run_command:
    - pip install --upgrade setuptools wheel
    - pip install huggingface_hub==0.19.4
    - apt-get update && apt-get install -y ffmpeg
    - pip install video2x
  set_workdir: /app

NodeSelector:
  gpu_count: 1
  min_vram_gb_per_gpu: 16

Chute:
  shutdown_after_seconds: 600
  concurrency: 1
  max_instances: 3
  scaling_threshold: 0.5
```

### Local Testing (Optional)

You can test your chute locally before deploying. This does **not** require HuggingFace or Chutes credentials -- it builds a Docker image from your local files.

1. Copy the chute template and your `chute_config.yml` into the same directory:
   ```bash
   # Compression
   cp chutes/compression/vidaio_compression_chute.py.j2 my_chute.py
   cp example_miners/compression/chute_config.yml .

   # Or upscaling
   cp chutes/upscaling/vidaio_upscaling_chute.py.j2 my_chute.py
   cp example_miners/upscaling/chute_config.yml .
   ```

   Edit the top of `my_chute.py` -- set any placeholder values (these are only used to name the local image, not for auth):
   ```python
   HF_REPO_NAME = "local/test"
   HF_REPO_REVISION = "main"
   CHUTES_USERNAME = "local"
   CHUTE_NAME = "my-test-chute"
   ```

   The template will load `chute_config.yml` from the local directory, skipping the HuggingFace download.

2. Build the Docker image locally:
   ```bash
   chutes build my_chute:chute --local
   ```

3. Run the container and start the chute:
   ```bash
   docker run -p 8000:8000 -e CHUTES_EXECUTION_CONTEXT=REMOTE -it <image-name> /bin/bash
   # Inside the container:
   chutes run my_chute:chute --dev --debug
   ```

4. Test the endpoints from another terminal:
   ```bash
   curl -X POST http://localhost:8000/health -d '{}'

   # Compression example
   curl -X POST http://localhost:8000/process -d '{
     "video_url": "https://example.com/video.mp4",
     "vmaf_threshold": 90.0,
     "target_codec": "av1",
     "codec_mode": "CRF",
     "target_bitrate": 10.0,
     "upload_url": "https://example.com/presigned-put"
   }'

   # Upscaling example
   curl -X POST http://localhost:8000/process -d '{
     "video_url": "https://example.com/video.mp4",
     "task_type": "SD2HD",
     "upload_url": "https://example.com/presigned-put"
   }'
   ```

### Deploying Your Chute

The deploy script handles the full workflow: uploading your miner to HuggingFace, rendering the chute template, building, deploying, and warming up.

```bash
# Deploy a compression chute
python scripts/deploy_chute.py \
    --task compression \
    --hf-username your-hf-username \
    --hf-token hf_xxx \
    --chutes-api-key cpk_xxx \
    --chutes-username your-chutes-username \
    --model-path example_miners/compression

# Deploy an upscaling chute
python scripts/deploy_chute.py \
    --task upscaling \
    --hf-username your-hf-username \
    --hf-token hf_xxx \
    --chutes-api-key cpk_xxx \
    --chutes-username your-chutes-username \
    --model-path example_miners/upscaling

# Redeploy from existing HF repo (skip upload)
python scripts/deploy_chute.py --task compression ... --no-upload

# Deploy without warmup
python scripts/deploy_chute.py --task compression ... --no-warmup
```

The script prints the chute slug on completion. You can also check manually:
```bash
chutes chutes list
chutes chutes get <chute-name>

# Test live endpoint
curl -X POST https://<YOUR-CHUTE-SLUG>.chutes.ai/health \
  -d '{}' \
  -H "Authorization: Bearer $CHUTES_API_KEY"
```

Set the chute slug in your miner environment:
```bash
export CHUTES__COMPRESSION_SLUG="your-slug"
# or
export CHUTES__UPSCALING_SLUG="your-slug"
```

## Example Miners

Reference implementations are provided in `example_miners/`. These are intentionally minimal -- miners are expected to significantly improve upon them.

- **`example_miners/compression/miner.py`** -- FFmpeg-based compression with a simple VMAF-to-CRF mapping.
- **`example_miners/upscaling/miner.py`** -- video2x + RealESRGAN upscaling with frame padding to avoid artifacts.

## Scoring

Validators score miner outputs using:

- **VMAF** (Video Multi-Method Assessment Fusion) for perceptual quality.
- **PieAPP** for perceptual image error assessment on upscaling tasks.
- **Compression ratio** relative to the original file size.

Higher quality and better compression ratios yield higher scores. See `vidaio_subnet_core/configs/score.py` for threshold details.

## Troubleshooting

| Symptom | Likely Cause |
|---------|-------------|
| `Chute returned error` in logs | Your `miner.py` crashed inside the Chute. Test locally first. |
| `Chute HTTP error: 401` | Invalid or missing `CHUTES__API_KEY`. |
| `Chute request timed out` | Processing exceeded 600s (default). Optimize your pipeline or increase `CHUTES__REQUEST_TIMEOUT`. |
| `Failed to generate presigned URL` | Bucket credentials or endpoint misconfigured. |
| `Your Miner is not registered` | Wallet hotkey not registered on subnet 85. Run `btcli register`. |
| Chute stays cold | Check `chute_config.yml` node requirements match available Chutes hardware. |
