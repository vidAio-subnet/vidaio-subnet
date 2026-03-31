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

### 2. Setup chute
Refer to Chute setup section below


### 3. Configure Environment

Once the chute is warmed up, set the relevant environment variables. Your chutes slug is formatted as `username-chute-name` if your chute is uploaded as `username/chute-name` as of writing, check the API tab on the private chute on chutes.ai to extract chutes slug.

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

### 4. Configure task type

Miners declare which task they handle via the `TaskWarrantProtocol`. Set the `warrant_task` variable in `neurons/miner.py`:

| Value | Task | Description |
|-------|------|-------------|
| `TaskType.COMPRESSION` | Compression | Reduce file size while preserving quality (VMAF-aware) |
| `TaskType.UPSCALING` | Upscaling | Increase resolution (SD->HD, HD->4K, etc.) |

Miners also declare their max content length via `LengthCheckProtocol` (currently 5 or 10 seconds).


### 5. Run the Miner

```bash
python neurons/miner.py \
    --netuid 85 \
    --wallet.name <your-coldkey> \
    --wallet.hotkey <your-hotkey> \
    --subtensor.network finney \
    --axon.port 8091
```

The miner will register its axon on the network and begin listening for validator requests.


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
  from_base: parachutes/python:3.12-cu13
  run_command:
    - uv pip install --upgrade setuptools wheel
    - uv pip install huggingface_hub==0.19.4
    - uv apt-get update && apt-get install -y ffmpeg
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
  from_base: parachutes/python:3.12-cu13
  set_user: root
  run_command:
    - apt-get update && apt-get install -y ffmpeg cargo
    - cargo install just --version=1.39.0
    - git clone --recurse-submodules https://github.com/vidAio-subnet/video2x /tmp/video2x && cd /tmp/video2x && export PATH="$HOME/.cargo/bin:$PATH" && just ubuntu2404 && dpkg -i video2x-linux-ubuntu-amd64.deb && rm -rf /tmp/video2x
  set_user: chutes
  run_command:
    - uv pip install --upgrade setuptools wheel
    - uv pip install huggingface_hub==0.19.4 minio
  set_user: chutes
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

### Local Testing

You can test your chute locally before deploying. This does **not** require HuggingFace or Chutes credentials -- it builds a Docker image from your local files.

1. Copy the chute template and your `chute_config.yml` into a test directory:
   ```bash
   mkdir -p local_dev
   # Compression
   cp chutes/compression/vidaio_compression_chute.py.j2 local_dev/my_chute.py
   cp example_miners/compression/chute_config.yml local_dev/
   cp example_miners/compression/miner.py local_dev/

   # Or upscaling
   cp chutes/upscaling/vidaio_upscaling_chute.py.j2 local_dev/my_chute.py
   cp example_miners/upscaling/chute_config.yml local_dev/
   cp example_miners/upscaling/miner.py local_dev/

   cd local_dev
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
   chutes build my_chute:chute --local --include-cwd
   cd ..
   ```

3. Run the container with `VIDAIO_LOCAL_TEST=1` and start the chute:
   ```bash
   docker run --gpus all -p 8000:8000 -e NVIDIA_DRIVER_CAPABILITIES=all \
   -e CHUTES_EXECUTION_CONTEXT=REMOTE \
   -e VIDAIO_LOCAL_TEST=1 -it my-test-chute:latest /bin/bash
   # Inside the container:
   chutes run my_chute:chute --dev --debug
   ```

   `VIDAIO_LOCAL_TEST=1` enables local testing mode:
   - Loads `miner.py` from the local directory instead of downloading from HuggingFace
   - Downloads and uploads videos directly instead of via the chutes proxy
   - `upload_url` is optional -- if omitted or upload fails, the result video is returned inline as base64 in `output_video_b64`

4. Test the endpoints from another terminal using the test script:
   ```bash
   # With S3 upload (set BUCKET_* env vars first)
   python scripts/test_chute_local.py \
       --task compression \
       --video-url https://example.com/video.mp4

   python scripts/test_chute_local.py \
       --task upscaling \
       --video-url https://example.com/video.mp4

   # Without S3 (result returned as base64, saved to local file)
   python scripts/test_chute_local.py \
       --task compression \
       --video-url https://example.com/video.mp4 \
       --no-s3
   ```

   The script handles health checks, presigned URL generation, payload construction, and displays the final S3 download link. You can also test manually with curl:
   ```bash
   curl -X POST http://localhost:8000/health -d '{}'

   curl -X POST http://localhost:8000/process -d '{
     "video_url": "https://example.com/video.mp4",
     "vmaf_threshold": 90.0,
     "target_codec": "h264",
     "codec_mode": "CRF",
     "target_bitrate": 10.0
   }'

   # Upscaling example
   curl -X POST http://localhost:8000/process -d '{
     "video_url": "https://example.com/video.mp4",
     "task_type": "SD2HD"
   }'
   ```

### Deploying Your Chute

Once you are certain that the Chute image and container builds locally and functions as intended, you can deploy the Chute remotely.

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

# Custom HF repo name and chute name (instead of auto-generated defaults)
python scripts/deploy_chute.py \
    --task compression \
    --hf-username your-hf-username \
    --hf-token hf_xxx \
    --chutes-api-key cpk_xxx \
    --chutes-username your-chutes-username \
    --model-path example_miners/compression \
    --uploaded-hf-repo-name your-hf-username/my-custom-repo \
    --uploaded-chute-name my-custom-chute

# Deploy without warmup
python scripts/deploy_chute.py --task compression ... --no-warmup
```

The script prints the chute slug on completion. You can also check manually:
```bash
chutes chutes list
chutes chutes get <chute-name>
```

### Testing a Deployed Chute

Use the test script with `--remote` to query a deployed chute on chutes.ai:
```bash
export CHUTES_API_KEY="cpk_xxx"
export CHUTE_SLUG="your-chutes-username/your-chute-name"

# S3 env vars are required in remote mode (the chute uploads results there)
export BUCKET_TYPE="backblaze"
export BUCKET_COMPATIBLE_ENDPOINT="s3.us-west-004.backblazeb2.com"
export BUCKET_COMPATIBLE_ACCESS_KEY="your-access-key"
export BUCKET_COMPATIBLE_SECRET_KEY="your-secret-key"
export BUCKET_NAME="your-bucket-name"

python scripts/test_chute_local.py \
    --task compression \
    --video-url https://example.com/video.mp4 \
    --remote

python scripts/test_chute_local.py \
    --task upscaling \
    --video-url https://example.com/video.mp4 \
    --remote
```

The script generates a presigned PUT URL, sends the request to `https://<CHUTE_SLUG>.chutes.ai/process` with Bearer auth, and prints a presigned GET URL for the result.

### Miner Environment

Set the chute slug in your miner environment so the subnet knows which chute to call:
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