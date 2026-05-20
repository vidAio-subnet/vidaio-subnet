# Running Miner

A high-performance decentralized video processing miner that forwards validator requests to containerized video services. The current miner framework supports:

- **Video2X upscaling** through the `upscaling-video2x` Docker Compose profile.
- **FFmpeg upscaling** through the `upscaling-ffmpeg` Docker Compose profile.
- **FFmpeg compression** through the `compression` Docker Compose service.

`neurons/miner.py` no longer needs the old PM2-managed `services/upscaling/server.py` or `services/compress/server.py` workers. It forwards directly to the service container ports.

---

## Machine Requirements

To achieve optimal results, we recommend the following setup:

- **Operating System**: Ubuntu 24.04 LTS or higher  
  [Learn more about Ubuntu 24.04 LTS](https://ubuntu.com/blog/tag/ubuntu-24-04-lts)
- **GPU**: NVIDIA RTX 4090 or higher
- **VRAM**: At least 16 GB per GPU
- **Docker** and **Docker Compose**
- **NVIDIA drivers** and `nvidia-container-toolkit`
- **Python**: Version 3.10 or higher for running the miner process

---

## Install System Dependencies

Install common host dependencies:

```bash
sudo apt update
sudo apt install -y git curl npm python3 python3-venv python3-pip
```

Install Docker and Docker Compose using the official Docker instructions for your host, then install NVIDIA Container Toolkit so Compose services can access the GPU.

Verify Docker can see the GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

---

## Install PM2

PM2 is optional for the processing containers, but useful for managing the miner process itself.

```bash
sudo npm install pm2 -g
pm2 update
```

---

## Install Project Dependencies

### 1. Clone the Repository

```bash
git clone https://github.com/vidaio-subnet/vidaio-subnet.git
cd vidaio-subnet
```

### 2. Set Up a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install the Package

```bash
pip install -e .
```

---

## Configure Storage

The container services download validator payload URLs, process videos locally, upload outputs to S3-compatible storage, and return presigned URLs to the miner.

Create a miner service environment file:

```bash
cp miner/.env.template miner/.env
```

Edit `miner/.env` with your storage credentials:

```env
COMPOSE_PROJECT_NAME=miner
ORGANIC_PROXY_STORAGE_PROVIDER=backblaze
ORGANIC_PROXY_STORAGE_S3_ACCESS_KEY_ID=your-access-key
ORGANIC_PROXY_STORAGE_S3_SECRET_ACCESS_KEY=your-secret-key
ORGANIC_PROXY_STORAGE_S3_REGION=us-east-005
ORGANIC_PROXY_STORAGE_S3_BUCKET_NAME=your-bucket-name
ORGANIC_PROXY_STORAGE_S3_ENDPOINT_URL=https://s3.us-east-005.backblazeb2.com
```

For AWS S3, use the same variable names and leave `ORGANIC_PROXY_STORAGE_S3_ENDPOINT_URL` blank.

---

## Start Processing Services

Run Docker Compose commands from the `miner` directory:

```bash
cd miner
```

### Option A: Video2X Upscaling Miner

Use this profile if you want Video2X-based upscaling. Complete the host/container setup first:

```bash
cat upscaling/SETUP.md 
```

After running all the commands listed above, start the Video2X upscaling service:

```bash
docker compose --profile upscaling-video2x up -d upscaling-video2x
```

The service is exposed on `http://localhost:8003`.

### Option B: FFmpeg Upscaling Miner

Use this profile if you want FFmpeg-based upscaling instead of Video2X:

```bash
docker compose --profile upscaling-ffmpeg up -d upscaling-ffmpeg
```

The FFmpeg upscaling service is exposed on `http://localhost:8005`.

Before starting `neurons/miner.py`, point the miner at that port:

```bash
export MINER_UPSCALING_SERVICE_URL=http://localhost:8005
```

### Option C: Compression Miner

Start the compression service:

```bash
docker compose up -d compression
```

The compression service is exposed on `http://localhost:8004`.

Return to the repository root before launching the miner:

```bash
cd ..
```

---

## Miner Forwarding Configuration

`neurons/miner.py` forwards requests to these service URLs:

```bash
export MINER_UPSCALING_SERVICE_URL=http://localhost:8003
export MINER_COMPRESSION_SERVICE_URL=http://localhost:8004
```

Defaults:

- `MINER_UPSCALING_SERVICE_URL` defaults to `http://localhost:8003` for Video2X upscaling.
- `MINER_COMPRESSION_SERVICE_URL` defaults to `http://localhost:8004`.
- For FFmpeg upscaling, override `MINER_UPSCALING_SERVICE_URL` to `http://localhost:8005`.

Useful service checks:

```bash
curl -sf http://localhost:8003/health
curl -sf http://localhost:8004/health
curl -sf http://localhost:8005/health
```

Only the services you started need to pass health checks.

---

## Run the Miner

Start the miner from the repository root:

```bash
python3 neurons/miner.py \
  --wallet.name [Your_Wallet_Name] \
  --wallet.hotkey [Your_Hotkey_Name] \
  --subtensor.network finney \
  --netuid 85 \
  --axon.port [port] \
  --logging.debug
```

### Run the Miner with PM2

Export any `MINER_*_SERVICE_URL` overrides in the same shell before starting PM2. If you change those variables later, restart with `--update-env`.

```bash
pm2 start "python3 neurons/miner.py --wallet.name [Your_Wallet_Name] --wallet.hotkey [Your_Hotkey_Name] --subtensor.network finney --netuid 85 --axon.port [port] --logging.debug" --name video-miner
```

### Parameters

- `--wallet.name`: Replace `[Your_Wallet_Name]` with your wallet name.
- `--wallet.hotkey`: Replace `[Your_Hotkey_Name]` with your hotkey name.
- `--subtensor.network`: Specify the target network, usually `finney`.
- `--netuid`: Specify the network UID, currently `85`.
- `--axon.port`: Replace `[port]` with the desired public axon port.
- `--logging.debug`: Enables debug-level logging.

---

## Managing Services

Processing containers:

```bash
cd miner
docker compose ps
docker compose logs -f upscaling-video2x
docker compose logs -f upscaling-ffmpeg
docker compose logs -f compression
docker compose down
```

Miner process:

```bash
pm2 logs video-miner
pm2 restart video-miner --update-env
pm2 stop video-miner
```

---

## Additional Notes

- Run either `upscaling-video2x` or `upscaling-ffmpeg` for an upscaling miner. They expose different host ports but both implement `/upscale`.
- Run `compression` for a compression miner. It implements `/compress`.
- The old PM2 service scripts for upscaling and compression are redundant with this container framework.
- For Video2X-specific preparation, use `miner/upscaling/SETUP.md`.
- For FFmpeg upscaling build/runtime notes, use `miner/upscaling/ffmpeg/SETUP.md`.
