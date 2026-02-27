# Running Validator

---

## Machine Requirements

To achieve optimal results, we recommend the following setup:

- **Operating System**: Ubuntu 24.04 LTS or higher
  [Learn more about Ubuntu 24.04 LTS](https://ubuntu.com/blog/tag/ubuntu-24-04-lts)
- **GPU**: NVIDIA RTX A6000 with 48GB VRAM and 30 CPU Cores

---

## Bootstrap System Dependencies

The `bootstrap.sh` script at the repository root automates the installation of core system-level dependencies:

- **NVIDIA GPU drivers** (default version 535)
- **Docker** and the **NVIDIA Container Toolkit**
- **Python 3.11**
- Base utilities (git, curl, wget, etc.)

Run the script **as root** with the `-E` flag to preserve environment variables:

```bash
sudo -E ./bootstrap.sh
```

> **Note:** If you encounter `dpkg` lock issues (common on platforms like TensorDock), wait ~15 minutes and re-run the script. The script will **automatically reboot** the machine if a new NVIDIA driver was installed.

#### Optional environment variables

| Variable | Default | Description |
|---|---|---|
| `NVIDIA_DRIVER_VERSION` | `535` | NVIDIA driver version to install |

Once the bootstrap completes (and any reboot finishes), proceed with the rest of this guide.

---

## Install PM2 (Process Manager)

**PM2** is used to manage and monitor the validator process. If you haven’t installed PM2 yet, follow these steps:

1. Install `npm` and PM2:
   ```bash
   sudo apt update
   sudo apt install npm -y
   sudo npm install pm2 -g
   pm2 update
   ```

2. For more details, refer to the [PM2 Documentation](https://pm2.io/docs/runtime/guide/installation/).

---

## Install Redis

1. Install 'redis'
   ```bash
   sudo apt update
   sudo apt install redis-server
   sudo systemctl start redis
   sudo systemctl enable redis-server
   sudo systemctl status redis
   ```


## Install Project Dependencies

### Prerequisites

- **Python**: Version 3.10 or higher
- **pip**: Python package manager
- **virtualenv** (optional): For dependency isolation

---

### 1. Clone the Repository

Clone the project repository to your local machine:
```bash
git clone https://github.com/vidaio-subnet/vidaio-subnet.git
cd vidaio-subnet
```

---

### 2. Set Up a Virtual Environment (Recommended)

Create and activate a virtual environment to isolate project dependencies:
```bash
python3 -m venv venv
source venv/bin/activate  
```

---

### 3. Install the Package and Dependencies

Install the project and its dependencies using `pip`:
```bash
pip install -e .
```

---

### 4. Configure Environment Variables

To configure environment variables, follow these steps:

1. Create a `.env` file in the project root directory by referencing the provided `.env.template` file:
   ```bash
   cp .env.template .env
   ```
2. Set up a bucket in cloud storage. The base miner code utilizes MinIO to connect with cloud storage services, so you'll need to prepare your bucket using a platform that supports MinIO integration, such as Backblaze. Alternatively, you can modify the code to suit your specific requirements. *IMPORTANT*: Note that currently the `region` of the storage is hardcoded, and must be adjusted in `vidaio_subnet_core/utilities/storage_client.py` for corresponding storage, such as AWS.
3. Add the required variables to the `.env` file. For example:
   ```env
   BUCKET_NAME="S3 buckent name"
   BUCKET_COMPATIBLE_ENDPOINT="S3 bucket endpoint"
   BUTKET_COMPATIBLE_ACCESS_KEY="S3 bucket personal access key"
   BUCKET_COMPATIBLE_SECRET_KEY="S3 bucket personal secret key"
   PEXELS_API_KEY="Your Pexels account api key"
   WANDB_API_KEY="Your WANDB account api key"
   ```

4. Ensure that the bucket is configured with the appropriate permissions to allow file uploads and enable public access for downloads via presigned URLs.

5. Create your Pexels API key and replace it. (https://www.pexels.com/)
6. Once the `.env` file is properly configured, the application will use the specified credentials for S3 bucket, Pexels and Wandb.


---

## Install FFMPEG

FFMPEG is required for processing video files. Install it using the following commands:
```bash
sudo apt update
sudo apt install ffmpeg -y
```

For more details, refer to the [FFMPEG Documentation](https://www.ffmpeg.org/download.html#build-linux).

---

## Install VMAF

To enable video quality validation, install **VMAF** by following the steps below to set up a clean virtual environment, install dependencies, and compile the tool.

---

Clone the VMAF repository into the working root directory of your `vidaio-subnet` package:

```bash
git clone https://github.com/vidAio-subnet/vmaf.git
cp vmaf_utils/Dockerfile vmaf/Dockerfile
cp vmaf_utils/Dockerfile.ffmpeg vmaf/Dockerfile.ffmpeg
cd vmaf
git stash && git reset --hard 332dde62838d91d8b5216e9822de58851f2fd64f && git stash apply
docker build -t vmaf .
docker build -t vmaf_ffmpeg:latest -f Dockerfile.ffmpeg .
```

---
## Running the Validator with PM2

To run the validator, use the following command:

```bash
pm2 start run.sh --name vidaio_v_autoupdater -- --wallet.name [Your_Wallet_Name] --wallet.hotkey [Your_Hotkey_Name] --subtensor.network finney --netuid 85 --axon.port [port] --logging.debug
```

### Parameters:
- **`--wallet.name`**: Replace `[Your_Wallet_Name]` with your wallet name.
- **`--wallet.hotkey`**: Replace `[Your_Hotkey_Name]` with your hotkey name.
- **`--subtensor.network`**: Specify the target network (e.g., `finney`).
- **`--netuid`**: Specify the network UID (e.g., `85`).
- **`--axon.port`**: Replace `[port]` with the desired port number.
- **`--logging.debug`**: Enables debug-level logging for detailed output.

---
