# Running Miner

A high-performance decentralized video processing miner that leverages **Video2X** for AI-powered video upscaling. This guide provides detailed instructions to set up, configure, and run the miner effectively

---

## Machine Requirements

To achieve optimal results, we recommend the following setup:

- **Operating System**: Ubuntu 24.04 LTS or higher  
  [Learn more about Ubuntu 24.04 LTS](https://ubuntu.com/blog/tag/ubuntu-24-04-lts)
- **GPU**: NVIDIA RTX 4090 or higher  
  (Required for efficient video upscaling using Video2X)

---

## Install PM2 (Process Manager)

**PM2** is used to manage and monitor the miner process. If you haven’t installed PM2 yet, follow these steps:

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

2. Set up a bucket in cloud storage. The base miner code utilizes MinIO to connect with cloud storage services, so you'll need to prepare your bucket using a platform that supports MinIO integration, such as Backblaze. Alternatively, you can modify the code to suit your specific requirements.
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

5. Once the `.env` file is properly configured, the application will use the specified credentials for S3 bucket and Pexels.


---

## Install Video2X

The miner requires **Video2X** for AI-powered video upscaling. Follow the steps below to install and configure Video2X.

---

### Step 1: Install FFMPEG

FFMPEG is required for processing video files. Install it using the following commands:
```bash
sudo apt update
sudo apt install ffmpeg -y
```

For more details, refer to the [FFMPEG Documentation](https://www.ffmpeg.org/download.html#build-linux).

---

### Step 2: Install CUDA and NVCC

Ensure your CUDA drivers and NVCC (NVIDIA Compiler) are properly installed and configured to support GPU acceleration.

1. Verify your CUDA installation:
   ```bash
   nvcc --version
   ```

2. Ensure you CUDA driver is installed correctly
   <!-- Install or update the CUDA drivers if they are not already installed:
   ```bash
   sudo apt update
   sudo apt install nvidia-cuda-toolkit -y
   ``` -->

For more information, refer to the [CUDA Toolkit Installation Guide](https://developer.nvidia.com/cuda-toolkit).

---

### Step 3: Install Video2X

---

### Option 1: Install Video2X by downloading Debian Package:


1. **Download the Video2X `.deb` package**:
   ```bash
   wget -P services/upscaling/models https://github.com/k4yt3x/video2x/releases/download/6.3.1/video2x-linux-ubuntu2404-amd64.deb
   ```

2. **Install the package using `dpkg`**:
   ```bash
   sudo dpkg -i services/upscaling/models/video2x-linux-ubuntu2404-amd64.deb
   ```

3. **Resolve dependencies** (if any):
   ```bash
   sudo apt-get install -f
   ```

### Option 2: Install Video2X by Building Debian Package:

1. Install Cargo
Cargo is required to build the Video2X package:
```bash
sudo apt-get update
sudo apt-get install cargo -y
cargo install just --version=1.39.0
```

2. Clone the Video2X Repository
you can clone this repository within the current vidaio-subnet package
```bash
git clone --recurse-submodules https://github.com/vidAio-subnet/video2x
cd video2x
```

3. Build the Video2X Project
Before building, ensure `~/.cargo/bin` is included in your `PATH` environment variable:
```bash
export PATH="$HOME/.cargo/bin:$PATH"
```

Run the following command to build the package:
```bash
just ubuntu2404
```

Once the build is complete, the `.deb` package will be located in the current directory.

4. Install the Built Package
Install the `.deb` package using:
```bash
sudo dpkg -i video2x-linux-ubuntu-amd64.deb
```

---
For additional details, refer to the [Video2X Documentation](https://docs.video2x.org/building/linux.html).

## Running the Video Upscaling Endpoint

You can run the video upscaling endpoint using **PM2** to manage the process:

```bash
pm2 start "python services/upscaling/server.py" --name video-upscaler
```

### Notes:
- The `video-upscaler` process will handle video upscaling requests.
- Use the following PM2 commands to manage the process:
  - **View Logs**: `pm2 logs video-upscaler`
  - **Restart**: `pm2 restart video-upscaler`
  - **Stop**: `pm2 stop video-upscaler`

---

## Running the Video Compression Endpoint

You can also run the video Compression endpoint using **PM2** to manage the process:

```bash
pm2 start "python services/compress/server.py" --name video-compressor
```

---

## Running the file deletion process

You can run the file deletion process using **PM2** to manage the process:

```bash
pm2 start "python services/miner_utilities/file_deletion_server.py" --name video-deleter
```

## Running the Miner with PM2

To run the miner, use the following command:

```bash
pm2 start "python3 neurons/miner.py --wallet.name [Your_Wallet_Name] --wallet.hotkey [Your_Hotkey_Name] --subtensor.network finney --netuid 85 --axon.port [port] --logging.debug" --name video-miner
```

### Parameters:
- **`--wallet.name`**: Replace `[Your_Wallet_Name]` with your wallet name.
- **`--wallet.hotkey`**: Replace `[Your_Hotkey_Name]` with your hotkey name.
- **`--subtensor.network`**: Specify the target network (e.g., `finney`).
- **`--netuid`**: Specify the network UID (e.g., `292`).
- **`--axon.port`**: Replace `[port]` with the desired port number.
- **`--logging.debug`**: Enables debug-level logging for detailed output.

### Managing the Miner Process:
- **Start the Miner**: The above command will start the miner as a PM2 process named `video-miner`.
- **View Logs**: Use `pm2 logs video-miner` to monitor miner logs in real time.
- **Restart the Miner**: Use `pm2 restart video-miner` to restart the process.
- **Stop the Miner**: Use `pm2 stop video-miner` to stop the process.

---

## Additional Notes

- Ensure all dependencies are installed and configured correctly before running the miner.
- Use a high-performance GPU and sufficient system resources for optimal performance.
- For troubleshooting and debugging, refer to the logs available in PM2.
