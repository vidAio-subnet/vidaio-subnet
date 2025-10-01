# Running Validator

---

## Machine Requirements

To achieve optimal results, we recommend the following setup:

- **Operating System**: Ubuntu 24.04 LTS or higher
  [Learn more about Ubuntu 24.04 LTS](https://ubuntu.com/blog/tag/ubuntu-24-04-lts)
- **GPU**: NVIDIA RTX A6000 with 48GB VRAM and 30 CPU Cores

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

Clone the VMAF repository into the working root directory of your `vidaio-subnet` package. If the `vidaio-subnet` virtual environment is currently active, deactivate it first:

```bash
git clone https://github.com/vidAio-subnet/vmaf.git
cd vmaf
```

---

### Step 1: Set Up a Virtual Environment in VMAF directory

1. Install `venv` if it’s not already installed:
   ```bash
   python3 -m venv vmaf-venv
   ```

2. Activate the virtual environment:
   ```bash
   source vmaf-venv/bin/activate
   ```

---

### Step 2: Install Dependencies

1. Install `meson`:
   ```bash
   pip install meson
   ```

2. Install system dependencies:
   ```bash
   sudo apt-get update
   sudo apt-get install nasm ninja-build doxygen xxd
   ```
   For Ninja, verify whether the package name is `ninja` or `ninja-build` before running the install command.

---

### Step 3: Compile VMAF


1. Set up the build environment:
   ```bash
   cd libvmaf
   meson setup build --buildtype release -Denable_avx512=true
   ```

2. Optional flags:
   - Use `-Denable_float=true` to enable floating-point feature extractors.
   - Use `-Denable_avx512=true` to enable AVX-512 SIMD instructions for faster processing on supported CPUs.  
   - Use `-Denable_cuda=true` to build with CUDA support (requires `nvcc` and CUDA >= 11).  
     **Note:** To enable CUDA successfully, ensure `nvcc` and the CUDA driver are installed. Refer to the [CUDA and NVCC setup guide](miner_setup.md#step-2-install-cuda-and-nvcc).
   - Use `-Denable_nvtx=true` to enable NVTX marker support for profiling with Nsight Systems.
   - **Recommendation:**
   We recommend adding `-Denable_avx512=true` to enhance validation speed. If CUDA is available, include the flag `-Denable_cuda=true` But At present, VMAF does not include support for CUDA integration.

3. Build the project:
   ```bash
   ninja -vC build
   ```

---

### Step 4: Test the Build

Run tests to verify the build:
```bash
ninja -vC build test
```

---

### Step 5: Install VMAF

Install the library, headers, and the command-line tool:
```bash
ninja -vC build install
```

---

### Step 6: Generate Documentation

Generate HTML documentation:
```bash
ninja -vC build doc/html
```

### Step 7: Deactivate vmaf-venv, activate project venv

```bash
deactivate
cd ..
cd ..
source venv/bin/activate
```

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
