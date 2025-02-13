# Running Validator

---

## Machine Requirements

To achieve optimal results, we recommend the following setup:

- **Operating System**: Ubuntu 24.04 LTS or higher
  [Learn more about Ubuntu 24.04 LTS](https://ubuntu.com/blog/tag/ubuntu-24-04-lts)
- **GPU**: NVIDIA RTX 4090 or higher  
  (Required for efficient video upscaling using Video2X)

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

2. Add the required variables to the `.env` file. For example:
   ```env
   GOOGLE_KEY_FILE_NAME="Your Google Drive Key File"
   PEXELS_API_KEY="Your Pexels account api key"
   ```

3. The `GOOGLE_KEY_FILE_NAME` should point to the **Google OAuth2 client secret JSON file** required for integrating Google Drive. This file is used to authenticate and upload video files to your Google Drive account.

4. To generate the **Google OAuth2 Client Secret JSON file**, follow these steps:
   - Use the Google Cloud Console UI to create the file. Refer to this [Google Support Guide](https://ragug.medium.com/how-to-upload-files-using-the-google-drive-api-in-python-ebefdfd63eab) for detailed instructions.
   - Download the JSON file and place it in the **project root directory**.
   - Ensure the `GOOGLE_KEY_FILE_NAME` in the `.env` file matches the name of the downloaded JSON file.
5. Create your Pexels API key and replace it
6. Once the `.env` file is properly configured, the application will use the specified credentials for Google Drive integration.


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
   python3 -m venv venv
   ```

2. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

---

### Step 2: Install Dependencies

1. Install `meson`:
   ```bash
   pip install meson
   ```

2. Install system dependencies:
   ```bash
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
   We recommend adding `-Denable_avx512=true` to enhance validation speed. If CUDA is available, include the flag `-Denable_cuda=true`

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

## Install LPIPS

....


## Running the Scoring endpoint

You can run the video upscaling endpoint using **PM2** to manage the process:

```bash
pm2 start "python services/scoring/server.py" --name scoring_endpoint
```

### Notes:
- The `scoring_endpoint` process will handle video upscaling requests.
- Use the following PM2 commands to manage the process:
  - **View Logs**: `pm2 logs scoring_endpoint`
  - **Restart**: `pm2 restart scoring_endpoint`
  - **Stop**: `pm2 stop scoring_endpoint`

---



## Running the Validator with PM2

To run the validator, use the following command:

```bash
pm2 start "python3 neurons/validator.py --wallet.name [Your_Wallet_Name] --wallet.hotkey [Your_Hotkey_Name] --subtensor.network test --netuid 292 --axon.port [port] --logging.debug" --name video-validator
```

### Parameters:
- **`--wallet.name`**: Replace `[Your_Wallet_Name]` with your wallet name.
- **`--wallet.hotkey`**: Replace `[Your_Hotkey_Name]` with your hotkey name.
- **`--subtensor.network`**: Specify the target network (e.g., `finney`).
- **`--netuid`**: Specify the network UID (e.g., `292`).
- **`--axon.port`**: Replace `[port]` with the desired port number.
- **`--logging.debug`**: Enables debug-level logging for detailed output.

### Managing the validator Process:
- **Start the validator**: The above command will start the validator as a PM2 process named `video-validator`.
- **View Logs**: Use `pm2 logs vidaio-validator` to monitor validator logs in real time.
- **Restart the Validator**: Use `pm2 restart video-validator` to restart the process.
- **Stop the Validator**: Use `pm2 stop video-validator` to stop the process.

---
