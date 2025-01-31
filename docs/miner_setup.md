# Decentralized Video Processing Miner

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
pip install .
```

---

### 4. Configure Environment Variables

Create a `.env` file in the project root directory and define the necessary environment variables, reference .env.template file:
```env
REDIS_HOST=localhost
REDIS_PORT=6379
DATABASE_URL=sqlite:///example.db
```

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

2. Install or update CUDA drivers if necessary:
   ```bash
   sudo apt update
   sudo apt install nvidia-cuda-toolkit -y
   ```

For more information, refer to the [CUDA Toolkit Installation Guide](https://developer.nvidia.com/cuda-toolkit).

---

### Step 3: Install Video2X

Install the **Video2X** package by following these steps:

#### 1. Install Cargo
Cargo is required to build the Video2X package:
```bash
sudo apt-get update
sudo apt-get install cargo -y
cargo install just
```

#### 2. Clone the Video2X Repository
```bash
git clone --recurse-submodules https://github.com/k4yt3x/video2x.git
cd video2x
```

#### 3. Build the Video2X Project
Before building, ensure `~/.cargo/bin` is included in your `PATH` environment variable:
```bash
export PATH="$HOME/.cargo/bin:$PATH"
```

Run the following command to build the package:
```bash
just ubuntu2404
```

Once the build is complete, the `.deb` package will be located in the current directory.

#### 4. Install the Built Package
Install the `.deb` package using:
```bash
sudo dpkg -i <package>.deb
```

For additional details, refer to the [Video2X Documentation](https://docs.video2x.org/building/linux.html).

---



## Running the Miner

Once the setup is complete, you can start the miner using PM2. Here’s an example command to start the miner:

```bash
pm2 start miner.py --name "video-miner"
```

To monitor the miner logs:
```bash
pm2 logs video-miner
```

To stop the miner:
```bash
pm2 stop video-miner
```

---

## Additional Notes

- Ensure all dependencies are installed and configured correctly before running the miner.
- Use a high-performance GPU and sufficient system resources for optimal performance.
- For troubleshooting and debugging, refer to the logs available in PM2.

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed explanation.

---

## Contact

For questions or support, feel free to reach out:
- **Author**: Your Name
- **Email**: your.email@example.com

---

## Acknowledgments

This project uses the following libraries and tools:
- [FastAPI](https://fastapi.tiangolo.com/)
- [Bittensor](https://bittensor.com/)
- [Video2X](https://github.com/k4yt3x/video2x)
- [FFMPEG](https://www.ffmpeg.org/)
- And many more!

