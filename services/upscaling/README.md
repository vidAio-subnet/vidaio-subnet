# Video2X Upscaling Script

This script utilizes the [Video2X](https://github.com/vidAio-subnet/video2x) model to perform high-quality video upscaling. this script is primarily designed for HD-to-4K video upscaling, supporting resolutions such as 1080x2048 to 2160x4096. The tool can support both 2X and 4X video upscaling, making it ideal for enhancing video quality while maintaining sharpness and clarity.

## Features
- **High-Quality Upscaling**: Supports 2X and 4X video upscaling for resolutions from HD to 4K.
- **Frame Loss Prevention**: Video2X has an issue where the last two frames of the video are lost during processing. To address this, the script duplicates the last frame at the end of the video before processing, ensuring frame consistency.   [Issue](https://github.com/k4yt3x/video2x/issues/1318)

## Installation

Follow these steps to download and install the Video2X package on your environment:

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
<!-- But we recommend you to follow miner setup guidance since this approach is not safe -->

## Usage

Once installed, Video2X can be used to upscale videos by running the appropriate commands. This scipt solved the Lossing last two frame problem by ensuring that the input video is prepared with an additional duplicated last frame. This can be achieved through simple video editing tools or by using an additional script.

### Example Command
To upscale a video using Video2X:
```bash
video2x -i input.mp4 -o output.mp4 -p realesrgan -s 2 -c libx264 -e preset=slow -e crf=24
```
Replace `-s 2` with `-s 4` for 4X upscaling.

## Known Issues
- **Frame Loss**: Video2X may lose the last two frames of the input video. To ensure frame consistency, duplicate the last frame of the input video before processing.
- **Performance**: Enhancing the resolution of large videos can demand substantial computational power. Make sure your system meets the necessary requirements, basically including the Ubuntu 24.04 operating system and RTX 6000 GPUs.

---
**Note**: This model are tailored for Ubuntu 24.04. Compatibility with other Linux distributions may vary.
