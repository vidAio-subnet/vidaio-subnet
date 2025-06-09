# fast hash_matcher for python

## Install pybind11
sudo apt install pybind11-dev


## Install dependency libs (OpenCV, FFTW3 installation)
```bash
# Download dependencies
apt install libopencv-dev libfftw3-dev cmake
```

## Install FFmpeg that supports nvenc
### Install required dependencies for FFMpeg
```bash
cuda toolkit install
nvidia-smi
sudo apt update
sudo apt install -y nvidia-driver-570
sudo reboot

sudo apt install nvidia-cuda-toolkit

sudo apt update
sudo apt install -y \
  git make build-essential pkg-config \
  libx264-dev libx265-dev libfdk-aac-dev libmp3lame-dev libopus-dev libvpx-dev \
  libass-dev libfreetype6-dev libnpp-dev

sudo apt install -y \
  autoconf automake build-essential cmake git-core libass-dev \
  libfreetype6-dev libtool libvorbis-dev libxcb1-dev \
  libxcb-shm0-dev libxcb-xfixes0-dev pkg-config texinfo zlib1g-dev \
  libx264-dev libx265-dev libnuma-dev libvpx-dev libfdk-aac-dev \
  libmp3lame-dev libopus-dev yasm libavdevice-dev libavfilter-dev
```
### Install ffnvcodec headers
```bash
git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git
cd nv-codec-headers
make
sudo make install
```
### ffmpeg-nv installation
```bash
git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg
cd ffmpeg

./configure \
  --enable-nonfree \
  --enable-cuda-nvcc \
  --enable-libnpp \
  --extra-cflags=-I/usr/local/cuda/include \
  --extra-ldflags=-L/usr/local/cuda/lib64 \
  --enable-cuda \
  --enable-cuvid \
  --enable-nvenc \
  --enable-libx264 --enable-libx265 \
  --enable-gpl

make -j$(nproc)
sudo make install

ffmpeg -hide_banner -encoders | grep nvenc

ffmpeg -hwaccel cuda -i input.mp4 -c:v h264_nvenc -preset p1 output.mp4
```

## git clone & compile
```bash
# Checkout source
git clone https://github.com/kingaimaster94/hash_matcher

# Build
cmake .
make

# Install as a system library
sudo make install
```

## Usage
```bash
import cmatcher
