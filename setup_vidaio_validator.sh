#!/bin/bash

# Vidaio Validator Setup Script
# Run with: bash setup_vidaio_validator.sh

set -e  # Exit on error

echo "======================================"
echo "Vidaio Validator Setup Script"
echo "======================================"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}➜ $1${NC}"
}

# Check if running on Ubuntu
if [ -f /etc/os-release ]; then
    . /etc/os-release
    if [[ "$ID" != "ubuntu" ]]; then
        print_error "Warning: This script is designed for Ubuntu 24.04 LTS or higher"
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

# Update system
print_info "Updating system packages..."
sudo apt update && sudo apt upgrade -y
print_success "System updated"

# Install PM2
print_info "Installing PM2..."
if ! command -v pm2 &> /dev/null; then
    sudo apt install npm -y
    sudo npm install pm2 -g
    pm2 update
    print_success "PM2 installed"
else
    print_success "PM2 already installed"
fi

# Install Redis
print_info "Installing Redis..."
if ! command -v redis-server &> /dev/null; then
    sudo apt install redis-server -y
    sudo systemctl start redis
    sudo systemctl enable redis-server
    sudo systemctl status redis --no-pager
    print_success "Redis installed and started"
else
    print_success "Redis already installed"
fi

# Install Python 3.10+ and pip
print_info "Installing Python and pip..."
sudo apt install python3 python3-pip python3-venv -y
print_success "Python and pip installed"

# Install FFMPEG
print_info "Installing FFMPEG..."
if ! command -v ffmpeg &> /dev/null; then
    sudo apt install ffmpeg -y
    print_success "FFMPEG installed"
else
    print_success "FFMPEG already installed"
fi

# Clone vidaio-subnet repository
print_info "Cloning vidaio-subnet repository..."
if [ ! -d "vidaio-subnet" ]; then
    git clone https://github.com/vidaio-subnet/vidaio-subnet.git
    print_success "Repository cloned"
else
    print_info "Repository already exists, pulling latest changes..."
    cd vidaio-subnet
    git pull
    cd ..
fi

cd vidaio-subnet

# Create and activate virtual environment
print_info "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Virtual environment created"
fi

source venv/bin/activate
print_success "Virtual environment activated"

# Install project dependencies
print_info "Installing project dependencies..."
pip install --upgrade pip
pip install -e .
print_success "Project dependencies installed"

# Configure environment variables
print_info "Setting up environment variables..."
if [ ! -f ".env" ]; then
    if [ -f ".env.template" ]; then
        cp .env.template .env
        print_success ".env file created from template"
        print_info "Please edit .env file with your credentials:"
        echo "  - BUCKET_NAME"
        echo "  - BUCKET_COMPATIBLE_ENDPOINT"
        echo "  - BUTKET_COMPATIBLE_ACCESS_KEY"
        echo "  - BUCKET_COMPATIBLE_SECRET_KEY"
        echo "  - PEXELS_API_KEY"
        echo "  - WANDB_API_KEY"
    else
        print_error ".env.template not found. Please create .env manually"
    fi
else
    print_info ".env file already exists"
fi

# Install VMAF dependencies
print_info "Installing VMAF system dependencies..."
sudo apt-get install nasm ninja-build doxygen xxd -y
print_success "VMAF dependencies installed"

# Clone VMAF repository
print_info "Cloning VMAF repository..."
if [ ! -d "vmaf" ]; then
    git clone https://github.com/vidAio-subnet/vmaf.git
    print_success "VMAF repository cloned"
else
    print_info "VMAF repository already exists"
fi

cd vmaf

# Deactivate current venv and create VMAF venv
deactivate

print_info "Creating VMAF virtual environment..."
if [ ! -d "vmaf-venv" ]; then
    python3 -m venv vmaf-venv
    print_success "VMAF virtual environment created"
fi

source vmaf-venv/bin/activate
print_success "VMAF virtual environment activated"

# Install meson
print_info "Installing meson..."
pip install meson
print_success "Meson installed"

# Build VMAF
cd libvmaf

print_info "Configuring VMAF build..."
meson setup build --buildtype release -Denable_avx512=true

print_info "Building VMAF (this may take a few minutes)..."
ninja -vC build

print_info "Testing VMAF build..."
ninja -vC build test

print_info "Installing VMAF..."
sudo ninja -vC build install

print_success "VMAF installed successfully"

# Return to project directory and activate project venv
cd ../..
deactivate
source venv/bin/activate

print_success "Returned to project directory and activated project venv"

echo ""
echo "======================================"
print_success "Installation Complete!"
echo "======================================"
echo ""
print_info "Next steps:"
echo "1. Edit the .env file with your credentials:"
echo "   nano .env"
echo ""
echo "2. Activate the main env:" 
echo "   source venv/bin/activate"
echo ""
echo "3. Start the validator with PM2:"
echo "   pm2 start run.sh --name vidaio_v_autoupdater -- \\"
echo "     --wallet.name [Your_Wallet_Name] \\"
echo "     --wallet.hotkey [Your_Hotkey_Name] \\"
echo "     --subtensor.network finney \\"
echo "     --netuid 85 \\"
echo "     --axon.port [port] \\"
echo "     --logging.debug"
echo ""
print_info "To activate the virtual environment manually:"
echo "   source venv/bin/activate"
echo ""