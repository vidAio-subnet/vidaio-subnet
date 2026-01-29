# TEE Miner Setup Guide

This guide explains how to set up and run a vidaio-subnet miner inside an Intel SGX Trusted Execution Environment (TEE) using Gramine.

## Prerequisites

### Hardware Requirements

1. **Intel CPU with SGX support**:
   - Intel Xeon (Scalable Processors, 3rd Gen+) with SGX2
   - Intel Core (6th Gen+) with SGX enabled in BIOS
   - Check SGX support: `cpuid | grep -i sgx`

2. **Memory**: 
   - Minimum 16GB RAM (8GB for enclave + system overhead)
   - Recommended 32GB+ for production workloads

3. **Storage**:
   - SSD recommended for video processing
   - Minimum 100GB free space

### Software Requirements

1. **Operating System**: Ubuntu 22.04 LTS (recommended)
2. **Intel SGX Software Stack**:
   - Intel SGX Driver
   - Intel SGX PSW (Platform Software)
   - Intel DCAP (Data Center Attestation Primitives)
3. **Docker** (optional, for containerized deployment)

## Installation

### Step 1: Enable SGX in BIOS

1. Reboot and enter BIOS settings
2. Navigate to Security or Advanced settings
3. Enable "Intel SGX" or "Software Guard Extensions"
4. Set SGX memory size to maximum available
5. Save and exit BIOS

### Step 2: Install Intel SGX Driver (Ubuntu 22.04)

```bash
# Add Intel SGX repository
echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/intel-sgx-archive-keyring.gpg] https://download.01.org/intel-sgx/sgx_repo/ubuntu jammy main' | sudo tee /etc/apt/sources.list.d/intel-sgx.list

# Add Intel signing key
wget -qO - https://download.01.org/intel-sgx/sgx_repo/ubuntu/intel-sgx-deb.key | sudo gpg --dearmor -o /usr/share/keyrings/intel-sgx-archive-keyring.gpg

# Update and install
sudo apt update
sudo apt install -y libsgx-epid libsgx-quote-ex libsgx-dcap-ql libsgx-dcap-default-qpl sgx-aesm-service
```

### Step 3: Verify SGX Installation

```bash
# Check SGX devices
ls -la /dev/sgx*
# Expected: /dev/sgx_enclave, /dev/sgx_provision

# Check AESM service
sudo systemctl status aesmd

# Test SGX availability
is-sgx-available
```

### Step 4: Install Gramine

```bash
# Add Gramine repository
curl -fsSLo /usr/share/keyrings/gramine-keyring.gpg https://packages.gramineproject.io/gramine-keyring.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/gramine-keyring.gpg] https://packages.gramineproject.io/ jammy main" | sudo tee /etc/apt/sources.list.d/gramine.list

# Install Gramine
sudo apt update
sudo apt install -y gramine

# Generate SGX signing key
gramine-sgx-gen-private-key
```

### Step 5: Clone and Build the Miner

```bash
# Clone repository
git clone https://github.com/your-org/vidaio-subnet.git
cd vidaio-subnet

# Install Python dependencies
pip install -r requirements.txt

# Additional TEE dependencies
pip install cryptography aiohttp boto3

# Build SGX manifest
cd gramine
make
```

The build process will output the **MRENCLAVE** value - save this for verification.

## Running the TEE Miner

### Option A: Direct Gramine SGX (Recommended)

```bash
# Set environment variables
export WALLET_NAME=miner
export WALLET_HOTKEY=default
export SUBTENSOR_NETWORK=finney
export NETUID=1

# Run inside SGX enclave
cd /path/to/vidaio-subnet/gramine
gramine-sgx miner
```

### Option B: Docker Container

```bash
# Build Docker image
docker build -f docker/Dockerfile.miner-tee -t vidaio-miner-tee .

# Run with SGX device access
docker run -d \
  --name vidaio-miner-tee \
  --device /dev/sgx_enclave \
  --device /dev/sgx_provision \
  -v /var/run/aesmd:/var/run/aesmd \
  -v ~/.bittensor:/root/.bittensor:ro \
  -p 8091:8091 \
  -e WALLET_NAME=miner \
  -e WALLET_HOTKEY=default \
  -e SUBTENSOR_NETWORK=finney \
  -e NETUID=1 \
  vidaio-miner-tee
```

### Option C: Docker Compose

```bash
# Configure environment
cp .env.example .env
# Edit .env with your wallet settings

# Start miner
docker-compose -f docker/docker-compose-miner-tee.yml up -d

# View logs
docker-compose -f docker/docker-compose-miner-tee.yml logs -f
```

## Verification

### Check Miner Status

```bash
# Check if enclave is running
docker logs vidaio-miner-tee

# Expected output:
# TEE Miner initialized (TEE mode: True)
# Running inside enclave: True
# Waiting for attestation requests from validators...
```

### Test Attestation

Validators will automatically attest your miner before sending tasks. You should see:

```
Received attestation request
Generated attestation response (SGX: True)
Stored session key: <session_id>
```

### Debug Mode (Without SGX)

For testing without SGX hardware:

```bash
# Run in direct mode (no SGX protection)
gramine-direct miner

# Or with Docker
docker run ... -e TEE_ENABLED=false vidaio-miner-tee
```

**Warning**: Direct mode provides no security guarantees and should only be used for testing.

## Troubleshooting

### SGX Device Not Found

```bash
# Check if SGX driver is loaded
lsmod | grep sgx

# Load SGX driver
sudo modprobe isgx  # Legacy driver
# or
sudo modprobe intel_sgx  # In-kernel driver (5.11+)
```

### AESM Service Issues

```bash
# Restart AESM
sudo systemctl restart aesmd

# Check AESM socket
ls -la /var/run/aesmd/aesm.socket
```

### Enclave Memory Issues

If you see "out of memory" errors:

1. Check BIOS SGX memory setting
2. Reduce `enclave_size` in `gramine/miner.manifest.template`
3. Ensure no other enclaves are running

### Attestation Failures

1. Check that DCAP is properly configured
2. Verify network connectivity to Intel's attestation services
3. Check `/var/log/syslog` for AESM errors

## Security Considerations

1. **Private Keys**: Never expose your Gramine signing key
2. **Debug Mode**: Never run debug enclaves in production (set `DEBUG_MODE=0`)
3. **Logging**: Enclave logs should not contain sensitive data
4. **Updates**: Keep SGX software stack updated for security patches

## Performance Notes

- CPU-based video processing (RAISR, SVT-AV1) is slower than GPU
- Enclave overhead adds ~5-10% processing time
- First attestation may take 10-30 seconds
- Subsequent tasks use cached session keys

## Support

For issues specific to:
- **SGX Hardware**: Intel SGX Developer Support
- **Gramine**: https://github.com/gramineproject/gramine/issues
- **vidaio-subnet TEE**: Open an issue in this repository
