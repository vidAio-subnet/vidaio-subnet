#!/bin/bash
# SN85 Miner Docker Entrypoint Script
# Handles initialization, configuration, and startup of miner services

set -e

# ==============================================================================
# Configuration and Environment Validation
# ==============================================================================

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# ==============================================================================
# Validation Functions
# ==============================================================================

validate_environment() {
    log_step "Validating environment..."

    # Check for GPU availability (optional but recommended)
    if command -v nvidia-smi &> /dev/null; then
        log_info "GPU detected:"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
    else
        log_warn "No GPU detected. CPU-only mode will impact performance."
    fi

    # Check required directories
    for dir in "$TEMP_DIR" "$OUTPUT_DIR" "$LOG_DIR"; do
        if [[ ! -d "$dir" ]]; then
            log_info "Creating directory: $dir"
            mkdir -p "$dir"
        fi
    done

    # Validate wallet configuration
    if [[ ! -d "/app/.bittensor/wallets" ]]; then
        log_warn "No Bittensor wallet found at /app/.bittensor/wallets"
        log_warn "You must mount your wallet directory: -v ~/.bittensor/wallets:/app/.bittensor/wallets:ro"
    fi

    # Check Chutes configuration if enabled
    if [[ "$USE_CHUTES" == "true" ]]; then
        if [[ -z "$CHUTES_API_KEY" ]]; then
            log_warn "USE_CHUTES=true but CHUTES_API_KEY not set. Local inference will be used as fallback."
        else
            log_info "Chutes inference enabled. Testing connectivity..."
            python3 << 'EOF' 2>/dev/null || log_warn "Chutes health check failed. Will use local fallback."
import os, httpx, asyncio
async def check():
    try:
        r = await httpx.AsyncClient().get(
            "https://api.chutes.ai/v1/health",
            headers={"Authorization": f"Bearer {os.getenv('CHUTES_API_KEY')}"},
            timeout=10
        )
        print(f"Chutes API status: {r.status_code}")
    except Exception as e:
        print(f"Chutes check warning: {e}")
asyncio.run(check())
EOF
        fi
    fi

    log_info "Environment validation complete"
}

# ==============================================================================
# Service Management Functions
# ==============================================================================

wait_for_redis() {
    log_step "Waiting for Redis at $REDIS_HOST:$REDIS_PORT..."
    local max_attempts=30
    local attempt=1

    while ! (echo > /dev/tcp/$REDIS_HOST/$REDIS_PORT) 2>/dev/null; do
        if [[ $attempt -ge $max_attempts ]]; then
            log_error "Redis not available after $max_attempts attempts"
            exit 1
        fi
        log_warn "Redis not ready, waiting... ($attempt/$max_attempts)"
        sleep 2
        ((attempt++))
    done
    log_info "Redis is ready"
}

cleanup_temp() {
    log_step "Cleaning up temporary files..."
    find "$TEMP_DIR" -type f -atime +1 -delete 2>/dev/null || true
    find "$OUTPUT_DIR" -type f -atime +7 -delete 2>/dev/null || true
}

# ==============================================================================
# Command Handlers
# ==============================================================================

cmd_start() {
    log_step "Starting SN85 Miner services..."

    validate_environment
    wait_for_redis

    log_info "Starting with PM2 process manager..."

    # Start services using PM2
    pm2-runtime start /app/pm2.config.js --env production
}

cmd_miner() {
    log_step "Starting miner only..."

    validate_environment

    local wallet_name="${WALLET_NAME:-default}"
    local wallet_hotkey="${WALLET_HOTKEY:-default}"
    local axon_port="${BT_AXON_PORT:-8091}"

    log_info "Wallet: $wallet_name/$wallet_hotkey"
    log_info "Network: $BT_NETWORK, NetUID: $BT_NETUID"
    log_info "Axon Port: $axon_port"

    # Run miner directly
    exec python3 neurons/miner.py \
        --wallet.name "$wallet_name" \
        --wallet.hotkey "$wallet_hotkey" \
        --subtensor.network "$BT_NETWORK" \
        --netuid "$BT_NETUID" \
        --axon.port "$axon_port" \
        --logging.debug
}

cmd_upscaler() {
    log_step "Starting video upscaler service..."

    validate_environment
    wait_for_redis

    exec python3 services/upscaling/server.py
}

cmd_compressor() {
    log_step "Starting video compressor service..."

    validate_environment
    wait_for_redis

    exec python3 services/compress/server.py
}

cmd_health() {
    /app/healthcheck.sh
}

cmd_shell() {
    log_info "Starting interactive shell..."
    exec /bin/bash
}

cmd_monitor() {
    /app/monitor.sh
}

cmd_version() {
    echo "SN85 (Vidaio) Miner Container"
    echo "=============================="
    echo "Python: $(python3 --version)"
    echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
    echo "CUDA Available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
    echo "Video2X: $(video2x --version 2>/dev/null || echo 'Not installed')"
    echo "FFmpeg: $(ffmpeg -version | head -1)"
    echo "Bittensor: $(python3 -c 'import bittensor as bt; print(bt.__version__)')"
    echo "PM2: $(pm2 --version)"
}

# ==============================================================================
# Main Entrypoint
# ==============================================================================

main() {
    echo "========================================"
    echo "SN85 (Vidaio) Miner Docker Container"
    echo "========================================"

    local cmd="${1:-start}"

    case "$cmd" in
        start)
            cmd_start
            ;;
        miner)
            cmd_miner
            ;;
        upscaler)
            cmd_upscaler
            ;;
        compressor)
            cmd_compressor
            ;;
        health)
            cmd_health
            ;;
        monitor)
            cmd_monitor
            ;;
        shell|bash)
            cmd_shell
            ;;
        version|info)
            cmd_version
            ;;
        *)
            echo "Usage: $0 {start|miner|upscaler|compressor|health|monitor|shell|version}"
            echo ""
            echo "Commands:"
            echo "  start      - Start all services with PM2 (default)"
            echo "  miner      - Start miner process only"
            echo "  upscaler   - Start video upscaler service only"
            echo "  compressor - Start video compressor service only"
            echo "  health     - Run health check"
            echo "  monitor    - Show system and miner status"
            echo "  shell      - Start interactive bash shell"
            echo "  version    - Show version information"
            exit 1
            ;;
    esac
}

main "$@"
