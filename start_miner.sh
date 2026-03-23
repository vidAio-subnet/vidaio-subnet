#!/bin/bash
#
# SN85 Miner Service Startup Script
# For production deployment on Ubuntu 24.04 + RTX 4090
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  SN85 (Vidaio) Miner Startup Script    ${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# =============================================================================
# CONFIGURATION
# =============================================================================

# Service ports (must match vidaio_subnet_core/config.py)
UPSCALER_PORT=29115
COMPRESSOR_PORT=29116
DELETER_PORT=29117

# PM2 process names
UPSCALER_NAME="video-upscaler"
COMPRESSOR_NAME="video-compressor"
DELETER_NAME="video-deleter"
MINER_NAME="sn85-miner"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

check_pm2() {
    if ! command -v pm2 &> /dev/null; then
        echo -e "${RED}ERROR: pm2 not found${NC}"
        echo "Install: npm install -g pm2"
        exit 1
    fi
}

check_redis() {
    if ! pgrep -x "redis-server" > /dev/null; then
        echo -e "${YELLOW}WARNING: Redis not running${NC}"
        echo "Start: sudo systemctl start redis"
        return 1
    fi
    echo -e "${GREEN}✓ Redis is running${NC}"
    return 0
}

check_video2x() {
    if ! command -v video2x &> /dev/null; then
        echo -e "${YELLOW}WARNING: video2x not found in PATH${NC}"
        if [ -n "$VIDEO2X_BINARY" ]; then
            echo "Using VIDEO2X_BINARY: $VIDEO2X_BINARY"
        else
            echo -e "${RED}ERROR: Video2X not found. Set VIDEO2X_BINARY env var${NC}"
            return 1
        fi
    else
        echo -e "${GREEN}✓ Video2X found: $(which video2x)${NC}"
    fi
    return 0
}

check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${GREEN}✓ NVIDIA GPU detected:${NC}"
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | head -1
    else
        echo -e "${YELLOW}WARNING: nvidia-smi not found - no GPU acceleration${NC}"
    fi
}

check_ffmpeg() {
    if ! command -v ffmpeg &> /dev/null; then
        echo -e "${RED}ERROR: ffmpeg not found${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ ffmpeg: $(ffmpeg -version | head -1 | cut -d' ' -f3)${NC}"
}

check_bucket_config() {
    if [ -z "$BUCKET_COMPATIBLE_ENDPOINT" ] || [ -z "$BUCKET_COMPATIBLE_ACCESS_KEY" ] || [ -z "$BUCKET_COMPATIBLE_SECRET_KEY" ]; then
        echo -e "${YELLOW}WARNING: Bucket credentials not set${NC}"
        echo "Required: BUCKET_COMPATIBLE_ENDPOINT, BUCKET_COMPATIBLE_ACCESS_KEY, BUCKET_COMPATIBLE_SECRET_KEY"
        return 1
    fi
    echo -e "${GREEN}✓ Bucket configured: ${BUCKET_COMPATIBLE_ENDPOINT}${NC}"
    return 0
}

check_wallet() {
    if [ -z "$BT_WALLET_NAME" ] || [ -z "$BT_WALLET_HOTKEY" ]; then
        # Check .env file
        if [ -f ".env" ]; then
            source .env
        fi
    fi

    if [ -z "$BT_WALLET_NAME" ] || [ -z "$BT_WALLET_HOTKEY" ]; then
        echo -e "${YELLOW}WARNING: Wallet not configured${NC}"
        echo "Set BT_WALLET_NAME and BT_WALLET_HOTKEY environment variables"
        return 1
    fi
    echo -e "${GREEN}✓ Wallet: ${BT_WALLET_NAME}:${BT_WALLET_HOTKEY}${NC}"
    return 0
}

start_service() {
    local name="$1"
    local cmd="$2"

    if pm2 describe "$name" &>/dev/null; then
        echo -e "${YELLOW}Reloading $name...${NC}"
        pm2 reload "$name"
    else
        echo -e "${GREEN}Starting $name...${NC}"
        pm2 start bash --name "$name" -- -c "$cmd"
    fi
}

stop_service() {
    local name="$1"
    if pm2 describe "$name" &>/dev/null; then
        echo -e "${YELLOW}Stopping $name...${NC}"
        pm2 stop "$name"
        pm2 delete "$name"
    fi
}

# =============================================================================
# COMMANDS
# =============================================================================

cmd_check() {
    echo -e "${BLUE}Checking prerequisites...${NC}"
    echo ""

    check_pm2
    check_ffmpeg
    check_video2x
    check_gpu
    check_redis
    check_bucket_config
    check_wallet

    echo ""
    echo -e "${BLUE}Environment:${NC}"
    echo "PYTHONPATH: ${PYTHONPATH:-.}"
    echo "Video2X: ${VIDEO2X_BINARY:-$(which video2x 2>/dev/null || echo 'not found')}"
    echo "Working dir: $(pwd)"
}

cmd_start() {
    cmd_check

    echo ""
    echo -e "${BLUE}Starting SN85 miner services...${NC}"
    echo ""

    # 1. Upscaler service
    start_service "$UPSCALER_NAME" "cd $(pwd) && PYTHONPATH=. python services/upscaling/server.py"

    # 2. Compression service
    start_service "$COMPRESSOR_NAME" "cd $(pwd) && PYTHONPATH=. python services/compress/server.py"

    # 3. File deletion service
    start_service "$DELETER_NAME" "cd $(pwd) && PYTHONPATH=. python services/miner_utilities/file_deletion_server.py"

    # 4. Miner (after services are ready)
    echo "Waiting for services to start (5s)..."
    sleep 5

    local wallet_name="${BT_WALLET_NAME:-default}"
    local wallet_hotkey="${BT_WALLET_HOTKEY:-default}"

    start_service "$MINER_NAME" "cd $(pwd) && PYTHONPATH=. python neurons/miner.py --wallet.name $wallet_name --wallet.hotkey $wallet_hotkey --netuid 85 --subtensor.network finney --logging.debug"

    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  SN85 Miner services started!          ${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    pm2 status
    echo ""
    echo "View logs: pm2 logs"
    echo "Monitor:   pm2 monit"
}

cmd_stop() {
    echo -e "${BLUE}Stopping SN85 miner services...${NC}"

    stop_service "$MINER_NAME"
    stop_service "$DELETER_NAME"
    stop_service "$COMPRESSOR_NAME"
    stop_service "$UPSCALER_NAME"

    echo -e "${GREEN}All services stopped${NC}"
}

cmd_restart() {
    cmd_stop
    sleep 2
    cmd_start
}

cmd_status() {
    echo -e "${BLUE}SN85 Miner Status${NC}"
    echo ""
    pm2 status
    echo ""
    echo -e "${BLUE}Service Health:${NC}"

    # Check if services are responding
    if curl -s "http://localhost:$UPSCALER_PORT/docs" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Upscaler: http://localhost:$UPSCALER_PORT${NC}"
    else
        echo -e "${RED}✗ Upscaler: not responding${NC}"
    fi

    if curl -s "http://localhost:$COMPRESSOR_PORT/docs" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Compressor: http://localhost:$COMPRESSOR_PORT${NC}"
    else
        echo -e "${RED}✗ Compressor: not responding${NC}"
    fi
}

cmd_logs() {
    pm2 logs "$@"
}

cmd_benchmark() {
    echo -e "${BLUE}Running compression benchmark...${NC}"

    if [ -z "$1" ]; then
        echo "Usage: $0 benchmark <video_path>"
        echo "Example: $0 benchmark test_video.mp4"
        exit 1
    fi

    if [ ! -f "$1" ]; then
        echo -e "${RED}ERROR: Video not found: $1${NC}"
        exit 1
    fi

    python benchmark_compression.py "$1" --range --output benchmark_results.json
}

cmd_help() {
    echo "SN85 (Vidaio) Miner Management Script"
    echo ""
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  check       Check prerequisites and configuration"
    echo "  start       Start all miner services (upscaler, compressor, miner)"
    echo "  stop        Stop all miner services"
    echo "  restart     Restart all services"
    echo "  status      Show service status"
    echo "  logs        Show logs (pass additional args to pm2 logs)"
    echo "  benchmark   Run compression benchmark on test video"
    echo ""
    echo "Environment Variables:"
    echo "  BT_WALLET_NAME            Wallet name (default: default)"
    echo "  BT_WALLET_HOTKEY          Hotkey name (default: default)"
    echo "  BUCKET_COMPATIBLE_*       S3-compatible storage credentials"
    echo "  VIDEO2X_BINARY            Path to video2x executable"
    echo ""
    echo "Examples:"
    echo "  $0 check"
    echo "  $0 start"
    echo "  $0 logs --lines 100"
    echo "  $0 benchmark sample.mp4"
}

# =============================================================================
# MAIN
# =============================================================================

case "${1:-help}" in
    check)
        cmd_check
        ;;
    start)
        cmd_start
        ;;
    stop)
        cmd_stop
        ;;
    restart)
        cmd_restart
        ;;
    status)
        cmd_status
        ;;
    logs)
        shift
        cmd_logs "$@"
        ;;
    benchmark)
        shift
        cmd_benchmark "$@"
        ;;
    help|--help|-h)
        cmd_help
        ;;
    *)
        echo "Unknown command: $1"
        cmd_help
        exit 1
        ;;
esac
