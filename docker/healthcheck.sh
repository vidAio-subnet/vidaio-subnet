#!/bin/bash
# SN85 Miner Health Check Script
# Comprehensive health monitoring for containerized miner

set -e

# Exit codes
HEALTHY=0
UNHEALTHY=1
DEGRADED=2

# Colors (only if terminal supports it)
if [[ -t 1 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    NC='\033[0m'
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    NC=''
fi

# Status tracking
ERRORS=()
WARNINGS=()

check_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

check_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    ERRORS+=("$1")
}

check_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    WARNINGS+=("$1")
}

check_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# ==============================================================================
# Component Health Checks
# ==============================================================================

check_gpu() {
    echo "=== GPU Status ==="

    if ! command -v nvidia-smi &> /dev/null; then
        check_warn "nvidia-smi not found - GPU monitoring unavailable"
        return
    fi

    if nvidia-smi &> /dev/null; then
        local gpu_info=$(nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
        check_pass "GPU detected: $gpu_info"
    else
        check_warn "No GPU available - running in CPU mode"
    fi
}

check_redis() {
    echo "=== Redis Connection ==="

    local redis_host="${REDIS_HOST:-redis}"
    local redis_port="${REDIS_PORT:-6379}"

    if timeout 5 bash -c "echo > /dev/tcp/$redis_host/$redis_port" 2>/dev/null; then
        check_pass "Redis connected at $redis_host:$redis_port"
    else
        check_fail "Cannot connect to Redis at $redis_host:$redis_port"
    fi
}

check_services() {
    echo "=== Service Processes ==="

    local services=("video-upscaler" "video-compressor" "video-miner")
    local any_running=false

    for service in "${services[@]}"; do
        if pm2 describe "$service" &> /dev/null; then
            local status=$(pm2 jlist 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print([p['pm2_env']['status'] for p in d.get('processes', []) if p['name']=='$service'][0] if [p for p in d.get('processes', []) if p['name']=='$service'] else 'unknown')")
            if [[ "$status" == "online" ]]; then
                check_pass "$service is running"
                any_running=true
            else
                check_fail "$service is not online (status: $status)"
            fi
        else
            check_warn "$service not registered in PM2"
        fi
    done

    if [[ "$any_running" == false ]]; then
        check_fail "No services are running"
    fi
}

check_disk_space() {
    echo "=== Disk Space ==="

    local temp_usage=$(df -h "$TEMP_DIR" 2>/dev/null | awk 'NR==2 {print $5}' | tr -d '%')
    local output_usage=$(df -h "$OUTPUT_DIR" 2>/dev/null | awk 'NR==2 {print $5}' | tr -d '%')

    if [[ -n "$temp_usage" && "$temp_usage" -lt 90 ]]; then
        check_pass "Temp directory: ${temp_usage}% used"
    elif [[ -n "$temp_usage" ]]; then
        check_warn "Temp directory: ${temp_usage}% used (critical)"
    fi

    if [[ -n "$output_usage" && "$output_usage" -lt 90 ]]; then
        check_pass "Output directory: ${output_usage}% used"
    elif [[ -n "$output_usage" ]]; then
        check_warn "Output directory: ${output_usage}% used (critical)"
    fi
}

check_temp_cleanup() {
    echo "=== Temp Directory Cleanup ==="

    local temp_dir="${TEMP_DIR:-/data/temp}"
    local file_count=$(find "$temp_dir" -type f 2>/dev/null | wc -l)

    if [[ "$file_count" -lt 100 ]]; then
        check_pass "Temp directory contains $file_count files"
    elif [[ "$file_count" -lt 500 ]]; then
        check_warn "Temp directory contains $file_count files"
    else
        check_fail "Temp directory has $file_count files (cleanup needed)"
    fi
}

check_memory() {
    echo "=== Memory Usage ==="

    local mem_info=$(free -m 2>/dev/null | awk 'NR==2{printf "%.0f%%", $3*100/$2 }')
    local mem_used=$(free -m 2>/dev/null | awk 'NR==2{printf "%s", $3}')
    local mem_total=$(free -m 2>/dev/null | awk 'NR==2{printf "%s", $2}')

    if [[ -n "$mem_info" ]]; then
        local usage_percent=${mem_info%%%}
        if [[ "$usage_percent" -lt 80 ]]; then
            check_pass "Memory: $mem_info used (${mem_used}MB/${mem_total}MB)"
        elif [[ "$usage_percent" -lt 95 ]]; then
            check_warn "Memory: $mem_info used (${mem_used}MB/${mem_total}MB)"
        else
            check_fail "Memory: $mem_info used (${mem_used}MB/${mem_total}MB) - critical"
        fi
    fi
}

check_chutes() {
    echo "=== Chutes Inference ==="

    if [[ "${USE_CHUTES:-false}" != "true" ]]; then
        check_info "Chutes inference disabled (USE_CHUTES=false)"
        return
    fi

    if [[ -z "${CHUTES_API_KEY:-}" ]]; then
        check_warn "Chutes enabled but CHUTES_API_KEY not set"
        return
    fi

    # Test Chutes connectivity (lightweight check)
    if python3 << 'EOF' 2>/dev/null; then
import os, sys, asyncio, httpx
async def check():
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(
                "https://api.chutes.ai/v1/health",
                headers={"Authorization": f"Bearer {os.getenv('CHUTES_API_KEY')}"}
            )
            sys.exit(0 if r.status_code < 400 else 1)
    except Exception:
        sys.exit(1)
asyncio.run(check())
EOF
        check_pass "Chutes API connectivity OK"
    else
        check_warn "Chutes API unreachable - local fallback will be used"
    fi
}

check_bittensor() {
    echo "=== Bittensor Connection ==="

    # Check if wallet exists
    if [[ -d "/app/.bittensor/wallets" ]]; then
        local wallet_count=$(find /app/.bittensor/wallets -maxdepth 1 -type d 2>/dev/null | wc -l)
        if [[ "$wallet_count" -gt 1 ]]; then
            check_pass "Bittensor wallets found"
        else
            check_warn "No Bittensor wallets configured"
        fi
    else
        check_warn "Bittensor wallet directory not mounted"
    fi
}

check_ffmpeg() {
    echo "=== FFmpeg ==="

    if command -v ffmpeg &> /dev/null; then
        local version=$(ffmpeg -version 2>/dev/null | head -1 | cut -d' ' -f3)
        check_pass "FFmpeg installed: v$version"

        # Check for NVIDIA support
        if ffmpeg -encoders 2>/dev/null | grep -q "h264_nvenc"; then
            check_pass "NVIDIA hardware encoding available"
        else
            check_warn "NVIDIA hardware encoding not available"
        fi
    else
        check_fail "FFmpeg not installed"
    fi
}

check_video2x() {
    echo "=== Video2X ==="

    if command -v video2x &> /dev/null; then
        local version=$(video2x --version 2>/dev/null || echo "unknown")
        check_pass "Video2X installed: $version"
    else
        check_warn "Video2X not in PATH"
    fi
}

# ==============================================================================
# Main Health Check
# ==============================================================================

main() {
    echo "========================================"
    echo "SN85 Miner Health Check"
    echo "========================================"
    echo "Timestamp: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo ""

    check_gpu
    echo ""
    check_ffmpeg
    echo ""
    check_video2x
    echo ""
    check_redis
    echo ""
    check_services
    echo ""
    check_disk_space
    echo ""
    check_memory
    echo ""
    check_temp_cleanup
    echo ""
    check_chutes
    echo ""
    check_bittensor

    echo ""
    echo "========================================"
    echo "Summary"
    echo "========================================"

    if [[ ${#ERRORS[@]} -gt 0 ]]; then
        echo -e "${RED}Errors: ${#ERRORS[@]}${NC}"
        printf '  - %s\n' "${ERRORS[@]}"
    fi

    if [[ ${#WARNINGS[@]} -gt 0 ]]; then
        echo -e "${YELLOW}Warnings: ${#WARNINGS[@]}${NC}"
        printf '  - %s\n' "${WARNINGS[@]}"
    fi

    if [[ ${#ERRORS[@]} -eq 0 && ${#WARNINGS[@]} -eq 0 ]]; then
        echo -e "${GREEN}All checks passed - System is healthy${NC}"
        exit $HEALTHY
    elif [[ ${#ERRORS[@]} -eq 0 ]]; then
        echo -e "${YELLOW}System is degraded but functional${NC}"
        exit $DEGRADED
    else
        echo -e "${RED}System is unhealthy${NC}"
        exit $UNHEALTHY
    fi
}

# Run health check
main
