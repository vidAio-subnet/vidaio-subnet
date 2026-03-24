#!/bin/bash
# SN85 Miner Monitoring Script
# Real-time system and miner status display

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

show_header() {
    clear
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}          SN85 (Vidaio) Miner - System Monitor                ${CYAN}║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

show_gpu_status() {
    echo -e "${BLUE}┌────────────────────────────────────────────────────────────────┐${NC}"
    echo -e "${BLUE}│ GPU Status${NC}                                                    ${BLUE}│${NC}"
    echo -e "${BLUE}└────────────────────────────────────────────────────────────────┘${NC}"

    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,power.limit \
            --format=csv,noheader,nounits 2>/dev/null | while IFS=',' read -r name temp util mem_used mem_total power power_limit; do
            echo "  GPU: $(echo $name | xargs)"
            echo "  Temperature: ${temp}°C"
            echo "  Utilization: ${util}%"
            echo "  Memory: ${mem_used}MB / ${mem_total}MB"
            echo "  Power: ${power}W / ${power_limit}W"
        done
    else
        echo "  No GPU detected"
    fi
    echo ""
}

show_processes() {
    echo -e "${BLUE}┌────────────────────────────────────────────────────────────────┐${NC}"
    echo -e "${BLUE}│ Service Processes${NC}                                             ${BLUE}│${NC}"
    echo -e "${BLUE}└────────────────────────────────────────────────────────────────┘${NC}"

    if command -v pm2 &> /dev/null; then
        pm2 list 2>/dev/null || echo "  No processes running"
    else
        echo "  PM2 not available"
    fi
    echo ""
}

show_network() {
    echo -e "${BLUE}┌────────────────────────────────────────────────────────────────┐${NC}"
    echo -e "${BLUE}│ Network Status${NC}                                                ${BLUE}│${NC}"
    echo -e "${BLUE}└────────────────────────────────────────────────────────────────┘${NC}"

    # Check if miner port is listening
    local axon_port="${BT_AXON_PORT:-8091}"
    if netstat -tuln 2>/dev/null | grep -q ":$axon_port"; then
        echo -e "  ${GREEN}✓${NC} Axon port $axon_port: LISTENING"
    else
        echo -e "  ${YELLOW}✗${NC} Axon port $axon_port: NOT LISTENING"
    fi

    # Redis connection
    local redis_host="${REDIS_HOST:-redis}"
    local redis_port="${REDIS_PORT:-6379}"
    if timeout 2 bash -c "echo > /dev/tcp/$redis_host/$redis_port" 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} Redis ($redis_host:$redis_port): CONNECTED"
    else
        echo -e "  ${YELLOW}✗${NC} Redis ($redis_host:$redis_port): DISCONNECTED"
    fi
    echo ""
}

show_disk_usage() {
    echo -e "${BLUE}┌────────────────────────────────────────────────────────────────┐${NC}"
    echo -e "${BLUE}│ Disk Usage${NC}                                                    ${BLUE}│${NC}"
    echo -e "${BLUE}└────────────────────────────────────────────────────────────────┘${NC}"

    local temp_dir="${TEMP_DIR:-/data/temp}"
    local output_dir="${OUTPUT_DIR:-/data/output}"

    echo "  Temp Directory ($temp_dir):"
    df -h "$temp_dir" 2>/dev/null | tail -1 | awk '{print "    Used: "$3" / "$2" ("$5")"}'
    local temp_files=$(find "$temp_dir" -type f 2>/dev/null | wc -l)
    echo "    Files: $temp_files"

    echo ""
    echo "  Output Directory ($output_dir):"
    df -h "$output_dir" 2>/dev/null | tail -1 | awk '{print "    Used: "$3" / "$2" ("$5")"}'
    local output_files=$(find "$output_dir" -type f 2>/dev/null | wc -l)
    echo "    Files: $output_files"
    echo ""
}

show_memory() {
    echo -e "${BLUE}┌────────────────────────────────────────────────────────────────┐${NC}"
    echo -e "${BLUE}│ Memory Usage${NC}                                                  ${BLUE}│${NC}"
    echo -e "${BLUE}└────────────────────────────────────────────────────────────────┘${NC}"

    free -h 2>/dev/null | tail -2 | head -1
    free -h 2>/dev/null | tail -1
    echo ""
}

show_recent_logs() {
    echo -e "${BLUE}┌────────────────────────────────────────────────────────────────┐${NC}"
    echo -e "${BLUE}│ Recent Miner Logs (last 5 lines)${NC}                              ${BLUE}│${NC}"
    echo -e "${BLUE}└────────────────────────────────────────────────────────────────┘${NC}"

    local log_file="${LOG_DIR:-/data/logs}/miner-out.log"
    if [[ -f "$log_file" ]]; then
        tail -5 "$log_file" 2>/dev/null | tail -5
    else
        echo "  No log file found at $log_file"
    fi
    echo ""
}

show_stats() {
    echo -e "${BLUE}┌────────────────────────────────────────────────────────────────┐${NC}"
    echo -e "${BLUE}│ Processing Stats${NC}                                              ${BLUE}│${NC}"
    echo -e "${BLUE}└────────────────────────────────────────────────────────────────┘${NC}"

    # Count processed videos
    local output_dir="${OUTPUT_DIR:-/data/output}"
    local video_count=$(find "$output_dir" -type f \( -name "*.mp4" -o -name "*.mkv" -o -name "*.avi" \) 2>/dev/null | wc -l)
    echo "  Total processed videos: $video_count"

    # Recent activity (files modified in last hour)
    local recent_count=$(find "$output_dir" -type f -mmin -60 2>/dev/null | wc -l)
    echo "  Recent uploads (last hour): $recent_count"

    # Temp files (active processing)
    local temp_dir="${TEMP_DIR:-/data/temp}"
    local temp_count=$(find "$temp_dir" -type f 2>/dev/null | wc -l)
    echo "  Active temp files: $temp_count"
    echo ""
}

show_config() {
    echo -e "${BLUE}┌────────────────────────────────────────────────────────────────┐${NC}"
    echo -e "${BLUE}│ Configuration${NC}                                                 ${BLUE}│${NC}"
    echo -e "${BLUE}└────────────────────────────────────────────────────────────────┘${NC}"

    echo "  Network: ${BT_NETWORK:-finney}"
    echo "  NetUID: ${BT_NETUID:-85}"
    echo "  Axon Port: ${BT_AXON_PORT:-8091}"
    echo "  Wallet: ${WALLET_NAME:-default}/${WALLET_HOTKEY:-default}"
    echo "  Chutes Enabled: ${USE_CHUTES:-false}"
    echo ""
}

# ==============================================================================
# Main Loop
# ==============================================================================

REFRESH_INTERVAL=${1:-5}
CONTINUOUS=false

if [[ "$REFRESH_INTERVAL" == "once" ]]; then
    REFRESH_INTERVAL=0
elif [[ "$2" == "continuous" ]]; then
    CONTINUOUS=true
fi

if [[ $REFRESH_INTERVAL -gt 0 ]] || [[ "$CONTINUOUS" == true ]]; then
    # Continuous mode
    while true; do
        show_header
        show_gpu_status
        show_processes
        show_network
        show_disk_usage
        show_memory
        show_stats
        show_config

        echo -e "${CYAN}Press Ctrl+C to exit. Refreshing every ${REFRESH_INTERVAL}s...${NC}"
        sleep "$REFRESH_INTERVAL"
    done
else
    # One-shot mode
    show_header
    show_gpu_status
    show_processes
    show_network
    show_disk_usage
    show_memory
    show_stats
    show_config
    show_recent_logs
fi
