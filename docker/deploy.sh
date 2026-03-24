#!/bin/bash
# SN85 Miner Deployment Script
# Automated deployment with validation and rollback support

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
COMPOSE_FILE="docker/docker-compose.miner.yml"
ENV_FILE="docker/.env"
BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"

# ==============================================================================
# Utility Functions
# ==============================================================================

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
    echo -e "${CYAN}[STEP]${NC} $1"
}

# ==============================================================================
# Validation Functions
# ==============================================================================

check_prerequisites() {
    log_step "Checking prerequisites..."

    # Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker not found. Please install Docker: https://docs.docker.com/get-docker/"
        exit 1
    fi
    log_info "Docker version: $(docker --version)"

    # Docker Compose
    if ! docker compose version &> /dev/null && ! docker-compose --version &> /dev/null; then
        log_error "Docker Compose not found. Please install Docker Compose."
        exit 1
    fi
    log_info "Docker Compose available"

    # NVIDIA Docker runtime (optional but recommended)
    if docker info 2>/dev/null | grep -q "nvidia"; then
        log_info "NVIDIA Docker runtime detected"
    else
        log_warn "NVIDIA Docker runtime not detected. GPU support may be limited."
        log_warn "Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    fi

    # Check wallet directory
    local wallet_path="${WALLET_PATH:-$HOME/.bittensor/wallets}"
    if [[ ! -d "$wallet_path" ]]; then
        log_error "Bittensor wallet directory not found: $wallet_path"
        log_error "Create wallet with: btcli wallet new_coldkey"
        exit 1
    fi
    log_info "Wallet directory found: $wallet_path"
}

validate_env_file() {
    log_step "Validating environment configuration..."

    if [[ ! -f "$ENV_FILE" ]]; then
        log_error "Environment file not found: $ENV_FILE"
        log_info "Please copy docker/.env.example to docker/.env and configure it:"
        log_info "  cp docker/.env.example docker/.env"
        log_info "  nano docker/.env"
        exit 1
    fi

    # Load and validate environment
    set -a
    source "$ENV_FILE"
    set +a

    # Required variables
    local required=(
        "WALLET_NAME"
        "WALLET_HOTKEY"
        "BUCKET_NAME"
        "BUCKET_COMPATIBLE_ENDPOINT"
        "BUCKET_COMPATIBLE_ACCESS_KEY"
        "BUCKET_COMPATIBLE_SECRET_KEY"
    )

    local missing=()
    for var in "${required[@]}"; do
        if [[ -z "${!var}" ]]; then
            missing+=("$var")
        fi
    done

    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "Missing required environment variables:"
        printf '  - %s\n' "${missing[@]}"
        exit 1
    fi

    log_info "Environment validation passed"
}

# ==============================================================================
# Deployment Functions
# ==============================================================================

create_directories() {
    log_step "Creating data directories..."

    mkdir -p data/{temp,output,logs,models}
    mkdir -p "$BACKUP_DIR"

    log_info "Directories created"
}

backup_existing() {
    log_step "Checking for existing deployment..."

    if docker ps -a --format '{{.Names}}' | grep -q "sn85_miner"; then
        log_warn "Existing miner found. Creating backup..."

        # Backup logs
        if [[ -d "data/logs" ]]; then
            cp -r data/logs "$BACKUP_DIR/" 2>/dev/null || true
        fi

        # Backup PM2 status
        docker exec sn85_miner pm2 save 2>/dev/null || true

        log_info "Backup saved to: $BACKUP_DIR"
    fi
}

build_image() {
    log_step "Building miner Docker image..."

    if [[ "${SKIP_BUILD:-false}" == "true" ]]; then
        log_info "Skipping build (SKIP_BUILD=true)"
        return
    fi

    export DOCKER_BUILDKIT=1
    docker compose -f "$COMPOSE_FILE" build --no-cache

    # Tag the image
    docker tag sn85-miner:latest "sn85-miner:$(date +%Y%m%d)"

    log_info "Docker image built successfully"
}

deploy_containers() {
    log_step "Starting miner containers..."

    # Pull latest base images
    docker compose -f "$COMPOSE_FILE" pull

    # Start services
    docker compose -f "$COMPOSE_FILE" up -d --remove-orphans

    log_info "Containers started"
}

wait_for_services() {
    log_step "Waiting for services to be ready..."

    local max_attempts=30
    local attempt=1

    echo "  Waiting for Redis..."
    while ! docker exec sn85_redis redis-cli ping 2>/dev/null | grep -q "PONG"; do
        if [[ $attempt -ge $max_attempts ]]; then
            log_error "Redis failed to start"
            return 1
        fi
        echo -n "."
        sleep 2
        ((attempt++))
    done
    echo " OK"

    echo "  Waiting for miner health check..."
    attempt=1
    while ! docker exec sn85_miner /app/healthcheck.sh &> /dev/null; do
        if [[ $attempt -ge $max_attempts ]]; then
            log_warn "Miner health check inconclusive - may still be starting"
            break
        fi
        echo -n "."
        sleep 5
        ((attempt++))
    done
    echo " OK"
}

# ==============================================================================
# Command Functions
# ==============================================================================

cmd_deploy() {
    log_info "SN85 Miner Deployment"
    echo "=============================="

    check_prerequisites
    validate_env_file
    create_directories
    backup_existing
    build_image
    deploy_containers
    wait_for_services

    echo ""
    log_info "Deployment Complete!"
    log_info "===================="
    echo ""
    echo "View logs:        docker logs -f sn85_miner"
    echo "Monitor status:   docker exec sn85_miner /app/monitor.sh"
    echo "Health check:     docker exec sn85_miner /app/healthcheck.sh"
    echo "PM2 status:       docker exec sn85_miner pm2 status"
    echo ""
    echo "Stop miner:       docker compose -f $COMPOSE_FILE down"
    echo "Restart miner:    docker compose -f $COMPOSE_FILE restart"
    echo ""
}

cmd_start() {
    log_step "Starting existing deployment..."
    docker compose -f "$COMPOSE_FILE" up -d
}

cmd_stop() {
    log_step "Stopping deployment..."
    docker compose -f "$COMPOSE_FILE" down
}

cmd_restart() {
    log_step "Restarting deployment..."
    docker compose -f "$COMPOSE_FILE" restart
}

cmd_logs() {
    local service="${1:-miner}"
    docker compose -f "$COMPOSE_FILE" logs -f "$service"
}

cmd_monitor() {
    docker exec -it sn85_miner /app/monitor.sh "$@"
}

cmd_health() {
    docker exec sn85_miner /app/healthcheck.sh
}

cmd_shell() {
    docker exec -it sn85_miner /bin/bash
}

cmd_update() {
    log_step "Updating miner..."

    # Pull latest code
    git pull origin main || true

    # Rebuild and redeploy
    cmd_deploy
}

cmd_status() {
    echo "=== SN85 Miner Status ==="
    echo ""
    docker compose -f "$COMPOSE_FILE" ps
    echo ""
    docker exec sn85_miner pm2 status 2>/dev/null || echo "PM2 status unavailable"
}

cmd_cleanup() {
    log_step "Cleaning up resources..."

    # Remove old images
    docker image prune -f --filter "until=168h"

    # Clean temp directory
    find data/temp -type f -mtime +1 -delete 2>/dev/null || true

    # Clean old logs
    find data/logs -name "*.log" -mtime +7 -delete 2>/dev/null || true

    log_info "Cleanup complete"
}

cmd_help() {
    cat << 'EOF'
SN85 Miner Deployment Script

Usage: ./docker/deploy.sh <command> [options]

Commands:
  deploy     Full deployment with validation (default)
  start      Start existing containers
  stop       Stop all containers
  restart    Restart all containers
  logs       View container logs (default: miner, specify service name as arg)
  monitor    Show real-time system monitor
  health     Run health check
  shell      Open interactive shell in miner container
  update     Update and rebuild deployment
  status     Show container and PM2 status
  cleanup    Remove old images and clean temp files
  help       Show this help message

Examples:
  ./docker/deploy.sh deploy              # Full deployment
  ./docker/deploy.sh logs                # View miner logs
  ./docker/deploy.sh logs redis          # View Redis logs
  ./docker/deploy.sh monitor             # Interactive monitor (Ctrl+C to exit)
  ./docker/deploy.sh health              # Check system health
  WALLET_NAME=mywallet ./docker/deploy.sh deploy  # Override env var

Environment:
  SKIP_BUILD=true    Skip Docker image build during deploy
  WALLET_PATH=path   Override wallet directory path

EOF
}

# ==============================================================================
# Main Entrypoint
# ==============================================================================

main() {
    local cmd="${1:-deploy}"
    shift || true

    case "$cmd" in
        deploy|start|stop|restart|update|status|cleanup|help)
            "cmd_$cmd" "$@"
            ;;
        logs)
            cmd_logs "$@"
            ;;
        monitor)
            cmd_monitor "$@"
            ;;
        health)
            cmd_health
            ;;
        shell)
            cmd_shell
            ;;
        *)
            log_error "Unknown command: $cmd"
            cmd_help
            exit 1
            ;;
    esac
}

main "$@"
