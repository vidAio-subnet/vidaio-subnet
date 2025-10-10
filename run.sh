#!/bin/bash

script="neurons/validator.py"
autoRunLoc=$(readlink -f "$0")
proc_name="video-validator"
args=()
version_location="./vidaio_subnet_core/__init__.py"
version="__version__"
old_args=$@
subnet=85
restart_video_scheduler=true

if ! command -v pm2 &> /dev/null
then
    echo "pm2 could not be found. To install see: https://pm2.keymetrics.io/docs/usage/quick-start/"
    exit 1
fi

version_less_than_or_equal() {
    [ "$1" = "$(echo -e "$1\n$2" | sort -V | head -n1)" ]
}

version_less_than() {
    [ "$1" = "$2" ] && return 1 || version_less_than_or_equal $1 $2
}

read_version_value() {
    while IFS= read -r line; do
        if [[ "$line" == *"$version"* ]]; then
            local value=$(echo "$line" | awk -F '=' '{print $2}' | tr -d ' ')
            strip_quotes "$value"
            return 0
        fi
    done < "$version_location"
    echo ""
}

check_package_installed() {
    local package_name="$1"
    if ! command -v "$package_name" &> /dev/null; then
        echo "Error: '$package_name' is not installed."
        echo "Installing '$package_name'..."
        sudo apt-get install -y "$package_name"
    fi
}

strip_quotes() {
    local input="$1"
    local stripped="${input#\"}"
    stripped="${stripped%\"}"
    echo "$stripped"
}

check_variable_value_on_github() {
    local repo="vidaio-subnet/vidaio-subnet"
    local branch="$1"
    local file_path="vidaio_subnet_core/__init__.py"
    local variable="$2"


    local content
    content=$(curl -s "https://api.github.com/repos/$repo/contents/$file_path?ref=$branch" | jq -r '.content' | base64 --decode)

    if [[ $? -ne 0 || -z "$content" ]]; then
        echo "Error: Could not retrieve file content from GitHub."
        return 1
    fi

    local value
    value=$(echo "$content" | grep "$variable" | awk -F '=' '{print $2}' | tr -d ' ')

    # Replace the strip_quotes call with direct quote removal
    echo "$value" | tr -d '"' | tr -d "'"
}

ensure_process() {
    local name="$1"
    local cmd="$2"
    local restart_flag="$3"   # "true" or "false"

    if pm2 describe "$name" &>/dev/null; then
        echo "Process '$name' already running."
        if [[ "$restart_flag" == "true" ]]; then
            echo "Reloading $name..."
            pm2 reload "$name"
        fi
    else
        echo "Starting $name..."
        pm2 start bash --name "$name" -- -c "$cmd"
    fi
}

ensure_config_process() {
    local config="$1"
    local name="$2"
    local restart_flag="$3"   # "true" or "false"

    if pm2 describe "$name" &>/dev/null; then
        echo "Process '$name' already running (from $config)."
        if [[ "$restart_flag" == "true" ]]; then
            echo "Reloading $name from config..."
            pm2 startOrReload "$config"
        fi
    else
        echo "Starting $name from config..."
        pm2 startOrReload "$config"
    fi
}

while [[ $# -gt 0 ]]; do
  arg="$1"
  if [[ "$arg" == -* ]]; then
    if [[ $# -gt 1 && "$2" != -* ]]; then
        if [[ "$arg" == "--script" ]]; then
            script="$2"
            shift 2
        elif [[ "$arg" == "--subnet" ]]; then
            subnet="$2"
            args+=("'--netuid'")
            args+=("'$2'")
            shift 2
        elif [[ "$arg" == "--netuid" ]]; then
            subnet="$2"
            args+=("'$arg'")
            args+=("'$2'")
            shift 2
        else
            args+=("'$arg'")
            args+=("'$2'")
            shift 2
        fi
    else
        if [[ "$arg" == "--no-video-scheduler" ]]; then
            restart_video_scheduler=false
            shift
        else
            args+=("'$arg'")
            shift
        fi
    fi
  else
    args+=("'$arg'")
    shift
  fi
done

branch=$(git branch --show-current)
echo "Watching branch: $branch"
echo "Reapplying git stash"
git stash pop
echo "PM2 process name: $proc_name"
if [[ -n "$subnet" ]]; then
    echo "Subnet: $subnet"
fi
echo "Restart video scheduler: $restart_video_scheduler"

current_version=$(read_version_value)

if pm2 status | grep -q $proc_name; then
    echo "The script is already running with pm2. Stopping and restarting..."
    pm2 delete $proc_name
fi

echo "Running $script with the following PM2 config:"

# Ensure netuid is always included in args
netuid_found=false
for arg in "${args[@]}"; do
    if [[ "$arg" == "'--netuid'" ]]; then
        netuid_found=true
        break
    fi
done

if [[ "$netuid_found" == "false" ]]; then
    args+=("'--netuid'")
    args+=("'$subnet'")
fi

joined_args=$(printf "%s," "${args[@]}")
joined_args=${joined_args%,}

echo "module.exports = {
  apps : [{
    name   : '$proc_name',
    script : '$script',
    interpreter: 'python3',
    min_uptime: '5m',
    max_restarts: '5',
    args: [$joined_args],
    cwd: '$(pwd)',
    env: {
      PYTHONPATH: '.'
    }
  }]
}" > app.config.js

cat app.config.js
ensure_config_process "app.config.js" "$proc_name" "true"


check_package_installed "jq"

# 🚀 START THE ADDITIONAL PM2 PROCESSES
ensure_process "scoring_endpoint" "bash -c 'PYTHONPATH=. python services/scoring/server.py'" "true"
ensure_process "video_scheduler_worker" "bash -c 'PYTHONPATH=. python services/video_scheduler/worker.py'" "$restart_video_scheduler"
ensure_process "video_scheduler_endpoint" "bash -c 'PYTHONPATH=. python services/video_scheduler/server.py'" "$restart_video_scheduler"
ensure_process "organic-gateway" "bash -c 'PYTHONPATH=. python services/organic_gateway/server.py'" "true"

# Auto-update loop
last_restart_time=$(date +%s)
restart_interval=$((30 * 3600))  # 30 hours in seconds

while true; do
    current_time=$(date +%s)
    time_since_last_restart=$((current_time - last_restart_time))

    if [ -d "./.git" ]; then
        latest_version=$(check_variable_value_on_github "$branch" "$version")

        if version_less_than $current_version $latest_version; then
            echo "Latest version: $latest_version"
            echo "Current version: $current_version"

            git stash
            if git pull origin "$branch"; then
                echo "New version detected. Updating..."
                pip install -e .

                current_version=$(read_version_value)
                last_restart_time=$current_time

                echo "Restarting script..."
                exec ./$(basename "$0") $old_args
            else
                echo "** Will not update **"
                echo "You have uncommitted changes. Please stash them using 'git stash'."
            fi
        else
            echo "** No update needed **"
            echo "$current_version is up-to-date with $latest_version."

            if [ $time_since_last_restart -ge $restart_interval ]; then
                echo "30 hours passed. Performing periodic PM2 restart..."
                pm2 restart scoring_endpoint
                if [[ "$restart_video_scheduler" == "true" ]]; then
                    pm2 restart video_scheduler_worker
                    pm2 restart video_scheduler_endpoint
                fi
                pm2 restart video-validator

                last_restart_time=$current_time
                echo "Periodic restart completed."
            fi
        fi
    else
        echo "The installation does not appear to be from Git. Please install from source at https://github.com/vidaio-subnet/vidaio-subnet."
    fi

    sleep 1800  # Sleep 30 minutes before checking again
done
